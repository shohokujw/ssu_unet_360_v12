# from model import *
from model_v2 import *
# from dataset import *

import torch
import torch.nn as nn
import os
import pickle
import numpy as np


from tqdm import tqdm
import matplotlib.pyplot as plt
from statistics import mean


class ModelUNet:
    """UNet 단독 학습을 위한 클래스 (Discriminator 없이 L1 Loss만 사용)"""

    def __init__(self, configs,
                 save_iter,
                 ckpt_dir, result_dir,
                 device, writer, gpu_id,
                 loader_train, loader_val, img_size,
                 z_slice=608,
                 test_epoch=None):

        self.ckpt_dir = ckpt_dir
        self.result_dir = result_dir

        self.num_epoch = configs['n_epochs']
        self.batch_size = configs['batch_size']

        self.lr = configs['lr']
        self.wgt_decay = configs.get('wgt_decay', 0.0001)

        self.norm = configs['norm']
        self.num_down = configs['num_down']
        self.dropout = configs.get('dropout', 0.5)
        self.init_gain = configs.get('init_gain', 0.02)

        self.save_iter = save_iter
        self.test_epoch = test_epoch

        self.device = device
        self.gpu_id = gpu_id

        self.writer = writer
        self.loader_train = loader_train
        self.loader_val = loader_val

        self.img_size = img_size
        self.z_slice = z_slice

    def save(self, ckpt_dir, netG, optimG, epoch):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        torch.save({'netG': netG.state_dict(),
                    'optimG': optimG.state_dict()},
                   '%s/model_epoch%04d.pth' % (ckpt_dir, epoch))

    def load(self, ckpt_dir, netG, optimG=None, epoch=None, mode='train'):
        if not epoch:
            ckpt = os.listdir(ckpt_dir)
            ckpt.sort()
            epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

        map_loc = f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu"
        dict_net = torch.load('%s/model_epoch%04d.pth' % (ckpt_dir, epoch),
                              map_location=map_loc)

        print('Loaded %dth network' % epoch)

        if mode == 'train':
            netG.load_state_dict(dict_net['netG'])
            optimG.load_state_dict(dict_net['optimG'])
            print("\nTrain continue from %dth........\n" % epoch)
            return netG, optimG, epoch

        elif mode == 'test':
            netG.load_state_dict(dict_net['netG'])
            return netG, epoch

    def train(self):
        _, max_l = normalize_param(self.img_size)

        # UNet만 생성 (Discriminator 없음)
        netG = UNet(nch_in=1, nch_out=1, num_down=self.num_down, dropout=self.dropout)

        init_net(netG, init_type='normal', init_gain=self.init_gain, gpu_ids=[self.gpu_id])

        paramsG = netG.parameters()

        # L1 Loss만 사용 (GAN Loss 없음)
        fn_L1 = nn.L1Loss().to(self.device)

        # Adam optimizer with weight decay
        optimG = torch.optim.Adam(paramsG, lr=self.lr, weight_decay=self.wgt_decay)

        st_epoch = 0
        ckpts = os.listdir(self.ckpt_dir)
        if len(ckpts) > 0:
            netG, optimG, st_epoch = self.load(self.ckpt_dir, netG, optimG, mode='train')

        for epoch in range(st_epoch + 1, self.num_epoch + 1):
            ## training phase
            netG.train()

            loss_l1_train = []

            for i, data in enumerate(tqdm(self.loader_train, desc=f"Epoch {epoch}/{self.num_epoch}", leave=True)):
                input = data[0].to(self.device)
                label = data[1].to(self.device)
                name = data[2]

                # forward
                output = netG(input)

                # backward
                optimG.zero_grad()

                loss_l1 = fn_L1(output, label)
                loss_l1.backward()
                optimG.step()

                # get losses
                loss_l1_train += [loss_l1.item()]

            self.writer.add_scalar('loss_l1_train', mean(loss_l1_train), epoch)

            input = input.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
            output = output.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
            label = label.to('cpu').detach().numpy().transpose(0, 2, 3, 1)

            plt.subplot(131)
            plt.title("output")
            plt.imshow(denormalize(output[0][:, :, 0], max_l), cmap='gray')
            plt.axis("off")

            plt.subplot(132)
            plt.title(name[0])
            plt.imshow(denormalize(input[0][:, :, 0], max_l), cmap='gray')
            plt.axis("off")

            plt.subplot(133)
            plt.title("label")
            plt.imshow(denormalize(label[0][:, :, 0], max_l), cmap='gray')
            plt.axis("off")

            plt.savefig(os.path.join(self.result_dir, f"output_train{epoch}.png"))
            plt.close()

            ## validation phase
            with torch.no_grad():
                netG.eval()

                loss_l1_val = []

                for i, data in enumerate(self.loader_val, 1):
                    input = data[1].to(self.device)
                    label = data[0].to(self.device)

                    # forward
                    output = netG(input)

                    loss_l1 = fn_L1(output, label)
                    loss_l1_val += [loss_l1.item()]

                self.writer.add_scalar('loss_l1_val', mean(loss_l1_val), epoch)

            if (epoch % self.save_iter) == 0:
                self.save(self.ckpt_dir, netG, optimG, epoch)

    def test(self):
        max_h, max_l = normalize_param(self.img_size)
        max_lh = 0.5 * (max_h + max_l)

        netG = UNet(nch_in=1, nch_out=1, num_down=self.num_down, dropout=self.dropout)
        init_net(netG, init_type='normal', init_gain=self.init_gain, gpu_ids=[self.gpu_id])

        netG, _ = self.load(ckpt_dir=self.ckpt_dir, netG=netG, epoch=self.test_epoch, mode='test')

        id = 0
        infer_lh = []

        with torch.no_grad():
            netG.eval()

            for i, data in enumerate(self.loader_val, 1):
                input = data[0].to(self.device)
                pickle_idx = data[3]
                name = data[2]

                output = netG(input)
                output = output.to('cpu').detach().numpy().transpose(0, 2, 3, 1)

                change_indices = np.where(np.diff(pickle_idx) != 0)[0]

                if change_indices.size == 0:
                    for idx in range(output.shape[0]):
                        infer_lh.append(output[idx][:, :, 0])
                        id += 1

                    if id == self.z_slice:
                        fin_lh = np.stack(infer_lh, axis=0)
                        fin_lh = denormalize(fin_lh, max_value=max_lh)

                        if self.img_size != 512:
                            fin_lh = linear_interpolation(fin_lh)

                        data = {'fin_lh': fin_lh}
                        path_save = os.path.join(self.result_dir, f'FIN_{name[idx]}.pkl')

                        with open(path_save, 'wb') as fid:
                            pickle.dump(data, fid)

                        infer_lh = []
                        id = 0
                else:
                    for idx in range(change_indices[0] + 1):
                        infer_lh.append(output[idx][:, :, 0])
                        id += 1

                    fin_lh = np.stack(infer_lh, axis=0)
                    fin_lh = denormalize(fin_lh, max_value=max_lh)

                    if self.img_size != 512:
                        fin_lh = linear_interpolation(fin_lh)

                    data = {'fin_lh': fin_lh}
                    path_save = os.path.join(self.result_dir, f'FIN_{name[idx]}.pkl')

                    with open(path_save, 'wb') as fid:
                        pickle.dump(data, fid)

                    infer_lh = []
                    id = 0

                    for idx in range(change_indices[0] + 1, output.shape[0]):
                        infer_lh.append(output[idx][:, :, 0])
                        id += 1


class Model:
    def __init__(self, configs, 
                 save_iter,
                 ckpt_dir,result_dir,
                 device, writer, gpu_id,
                 loader_train,loader_val,img_size,
                 z_slice = 608,
                 test_epoch=None):

        
        self.ckpt_dir = ckpt_dir

        self.result_dir = result_dir

        self.num_epoch = configs['n_epochs']
        self.batch_size = configs['batch_size']

        self.lr_G = configs['lr_G']
        self.lr_D = configs['lr_D']

        self.wgt_l1 = configs['wgt_l1']
        self.wgt_gan = configs['wgt_gan']

        self.beta1 = configs['beta1']
        self.beta2 = configs['beta2']
        
        self.norm = configs['norm']
        
        self.num_down_G = configs['num_down_G']
        self.num_down_D = configs['num_down_D']
        
        self.init_gain = configs['init_gain']
        
        
        self.save_iter = save_iter

        self.test_epoch = test_epoch

        self.device = device
        self.gpu_id = gpu_id
        
        self.writer = writer
        self.loader_train = loader_train
        self.loader_val = loader_val
        
        self.img_size = img_size
        self.z_slice = z_slice
        
    def save(self, ckpt_dir, netG, netD, optimG, optimD, epoch):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        torch.save({'netG': netG.state_dict(), 'netD': netD.state_dict(),
                    'optimG': optimG.state_dict(), 'optimD': optimD.state_dict()},
                   '%s/model_epoch%04d.pth' % (ckpt_dir, epoch))

    def load(self, ckpt_dir, netG, netD=[], optimG=[], optimD=[], epoch=[], mode='train'):
        if not epoch:
            ckpt = os.listdir(ckpt_dir)
            ckpt.sort()
            epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

        map_loc = f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu"
        dict_net = torch.load('%s/model_epoch%04d.pth' % (ckpt_dir, epoch), map_location=map_loc)

        print('Loaded %dth network' % epoch)

        if mode == 'train':
            netG.load_state_dict(dict_net['netG'])
            netD.load_state_dict(dict_net['netD'])
            optimG.load_state_dict(dict_net['optimG'])
            optimD.load_state_dict(dict_net['optimD'])
            
            print("\nTrain continue from %dth........\n"% epoch)
            return netG, netD, optimG, optimD, epoch

        elif mode == 'test':
            netG.load_state_dict(dict_net['netG'])

            return netG, epoch

    def train(self):


        _, max_l = normalize_param(self.img_size)
        
        
        netG = UNet(nch_in=1, nch_out=1, num_down=self.num_down_G)
        netD = Discriminator(nch_in=2, num_down=self.num_down_D)
        
        
        init_net(netG, init_type='normal', init_gain=0.02, gpu_ids=[self.gpu_id])
        init_net(netD, init_type='normal', init_gain=0.02, gpu_ids=[self.gpu_id])
        
        
        paramsG = netG.parameters()
        paramsD = netD.parameters()
        
        
        fn_L1 = nn.L1Loss().to(self.device) # L1
        fn_GAN = nn.BCEWithLogitsLoss().to(self.device)

        optimG = torch.optim.Adam(paramsG, lr=self.lr_G, betas=(self.beta1, self.beta2))
        optimD = torch.optim.Adam(paramsD, lr=self.lr_D, betas=(self.beta1, self.beta2))
        
        
        
        st_epoch = 0
        ckpts = os.listdir(self.ckpt_dir)
        # print(self.rank)
        if len(ckpts) > 0:
            netG, netD, optimG, optimD, st_epoch = self.load(self.ckpt_dir, netG, netD, optimG, optimD, mode='train')


        # if isinstance(netG, DDP):
        #     print("Model is wrapped with DDP.")

        # else:
        #     print("Model is a standard nn.Module (no parallel wrapper).")


        for epoch in range(st_epoch + 1, self.num_epoch + 1):
            ## training phase
            netG.train()
            netD.train()

            loss_G_l1_train = []
            loss_G_gan_train = []
            loss_D_real_train = []
            loss_D_fake_train = []

            # for i, data in enumerate(loader_train, 1):
            for i, data in enumerate(tqdm(self.loader_train, desc=f"Epoch {epoch}/{self.num_epoch}", leave=True)):
                # TODO
                input = data[0].to(self.device)
                label = data[1].to(self.device)
                name  = data[2]

                # forward netG
                output = netG(input)

                # backward netD
                fake = torch.cat([input, output], dim=1)
                real = torch.cat([input, label], dim=1)

                set_requires_grad(netD, True)
                optimD.zero_grad()

                pred_real = netD(real)
                pred_fake = netD(fake.detach())

                loss_D_real = fn_GAN(pred_real, torch.ones_like(pred_real))
                loss_D_fake = fn_GAN(pred_fake, torch.zeros_like(pred_fake))
                loss_D = 0.5 * (loss_D_real + loss_D_fake)

                loss_D.backward()
                optimD.step()

                # backward netG
                fake = torch.cat([input, output], dim=1)

                set_requires_grad(netD, False)
                optimG.zero_grad()

                pred_fake = netD(fake)

                loss_G_gan = fn_GAN(pred_fake, torch.ones_like(pred_fake))
                loss_G_l1 = fn_L1(output, label)
                loss_G = (self.wgt_l1 * loss_G_l1) + (self.wgt_gan * loss_G_gan)

                loss_G.backward()
                optimG.step()

                # get losses
                loss_G_l1_train += [loss_G_l1.item()]
                loss_G_gan_train += [loss_G_gan.item()]
                loss_D_fake_train += [loss_D_fake.item()]
                loss_D_real_train += [loss_D_real.item()]


        
            self.writer.add_scalar('loss_G_l1_train', mean(loss_G_l1_train), epoch)
            self.writer.add_scalar('loss_G_gan_train', mean(loss_G_gan_train), epoch)
            self.writer.add_scalar('loss_D_fake_train', mean(loss_D_fake_train), epoch)
            self.writer.add_scalar('loss_D_real_train', mean(loss_D_real_train), epoch)


            input = input.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
            output = output.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
            label = label.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
            
            
            
            
            
            


            plt.subplot(131)
            plt.title("output")
            plt.imshow(denormalize(output[0][:,:,0],max_l) , cmap='gray')  # Grayscale 이미지
            # plt.imshow((output_num[0] * 0.5) + 0.5,cmap='gray')  # Grayscale 이미지
            plt.axis("off")  # 축 제거

            plt.subplot(132)
            plt.title(name[0])
            plt.imshow((denormalize(input[0][:,:,0],max_l) ),cmap='gray')  # Grayscale 이미지
            plt.axis("off")  # 축 제거



            plt.subplot(133)
            plt.title("label")
            plt.imshow(denormalize(label[0][:,:,0] ,max_l),cmap='gray')  # Grayscale 이미지
            plt.axis("off")  # 축 제거


            # 이미지 저장
            plt.savefig(os.path.join(self.result_dir,f"output_train{epoch}.png"))



            ## validation phase
            with torch.no_grad():
                netG.eval()
                netD.eval()

                loss_G_l1_val = []
                loss_G_gan_val = []
                loss_D_real_val = []
                loss_D_fake_val = []

                for i, data in enumerate(self.loader_val, 1):


                    input = data[1].to(self.device)
                    label = data[0].to(self.device)

                    # forward netG
                    output = netG(input)

                    fake = torch.cat([input, output], dim=1)
                    real = torch.cat([input, label], dim=1)

                    # forward netD
                    pred_fake = netD(fake)
                    pred_real = netD(real)

                    loss_D_real = fn_GAN(pred_real, torch.ones_like(pred_real))
                    loss_D_fake = fn_GAN(pred_fake, torch.zeros_like(pred_fake))
                    loss_D = 0.5 * (loss_D_real + loss_D_fake)

                    loss_G_gan = fn_GAN(pred_fake, torch.ones_like(pred_fake))
                    loss_G_l1 = fn_L1(output, label)
                    loss_G = (self.wgt_l1 * loss_G_l1) + (self.wgt_gan * loss_G_gan)

                    loss_G_l1_val += [loss_G_l1.item()]
                    loss_G_gan_val += [loss_G_gan.item()]
                    loss_D_real_val += [loss_D_real.item()]
                    loss_D_fake_val += [loss_D_fake.item()]

                self.writer.add_scalar('loss_G_l1_val', mean(loss_G_l1_val), epoch)
                self.writer.add_scalar('loss_G_gan_val', mean(loss_G_gan_val), epoch)
                self.writer.add_scalar('loss_D_fake_val', mean(loss_D_fake_val), epoch)
                self.writer.add_scalar('loss_D_real_val', mean(loss_D_real_val), epoch)
                
                        


                        
            if (epoch % self.save_iter) == 0:
                self.save(self.ckpt_dir, netG, netD, optimG, optimD, epoch)


    def test(self):


        max_h, max_l = normalize_param(self.img_size)
        max_lh = 0.5*(max_h + max_l)
        netG = UNet(nch_in=1, nch_out=1, num_down=self.num_down_G)
        init_net(netG, init_type='normal', init_gain=0.02, gpu_ids=[self.gpu_id])


        netG, _ = self.load(ckpt_dir=self.ckpt_dir, netG=netG, epoch=self.test_epoch,mode='test')

        id = 0 
        infer_lh = []
        
        with torch.no_grad():
            netG.eval()

            for i, data in enumerate(self.loader_val, 1):
                input = data[0].to(self.device)
                pickle_idx = data[3]
                name = data[2]                

                output = netG(input)
                output = output.to('cpu').detach().numpy().transpose(0, 2, 3, 1)                                 
                                    
                change_indices = np.where(np.diff(pickle_idx) != 0 )[0]
                # 1 batch
                if change_indices.size == 0:
                    for idx in range(output.shape[0]):
                        
                        infer_lh.append(output[idx][:,:,0])

                        id += 1
                    if id == self.z_slice:
                        fin_lh = np.stack(infer_lh,axis=0)
                        
                        fin_lh = denormalize(fin_lh,max_value=max_lh)
                        
                        if self.img_size != 512 :
                            
                          fin_lh = linear_interpolation(fin_lh)
                        
                        data = {'fin_lh': fin_lh}
                        
                        path_save = os.path.join(self.result_dir, f'FIN_{name[idx]}.pkl')
                        
                        with open(path_save, 'wb') as fid:
                            pickle.dump(data, fid)   
                        infer_lh=[]
                        id = 0
                else : 
                    for idx in range(change_indices[0]+1):
                        infer_lh.append(output[idx][:,:,0])

                        id += 1
                        
                    fin_lh = np.stack(infer_lh,axis=0)
                    
                    
                    fin_lh = denormalize(fin_lh,max_value=max_lh)
                        
                    if self.img_size != 512 :
                            
                        fin_lh = linear_interpolation(fin_lh)                
                    
                    data = {'fin_lh': fin_lh}
                    path_save = os.path.join(self.result_dir, f'FIN_{name[idx]}.pkl')
                    with open(path_save, 'wb') as fid:
                        pickle.dump(data, fid)   

                    infer_lh=[]
                    id = 0
                    
                    for idx in range(change_indices[0]+1,output.shape[0]):
                        infer_lh.append(output[idx][:,:,0])
                        
                        id += 1
                        



def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def append_index(dir_result, fileset, step=False):
    index_path = os.path.join(dir_result, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        for key, value in fileset.items():
            index.write("<th>%s</th>" % key)
        index.write('</tr>')

    # for fileset in filesets:
    index.write("<tr>")

    if step:
        index.write("<td>%d</td>" % fileset["step"])
    index.write("<td>%s</td>" % fileset["name"])

    del fileset['name']

    for key, value in fileset.items():
        index.write("<td><img src='images/%s'></td>" % value)

    index.write("</tr>")
    return index_path


def add_plot(output, label, writer, epoch=[], ylabel='Density', xlabel='Radius', namescope=[]):
    fig, ax = plt.subplots()

    ax.plot(output.transpose(1, 0).detach().numpy(), '-')
    ax.plot(label.transpose(1, 0).detach().numpy(), '--')

    ax.set_xlim(0, 400)

    ax.grid(True)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    writer.add_figure(namescope, fig, epoch)