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


def _load_weights_only(ckpt_path, nets, device):
    """Seed a fresh run from another run's weights (no optimiser, no epoch).

    v12 fine-tunes the v11 denoiser on a different input distribution, so we
    want its parameters but not its optimiser moments (they encode gradients
    for the old inputs) and not its epoch counter (the LR schedule should start
    over). Tolerates a state_dict saved with or without the DataParallel
    "module." prefix.

    Args:
        ckpt_path: file to read.
        nets: dict of {key in the checkpoint: module to load into}.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    for key, net in nets.items():
        if key not in ckpt:
            print(f'  init_from: no "{key}" in {ckpt_path}, skipping')
            continue
        sd = ckpt[key]
        wrapped = any(k.startswith('module.') for k in net.state_dict())
        saved_wrapped = any(k.startswith('module.') for k in sd)
        if wrapped and not saved_wrapped:
            sd = {'module.' + k: v for k, v in sd.items()}
        elif saved_wrapped and not wrapped:
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        net.load_state_dict(sd)
        print(f'  init_from: loaded {key} from {ckpt_path}')
    return nets


class ModelUNet:
    """UNet 단독 학습을 위한 클래스 (Discriminator 없이 L1 Loss만 사용)"""

    def __init__(self, configs,
                 save_iter,
                 ckpt_dir, result_dir,
                 device, writer, gpu_id,
                 loader_train, loader_val, img_size,
                 z_slice=608,
                 test_epoch=None,
                 init_from=None):

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
        self.init_from = init_from

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
            # Resuming this run always wins over seeding a new one.
            netG, optimG, st_epoch = self.load(self.ckpt_dir, netG, optimG, mode='train')
        elif self.init_from:
            _load_weights_only(self.init_from, {'netG': netG}, self.device)

        # Chosen once, before training, so every preview shows the same slices.
        preview = preview_samples(self.loader_train, self.loader_val)

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

            ## validation phase
            with torch.no_grad():
                netG.eval()

                loss_l1_val = []

                for i, data in enumerate(self.loader_val, 1):
                    # The dataset yields (input, label, ...) -- these two were
                    # swapped here, so loss_l1_val measured the FBP against a
                    # denoised prediction of the label. Any UNet val curve from
                    # before this fix is not comparable with one after it.
                    input = data[0].to(self.device)
                    label = data[1].to(self.device)

                    # forward
                    output = netG(input)

                    loss_l1 = fn_L1(output, label)
                    loss_l1_val += [loss_l1.item()]

                self.writer.add_scalar('loss_l1_val', mean(loss_l1_val), epoch)

            if epoch == 1 or (epoch % self.save_iter) == 0:
                save_preview(netG, epoch, max_l, preview,
                             self.device, self.result_dir, self.writer)

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


def fixed_sample(loader, index):
    """A deterministic sample straight from the dataset, augmentation off.

    The preview has to show the SAME slice every epoch or the images cannot be
    compared across epochs -- and the train loader both shuffles and applies
    random flips/rotations. Reading the dataset directly with `transform`
    temporarily disabled sidesteps both.

    Returns (input, label, name) with a batch axis, or None if the split is
    empty.
    """
    dataset = loader.dataset
    if len(dataset) == 0:
        return None
    index = index % len(dataset)

    transform, dataset.transform = getattr(dataset, 'transform', None), None
    try:
        inp, lab, name = dataset[index][:3]
    finally:
        dataset.transform = transform

    return inp.unsqueeze(0), lab.unsqueeze(0), name


def preview_samples(loader_train, loader_val):
    """One fixed train slice and one fixed val slice, picked once.

    Deliberately no test slice: judging checkpoints by how the test images look
    leaks the held-out set into model selection. Pick the epoch on val, then
    run step6 on test once.
    """
    samples = []
    for split, loader in (('train', loader_train), ('val', loader_val)):
        if loader is None:
            continue
        # Mid-dataset rather than index 0 -- the first slices of a volume are
        # usually near-empty air.
        got = fixed_sample(loader, len(loader.dataset) // 2)
        if got is not None:
            samples.append((split, got))
    return samples


def save_preview(netG, epoch, max_l, samples, device, result_dir, writer):
    """Save/log the fixed previews: input, output, label, |output-label|.

    The difference panel is the point of this figure -- v12 expects the error
    to change character (long streaks -> local texture), and that is visible in
    the residual long before it moves the L1 number much.
    """
    if not samples:
        return

    was_training = netG.training
    netG.eval()

    # The panels are square, so each row needs its column width PLUS room for a
    # title -- sized to the width, titles land on the row above. constrained
    # layout then keeps them clear of the suptitle as well.
    col_w = 13 / 4
    fig, axes = plt.subplots(len(samples), 4,
                             figsize=(13, (col_w + 0.55) * len(samples) + 0.4),
                             squeeze=False, layout='constrained')
    with torch.no_grad():
        for row, (split, (inp, lab, name)) in enumerate(samples):
            out = netG(inp.to(device))

            img_in = denormalize(inp[0, 0].cpu().numpy(), max_l)
            img_out = denormalize(out[0, 0].cpu().numpy(), max_l)
            img_lab = denormalize(lab[0, 0].cpu().numpy(), max_l)
            diff = np.abs(img_out - img_lab)

            # input/output/label share one window so the three are directly
            # comparable; the residual gets its own. Scaled to the 99th
            # percentile, not the max -- a couple of hot pixels at a metal edge
            # would otherwise push everything else to black and hide exactly
            # the low-level texture this panel exists to show.
            vmin, vmax = float(img_lab.min()), float(img_lab.max())
            dmax = float(np.percentile(diff, 99)) or float(diff.max()) or 1.0
            panels = [(f'{name} input', img_in, vmin, vmax, 'gray'),
                      ('output', img_out, vmin, vmax, 'gray'),
                      ('label', img_lab, vmin, vmax, 'gray'),
                      (f'|diff| (p99)  MAE={diff.mean():.4g}',
                       diff, 0.0, dmax, 'inferno')]

            for col, (title, img, lo, hi, cmap) in enumerate(panels):
                ax = axes[row][col]
                ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi)
                ax.set_title(f'[{split}] {title}' if col == 0 else title,
                             fontsize=9)
                ax.axis('off')

    fig.suptitle(f'epoch {epoch}', fontsize=11)

    os.makedirs(result_dir, exist_ok=True)
    fig.savefig(os.path.join(result_dir, f'preview_epoch{epoch:04d}.png'),
                dpi=110)

    # Also into TensorBoard, so epochs can be compared with the step slider
    # instead of opening files one by one.
    if writer is not None:
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        writer.add_image('preview', rgb, epoch, dataformats='HWC')

    plt.close(fig)

    if was_training:
        netG.train()


class Model:
    def __init__(self, configs,
                 save_iter,
                 ckpt_dir,result_dir,
                 device, writer, gpu_id,
                 loader_train,loader_val,img_size,
                 z_slice = 608,
                 test_epoch=None,
                 init_from=None):

        
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
        self.init_from = init_from

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

    def preview_samples(self):
        return preview_samples(self.loader_train, self.loader_val)

    def save_preview(self, netG, epoch, max_l, samples):
        return save_preview(netG, epoch, max_l, samples,
                            self.device, self.result_dir, self.writer)

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
            # Resuming this run always wins over seeding a new one.
            netG, netD, optimG, optimD, st_epoch = self.load(self.ckpt_dir, netG, netD, optimG, optimD, mode='train')
        elif self.init_from:
            _load_weights_only(self.init_from, {'netG': netG, 'netD': netD}, self.device)


        # if isinstance(netG, DDP):
        #     print("Model is wrapped with DDP.")

        # else:
        #     print("Model is a standard nn.Module (no parallel wrapper).")


        # Chosen once, before training, so every preview shows the same slices.
        preview = self.preview_samples()

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


            ## validation phase
            with torch.no_grad():
                netG.eval()
                netD.eval()

                loss_G_l1_val = []
                loss_G_gan_val = []
                loss_D_real_val = []
                loss_D_fake_val = []

                for i, data in enumerate(self.loader_val, 1):
                    # The dataset yields (input, label, ...) -- these two were
                    # swapped here, so every val loss was measured on the model
                    # running backwards: fed the 720-view label and scored
                    # against the sparse FBP. Val curves from before this fix
                    # are not comparable with ones after it.
                    input = data[0].to(self.device)
                    label = data[1].to(self.device)

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
                
                        


                        
            # Epoch 1 too, so there is a "before" frame to compare against
            # rather than waiting a whole save_iter for the first image.
            if epoch == 1 or (epoch % self.save_iter) == 0:
                self.save_preview(netG, epoch, max_l, preview)

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