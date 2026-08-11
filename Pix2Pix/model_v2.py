from layer import *

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

class UNet(nn.Module):
    def __init__(self, nch_in=2, nch_out=2, nch_ker=64, norm='bnorm', num_down=8, dropout=0.5):
        """
        num_down controls how many times to downsample (encoder) 
        and then upsample (decoder). For a 256x256 input, 8 downsamplings
        give a similar architecture to the original Pix2Pix/CycleGAN UNet.
        """
        super(UNet, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm
        self.num_down = num_down
        self.dropout = dropout

        # Decide bias usage based on normalization
        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        #-----------------------------------------
        # Create encoder layers
        #-----------------------------------------
        self.enc_layers = nn.ModuleList()
        current_in_channels = nch_in
        current_out_channels = nch_ker

        channels = []
        
        
        for i in range(num_down):
            # Use leaky ReLU(0.2) for all but the last encoder, which might be 0.0
            # to mimic the original code's final layer
            if i < (num_down - 1):
                relu_val = 0.2
            else:
                relu_val = 0.0

            # Add CNR2d block
            self.enc_layers.append(
                CNR2d(current_in_channels,
                      current_out_channels,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      norm=self.norm,
                      relu=relu_val,
                      drop=[])
            )
            channels.append(current_in_channels)
            # channels.append(current_out_channels)
            
            current_in_channels = current_out_channels
            # Double the channels up to a max of 8 * nch_ker
            if current_out_channels < 8 * nch_ker:
                current_out_channels *= 2

        #-----------------------------------------
        # Create decoder layers
        #-----------------------------------------
        self.bottle_neck_layer = DECNR2d(current_out_channels, current_out_channels, stride=2, norm=self.norm, relu=0.0, drop=self.dropout)
        self.dec_layers = nn.ModuleList()
        # for i in range(num_down-1):
        for i in reversed(range(num_down-1)):
            
            if i > num_down-4:
                if self.dropout > 0:
                    drop_val = self.dropout
                else:
                    drop_val = []
            else:
                drop_val = []

            if i ==0:
                relu_val = []
                norm_dec = []
                bias = False
            else:
                norm_dec = self.norm
                relu_val = 0.0
                bias = []
                
            current_out_channels = channels[i]
            current_in_channels = channels[i+1]*2    
                
            # We'll temporarily set in/out channels; in forward we adapt
            self.dec_layers.append(
                DECNR2d(current_in_channels, current_out_channels,  # dummy channels, replaced at forward pass
                        stride=2,
                        norm=norm_dec,
                        relu=relu_val,
                        drop=drop_val,
                        bias=bias)
            )


           


    def forward(self, x):
        skips = []
        # Encoder forward
        for enc in self.enc_layers:
            x = enc(x)
            skips.append(x)

        
        x = self.bottle_neck_layer(x)
        
        for idx, dec in enumerate(self.dec_layers):
            

            x = dec(torch.cat([skips[-(idx+2)], x], dim=1))
        
        x = torch.tanh(x)
        
        return x
        



#=====================================================
# Flexible Discriminator with variable downsamplings
#=====================================================
class Discriminator(nn.Module):
    def __init__(self, nch_in=4, nch_ker=64, norm='bnorm', num_down=4):
        """
        num_down controls how many times the discriminator downsamples
        for patch-level classification. For a 256x256 input, 4 downsamplings
        is standard in PatchGAN (70x70 patch).
        """
        super(Discriminator, self).__init__()

        self.nch_in = nch_in
        self.nch_ker = nch_ker
        self.norm = norm
        self.num_down = num_down

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        # We will store the downsampling blocks in a loop.
        # The final layer will output a 1-channel map (for real/fake).
        self.layers = nn.ModuleList()

        current_in_channels = nch_in
        current_out_channels = nch_ker
        for i in range(num_down):


            # Add a block
            self.layers.append(
                CNR2d(current_in_channels,
                      current_out_channels,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      norm=self.norm ,
                      relu=0.2,
                      bias=self.bias)
            )

            current_in_channels = current_out_channels
            if current_out_channels < 8 * self.nch_ker:
                current_out_channels *= 2

        # After these downsampling blocks, add a final 1-channel conv (if you prefer):
        # Or if the last block above is your final output, skip it.
        # We'll do a 1x1 layer for the logit:
        self.final = CNR2d(current_in_channels, 1, kernel_size=4, stride=1, padding=1, norm=[],relu=[], bias=False)



    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
        x = self.final(x)
        # out = torch.sigmoid(out)  # or leave it raw for the loss
        return x

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func) 
    # net.module.apply(init_func) # apply the initialization function <init_func>






def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if gpu_ids and torch.cuda.is_available():
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net




def normalize_param(img_size):
    if img_size==512:    
        max_h = 0.51
        max_l = 0.37
        
    elif img_size==256:
        max_h = 0.47
        max_l = 0.37
        
    elif img_size==128:
        max_h = 0.47
        max_l = 0.31
    
    else :    
    
        raise ValueError("wrong image size")

    return max_h,max_l


def denormalize(array, max_value, mean=1.0, std = 0.5):
    
    array = ((array + mean )*std)*max_value
    
    return array



# def init_net(net, device, train_continue ,init_type='normal', init_gain=0.02, gpu_ids=[]): # TODO device 인수추가가
#     """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
#     Parameters:
#         net (network)      -- the network to be initialized
#         init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
#         gain (float)       -- scaling factor for normal, xavier and orthogonal.
#         gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

#     Return an initialized network.
#     """
#     if gpu_ids:
#         assert(torch.cuda.is_available())
#         print("\nok\n")
#         # net.to(gpu_ids[0])
#         net.to(device)
#         # net = DDP(net, gpu_ids)  
#         net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
#     # TODO train_continue 추가
#     if train_continue :
#         return net
#     else :
#         init_weights(net, init_type, init_gain=init_gain)
#         return net

