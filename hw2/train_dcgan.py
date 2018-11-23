import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from random import choice
import utils
import numpy as np


class Discriminator(nn.Module):

    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv2d(args.nc, args.ndf, 
                               kernel_size=4, stride=2, padding=1, 
                               bias=False)

        self.conv2 = nn.Conv2d(args.ndf, args.ndf*2, 
                               kernel_size=4, stride=2, padding=1, 
                               bias=False)    
        
        self.bn2 = nn.BatchNorm2d(args.ndf*2) 
        self.conv3 = nn.Conv2d(args.ndf*2, args.ndf*4, 
                               kernel_size=4, stride=2, padding=1, 
                               bias=False)    
        self.bn3 = nn.BatchNorm2d(args.ndf*4) 
        
        self.conv4 = nn.Conv2d(args.ndf*4, 1, 
                               kernel_size=4, stride=1, padding=0, 
                               bias=False)   

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.lrelu(x)
        
        x = self.bn2(self.conv2(x))
        x = self.lrelu(x)
        
        x = self.bn3(self.conv3(x))
        x = self.lrelu(x)
        
        x = self.conv4(x)
        
        return F.sigmoid(x)


    def load_model(self, filename):
        """ Load the pretrained weights stored in file [filename] into the model.
        Args:
            [filename]  The filename of the checkpoint saved from the main procedure
                        (i.e. the 'dcgan.pth.tar' file below.)
        Usage:
            net = Generator(args)
            net.load_model('dcgan.pth.tar')
            # Here [net] should be loaded with weights from file 'dcgan.pth.tar'
        """
        raise NotImplementedError()


class Generator(nn.Module):

    def __init__(self, args):
        super(Generator, self).__init__()
        self.relu = nn.ReLU()
        
        self.proj = nn.Linear(args.nz, args.ngf*4*4*4)
        self.bn0 = nn.BatchNorm1d(args.ngf*4*4*4)
        
        self.dconv1 = nn.ConvTranspose2d(args.ngf*4, args.ngf*2, 
                                         kernel_size=4, padding=1, stride=2, 
                                         bias=False)
        
        self.bn1 = nn.BatchNorm2d(args.ngf*2) 
        
        self.dconv2 = nn.ConvTranspose2d(args.ngf*2, args.ngf, 
                                         kernel_size=4, padding=1, stride=2, 
                                         bias=False)
        self.bn2 = nn.BatchNorm2d(args.ngf) 
        
        self.dconv3 = nn.ConvTranspose2d(args.ngf, args.nc, 
                                         kernel_size= 4, padding=1, stride=2, 
                                         bias=False)        

    def forward(self, z, c=None):
        x = self.proj(z)
        x = self.bn0(x)
        
        x = x.view(x.size(0), 4*args.ngf, 4, 4)
        x = self.bn1(self.dconv1(x))
        x = self.relu(x)
        
        x = self.bn2(self.dconv2(x))
        x = self.relu(x)
        
        x = F.tanh(self.dconv3(x))
        
        return x

    def load_model(self, filename):
        """ Load the pretrained weights stored in file [filename] into the model.
        Args:
            [filename]  The filename of the checkpoint saved from the main procedure
                        (i.e. the 'dcgan.pth.tar' file below.)
        Usage:
            net = Generator(args)
            net.load_model('dcgan.pth.tar')
            # Here [net] should be loaded with weights from file 'dcgan.pth.tar'
        """
        raise NotImplementedError()


def d_loss(dreal, dfake):
    """
    Args:
        [dreal]  FloatTensor; The output of D_net from real data.
                 (already applied sigmoid)
        [dfake]  FloatTensor; The output of D_net from fake data.
                 (already applied sigmoid)
    Rets:
        DCGAN loss for Discriminator.
    """
    
    real_target = torch.ones(dreal.shape[0]).float()
    fake_target = torch.zeros(dfake.shape[0]).float()
    
    if dreal.is_cuda or dfake.is_cuda():
        real_target = real_target.cuda()
        fake_target = fake_target.cuda()
    
    pred = torch.cat([dreal, dfake]).squeeze()
    target = torch.cat((real_target, fake_target)).squeeze()
    
    return F.binary_cross_entropy(pred, target)

def g_loss(dreal, dfake):
    """
    Args:
        [dreal]  FloatTensor; The output of D_net from real data.
                 (already applied sigmoid)
        [dfake]  FloatTensor; The output of D_net from fake data.
                 (already applied sigmoid)
    Rets:
        DCGAN loss for Generator.
    """
    real_target = torch.ones(dreal.shape[0]).float()
    fool_target = torch.ones(dfake.shape[0]).float()
    
    if dreal.is_cuda or dfake.is_cuda():
        real_target = real_target.cuda()
        fool_target = fool_target.cuda()
    
    pred = torch.cat([dreal, dfake]).squeeze()
    target = torch.cat((real_target, fool_target)).squeeze()
    
    return F.binary_cross_entropy(pred, target)

def train_batch(input_data, g_net, d_net, g_opt, d_opt, sampler, args, writer=None):
    """Train the GAN for one batch iteration.
    Args:
        [input_data]    Input tensors (tuple). Should contain the images and the labels.
        [g_net]         The generator.
        [d_net]         The discriminator.
        [g_opt]         Optimizer that updates [g_net]'s parameters.
        [d_opt]         Optimizer that updates [d_net]'s parameters.
        [sampler]       Function that could output the noise vector for training.
        [args]          Commandline arguments.
        [writer]        Tensorboard writer.
    Rets:
        [L_d]   (float) Discriminator loss (before discriminator's update step).
        [L_g]   (float) Generator loss (before generator's update step)
    """
    g_opt.zero_grad()
    d_opt.zero_grad()
    inputs, labels = input_data
    if args.cuda:
        inputs = inputs.cuda()
        labels = labels.cuda()
        
    '''Gen'''
    z = get_z()
    fake = g_net(z)
    dfake = d_net(fake)
    dreal = d_net(inputs)
    L_g = g_loss(dreal.detach(), dfake)
    L_g.backward()
    g_opt.step()
    
    '''Dis'''
    g_opt.zero_grad()
    d_opt.zero_grad()
    
    z = get_z()
    fake = g_net(z)
    dfake = d_net(fake.detach())
    dreal = d_net(inputs)
    L_d = d_loss(dreal, dfake)
    L_d.backward()
    d_opt.step()
    
    return L_d, L_g
    

def sample(model, n, sampler, args):
    """ Sample [n] images from [model] using noise created by the sampler.
    Args:
        [model]     Generator model that takes noise and output images.
        [n]         Number of images to sample.
        [sampler]   [sampler()] will return a batch of noise.
    Rets:
        [imgs]      (B, C, W, H) Float, numpy array.
    """
    
    s = []
    for i in range(n//args.batch_size):
        z = sampler()
        out = model(z)
        s += [out.detach()]
        
    rmdr = n%args.batch_size
    z = sampler()
    out = model(z)
    s += [out[:rmdr].detach()]
    sampled = torch.cat(s,0).cpu().numpy()
    return sampled


############################################################
# DO NOT MODIFY CODES BELOW
############################################################
if __name__ == "__main__":
    args = utils.get_args()
    loader = utils.get_loader(args)['train']
    writer = SummaryWriter()

    d_net = Discriminator(args)
    g_net = Generator(args)
    if args.cuda:
        d_net = d_net.cuda()
        g_net = g_net.cuda()

    d_opt = torch.optim.Adam(
            d_net.parameters(), lr=args.lr_d, betas=(args.beta_1, args.beta_2))
    g_opt = torch.optim.Adam(
            g_net.parameters(), lr=args.lr_g, betas=(args.beta_1, args.beta_2))

    def get_z():
        z = torch.rand(args.batch_size, args.nz)
        if args.cuda:
            z = z.cuda(async=True)
        return z

    step = 0
    for epoch in range(args.nepoch):
        for input_data in loader:
            l_d, l_g = train_batch(
                input_data, g_net, d_net, g_opt, d_opt, get_z, args, writer=writer)

            step += 1
            print("Step:%d\tLossD:%2.5f\tLossG:%2.5f"%(step, l_d, l_g))

    utils.save_checkpoint('dcgan.pth.tar', **{
        'gnet' : g_net.state_dict(),
        'dnet' : d_net.state_dict(),
        'gopt' : g_opt.state_dict(),
        'dopt' : d_opt.state_dict()
    })

    gen_img = sample(g_net, 60000, get_z, args)
    np.save('dcgan_out.npy', gen_img)


