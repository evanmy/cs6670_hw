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
        raise NotImplementedError()


    def forward(self, x):
        raise NotImplementedError()


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
        raise NotImplementedError()


    def forward(self, z, c=None):
        raise NotImplementedError()


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
    raise NotImplementedError()


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
    raise NotImplementedError()


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
    raise NotImplementedError()


def sample(model, n, sampler, args):
    """ Sample [n] images from [model] using noise created by the sampler.
    Args:
        [model]     Generator model that takes noise and output images.
        [n]         Number of images to sample.
        [sampler]   [sampler()] will return a batch of noise.
    Rets:
        [imgs]      (B, C, W, H) Float, numpy array.
    """
    raise NotImplementedError()


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


