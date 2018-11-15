import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import numpy as np
from random import choice
import utils


def recon_loss(g_out, labels, args):
    """
    Args:
        [g_out]     FloatTensor, (B, C, W, H), Output of the generator.
        [labels]    FloatTensor, (B, C, W, H), Ground truth images.
    Rets:
        Reconstruction loss with both L1 and L2.
    """
    raise NotImplementedError()


class Encoder(nn.Module):

    def __init__(self, args):
        super(Encoder, self).__init__()
        raise NotImplementedError()


    def forward(self, x):
        raise NotImplementedError()


    def load_model(self, filename):
        """ Load the pretrained weights stored in file [filename] into the model.
        Args:
            [filename]  The filename of the checkpoint saved from the main procedure
                        (i.e. the 'autoencoder.pth.tar' file below.)
        Usage:
            enet = Encoder(args)
            enet.load_model('autoencoder.pth.tar')
            # Here [enet] should be loaded with weights from file 'autoencoder.pth.tar'
        """
        raise NotImplementedError()


class Decoder(nn.Module):

    def __init__(self, args):
        super(Decoder, self).__init__()
        raise NotImplementedError()


    def forward(self, z, c=None):
        raise NotImplementedError()


    def load_model(self, filename):
        """ Load the pretrained weights stored in file [filename] into the model.
        Args:
            [filename]  The filename of the checkpoint saved from the main procedure
                        (i.e. the 'autoencoder.pth.tar' file below.)
        Usage:
            dnet = Decoder(args)
            dnet.load_model('autoencoder.pth.tar')
            # Here [dnet] should be loaded with weights from file 'autoencoder.pth.tar'
        """
        raise NotImplementedError()


def train_batch(input_data, encoder, decoder, enc_opt, dec_opt, args, writer=None):
    """Train the AutoEncoder for one iteration (i.e. forward, backward, and update
       weights for one batch of data)
    Args:
        [input_data]    Input tensors tuple from the data loader.
        [encoder]       Encoder module.
        [decoder]       Decoder module.
        [enc_opt]       Optimizer to update encoder's weights.
        [dec_opt]       Optimizer to update decoder's weights.
        [args]          Commandline arguments.
        [writer]        Tensorboard writer (optional)
    Rets:
        [loss]  (float) Reconstruction loss of the batch (before the update).
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

    decoder = Decoder(args)
    encoder = Encoder(args)
    if args.cuda:
        decoder = decoder.cuda()
        encoder = encoder.cuda()

    dec_opt = torch.optim.Adam(
            decoder.parameters(), lr=args.lr_dec, betas=(args.beta_1, args.beta_2))
    enc_opt = torch.optim.Adam(
            encoder.parameters(), lr=args.lr_enc, betas=(args.beta_1, args.beta_2))

    step = 0
    for epoch in range(args.nepoch):
        for input_data in loader:
            l = train_batch(input_data, encoder, decoder, enc_opt, dec_opt, args, writer=writer)
            step += 1
            if step % 50 == 0:
                print("Step:%d\tLoss:%2.5f"%(step, l))

    utils.save_checkpoint('autoencoder.pth.tar', **{
        'decoder' : decoder.state_dict(),
        'encoder' : encoder.state_dict(),
        'dec_opt' : dec_opt.state_dict(),
        'enc_opt' : enc_opt.state_dict()
    })


    def get_z():
        z = torch.rand(args.batch_size, args.nz)
        if args.cuda:
            z = z.cuda(async=True)
        return z

    gen_img = sample(decoder, 60000, get_z, args)
    np.save('autoencoder_out.npy', gen_img)


