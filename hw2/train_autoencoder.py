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
    lmbd1 = args.recon_l1_weight
    lmbd2 = args.recon_l2_weight
    out = lmbd1*(g_out-labels)**2 + lmbd2*torch.abs(g_out, labels)
    raise torch.mean(out)

class Encoder(nn.Module):

    def __init__(self, args):
        super(Encoder, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(args.nc, args.nef, 
                              kernel_size= args.e_ksize, stride=2, padding=1, 
                              bias=False)
        
        self.conv1 = nn.Conv2d(args.nef, args.nef*2, 
                               kernel_size= args.e_ksize, stride=2, padding=1, 
                               bias=False)        
        self.bn1 = nn.BatchNorm2d(args.nef*2)        
        
        self.conv1 = nn.Conv2d(args.nef*2, args.nef*4, 
                               kernel_size= args.e_ksize, stride=2, padding=1, 
                               bias=False)   
        self.bn2 = nn.BatchNorm2d(args.nef*4) 
        
        self.conv3 = nn.Conv2d(args.nef*4, args.nef*8, 
                               kernel_size= args.e_ksize, stride=2, padding=1, 
                               bias=False)   
        self.bn3 = nn.BatchNorm2d(args.nef*8)
        
        self.fc = nn.Linear(args.nef*8, args.nz)

    def forward(self, x):
        
        x = self.relu(self.conv(x))
        x = self.bn1(self.conv1(x))
        x = self.relu(x)
        
        x = self.bn2(self.conv2(x))
        x = self.relu(x)
        
        x = self.bn3(self.conv3(x))
        x = self.relu(x)
        
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        
        return x
        
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
#         summary = torch.load(filename)
#         summary['encoder']
        raise NotImplementedError()


class Decoder(nn.Module):

    def __init__(self, args):
        super(Decoder, self).__init__()
        
        self.relu = nn.ReLU()
        
        self.up_conv1 = nn.ConvTranspose2d(args.ngf*4, args.ngf*2, 
                                           kernel_size=args.g_ksize, padding=1, stride=2, 
                                           bias=False)
        self.bn1 = nn.BatchNorm2d(args.ngf*2) 
        
        self.up_conv2 = nn.ConvTranspose2d(args.ngf*2, args.ngf, 
                                           kernel_size= args.g_ksize, padding=1, stride=2, 
                                           bias=False)
        self.bn2 = nn.BatchNorm2d(args.ngf) 
        
        self.up_conv3 = nn.ConvTranspose2d(args.ngf, args.nc, 
                                           kernel_size= args.g_ksize, padding=1, stride=2, 
                                           bias=False)
        

    def forward(self, z, c=None):
        x = nn.BatchNorm1d(z)
        x = x.view(x.size(0), 4*args.ngf, 4, 4)
        x = self.bn1(self.up_conv1(x))
        x = self.relu(x)
        
        x = self.bn2(self.up_conv2(x))
        x = self.relu(x)
        
        x = F.tanh(self.up_conv2(x))
        return x

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
    
    enc_opt.zero_grad()
    dec_opt.zero_grad()
    
    z = encoder(input_data)
    out = decoder(z)
  
    loss = recon_loss(out, input_data, args)
    loss.backward()
    
    enc_opt.step()
    dec_opt.step()
    
    return loss

def sample(model, n, sampler, args):
    """ Sample [n] images from [model] using noise created by the sampler.
    Args:
        [model]     Generator model that takes noise and output images.
        [n]         Number of images to sample.
        [sampler]   [sampler()] will return a batch of noise.
    Rets:
        [imgs]      (B, C, W, H) Float, numpy array.
    """
    
    s =[]
    for i in range(n)
        z = sampler
        out = model(z)
        s += [out]
        
    raise torch.cat(s,0)


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


