import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '0'

import utils 
from train_autoencoder import *
import torch

summary = torch.load('../trained_model/autoencoder.pth.tar')
 
args = utils.get_args()
encoder = Encoder(args)
encoder.load_state_dict(summary['encoder'])

print('Success')
