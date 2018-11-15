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


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()


def train_batch(x, y, clf, opt, args):
    """Training step for one single batch (one forward, one backward, and one update)
    Args:
        [x]     FloatTensor (B, C, W, H), Images
        [y]     LongTensor  (B, 1), labels
        [clf]   Classifier model
        [opt]   Optimizer that updates the weights for the classifier.
        [args]  Commandline arguments.
    Rets:
        [loss]      (float) Loss of the batch (before update).
        [ncorrect]  (float) Number of correct examples in the batch (before update).
    """
    raise NotImplementedError()


def evaluate(clf, loader, args):
    """Evaluate the classfier on the dataset loaded by the [loader]
    Args:
        [clf]       Model to be evaluated.
        [loader]    Dataloader, which will provide the test-set.
        [args]      Commandline arguments.
    Rets:
        Dictionary with following structure: {
            'loss'  : test loss (averaged across samples),
            'acc'   : test accuracy
        }
    """
    raise NotImplementedError()


def resume_model(filename, args):
    """Resume the training (both model and optimizer) with the checkpoint.
    Args:
        [filename]  Str, file name of the checkpoint.
        [args]      Commandline arguments.
    Rets:
        [clf]   CNN with weights loaded with the pretrained weights from checkpoint [filename]
        [opt]   Optimizer with parameters resumed from the checkpoint [filename]
    """
    raise NotImplementedError()


############################################################
# DO NOT MODIFY CODES BELOW
############################################################
if __name__ == "__main__":
    args = utils.get_args()
    loaders = utils.get_loader(args)
    writer = SummaryWriter()

    clf = CNN()
    if args.cuda:
        clf = clf.cuda()
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    if args.resume is not None:
        clf, opt = resume_model(args.resume, args)

    step = 0
    for epoch in range(20):
        print("Epoch:%d"%(epoch+1))
        for x, y in loaders['train']:
            l, c = train_batch(x, y, clf, opt, args)
            acc = c/float(x.size(0))
            step += 1
            if step % 250 == 0:
                print("Step:%d\tLoss:%2.5f\tAcc:%2.5f"%(step, l, acc))

        print("Evaluating testing error:")
        res = evaluate(clf, loaders['test'], args)
        print(res)

    utils.save_checkpoint('clf.pth.tar', **{
        'clf' : clf.state_dict(),
        'opt' : opt.state_dict()
    })

