import torch
import torchvision
import numpy as np
from train_clf import resume_model
import tqdm
import scipy.linalg as la
import utils


def IS(gen_imgs, model, args):
    """ Compute the Inception Score using [model] to replace the inception model
        NOTE: do not using batch/splits for this implementation.
    Aggs:
        [gen_imgs]  Generated images, numpy array (N, C, W, H)
        [model]     A model that could take a batch of image and produce a logits
                    NOTE: the logits is before softmax
        [args]       Commandline arguments.
    Rets:
        [score]     float, the inception score.
    """
    raise NotImplementedError()


def FID(gen_imgs, gtr_imgs, feat_extr, args, eps=1e-8):
    """ Compute FID scores for [gen_imgs] using [gtr_imgs] as ground truth and
        [feat_extr] as feature extractor.
    Args:
        [gen_imgs]   Generated images.
        [gtr_imgs]   Ground truth images.
        [feat_extr]  Feature extractor.
        [args]       Commandline arguments.
        [eps]        Small perturbation used for numerical stability.
    Rets:
        [score]     float, the FID score.
    """
    raise NotImplementedError()


############################################################
# DO NOT MODIFY CODES BELOW
############################################################
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', help='mnist only.')
    parser.add_argument('--dataroot', type=str, default='.', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--gen_img_file', type=str, default='autoencoder_out.npy',
            help='Numpy file that contains the generated images.')
    parser.add_argument('--score_type', type=str, default='IS', help='IS|FID.')
    parser.add_argument('--clf_ckpt', type=str, default='clf.pth.tar',
            help='Checkpoint filename for the classifier or feature extractor.')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    args = parser.parse_args()

    clf, _ = resume_model(args.clf_ckpt, args)
    gen_imgs = np.load(args.gen_img_file)
    if args.score_type == 'IS':
        score = IS(gen_imgs, clf, args)
    elif args.score_type == 'FID':
        loader = utils.get_loader(args)['train']
        gtr_imgs = []
        for x,_ in tqdm.tqdm(loader):
            gtr_imgs.append(x.detach().cpu().numpy())
        gtr_imgs = np.concatenate(gtr_imgs, axis=0)
        np.save('train_data.npy', gtr_imgs)
        score = FID(gen_imgs, gtr_imgs, clf, args, eps=1e-10)
    else:
        raise Exception("Invalid score type:%s"%(args.score_type))

    print(score)
