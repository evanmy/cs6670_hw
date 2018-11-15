"""
Implement a generative model that could beat both of the baseline in all metrics.
"""
from tensorboardX import SummaryWriter
import utils

if __name__ == "__main__":
    args = utils.get_args()
    loaders = utils.get_loader(args)
    writer = SummaryWriter()

    generator = None
    utils.save_checkpoint('mygan.pth.tar', **{
        'generator':generator
    })

    gen_img = None
    np.save('mygan_out.npy', gen_img)

