from inspect import ArgSpec
import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.animation as animation

from utils import bounding_box

## python3 plot_glimpses.py --plot_dir=./plots/ram_6_8x8_2_2/ --epoch=111  -- use this to run and save gif
def parse_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument("--plot_dir", type=str,
                     help="path to directory containing pickle dumps")
    arg.add_argument("--epoch", type=int, default=111,
                     help="epoch of desired plot")
    arg.add_argument("--nagents", type=int, default=2,
                     help="Total number of agents in a debate")
    arg.add_argument("--name", type=str, default='plots', 
                    help='Name of the plots')
    arg.add_argument('--patch_size', type=int, default=32, 
                    help='size of extracted patch at highest res')
    arg.add_argument('--glimpse_scale', type=int, default=2, 
                    help='scale factor for glimpse')
    arg.add_argument('--nglimpses', type=int, default=10, 
                    help='# of arguments used in debate')

    return arg.parse_args()

normalize = lambda x: np.uint8(255*(x - x.min())/(x.max() - x.min()))

def main_image(args):
    plot_dir = args.plot_dir
    epoch = args.epoch 

    # grab useful params
    patch_size = args.patch_size
    glimpse_scale = args.glimpse_scale
    num_patches = args.nglimpses
    
    images = {}; locations = {}; predictions = {}
    for ai in range(args.nagents):
        name = str(ai) + '_' + args.name
        # read in pickle files
        with open(os.path.join(plot_dir, "{}g_{}.p".format(name, epoch)), "rb") as f:
            images[ai] = pickle.load(f)
        with open(os.path.join(plot_dir, "{}l_{}.p".format(name, epoch)), "rb") as f:
            locations[ai] = pickle.load(f)
        with open(os.path.join(plot_dir, "{}Ys_{}.p".format(name, epoch)), "rb") as f:
            labels = pickle.load(f)
        with open(os.path.join(plot_dir, "{}preds_{}.p".format(name, epoch)), "rb") as f:
            predictions[ai] = pickle.load(f)
        with open(os.path.join(plot_dir, "{}jpreds_{}.p".format(name, epoch)), "rb") as f:
            classifierPred = pickle.load(f)
        with open(os.path.join(plot_dir, "{}dpreds_{}.p".format(name, epoch)), "rb") as f:
            debate_outcome = pickle.load(f)

    narguments = len(locations[0][0])
    num_imgs = images[0].shape[0]
    img_shape = np.asarray([images[0][0].shape[1:]])
    
    # denormalize coordinates
    coords = {ai: [0.5 * ((l + 1.0) * img_shape) for l in locations[ai]] for ai in locations.keys()}
    nrows = args.nagents*num_imgs
    ncols = narguments + args.nagents

    fig = plt.figure()
    fig.set_figheight(nrows*3)
    fig.set_figwidth(ncols*3)
    fig.set_dpi(100)

    print (ncols, nrows, narguments, num_imgs, img_shape)
    # plot base image
    color = ['r', 'b', 'g', 'c', 'm', 'y']
    for imgidx in range(0, nrows, args.nagents):
        title = 'Y:{}, CPred: {}, DOutcome:{}, '.format(labels[imgidx//args.nagents],
                                    classifierPred[imgidx//args.nagents],
                                    debate_outcome[imgidx//args.nagents])
        if labels[imgidx//args.nagents] != classifierPred[imgidx//args.nagents]: continue

        for aidx in range(args.nagents):
            title += 'P{}: {}, '.format(aidx+1, predictions[aidx][imgidx//args.nagents])
                                 
            for argidx in range(ncols):
                if argidx == 0:
                    ax = plt.subplot2grid(shape=(nrows, ncols), 
                                            loc=(imgidx, 0), 
                                            colspan=args.nagents,
                                            rowspan=args.nagents)
                    ax.imshow(normalize(images[aidx][imgidx//args.nagents].transpose(1,2,0)),cmap='gray')
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    ax.set_title(title)
                    
                elif argidx < args.nagents:
                    continue
                else:
                    # print (imgidx, aidx, argidx, imgidx//args.nagents + aidx)
                    ax = plt.subplot2grid(shape=(nrows, ncols), 
                                            loc=(imgidx + aidx, argidx))
                    ax.imshow(normalize(images[aidx][imgidx//args.nagents].transpose(1,2,0)),cmap='gray')
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    ax.set_title('Player:{} Arg:{}'.format(aidx +1, argidx - args.nagents + 1))

                    c = coords[aidx][imgidx//args.nagents][argidx - args.nagents]

                    for l in range(num_patches):
                        rect = bounding_box(
                            c[0], c[1], 
                            patch_size*(glimpse_scale**l), 
                            color[argidx - args.nagents]
                            )
                        ax.add_patch(rect)
    # save as png
    path = os.path.join(args.plot_dir, 'pngs/')
    path = os.path.join(path, 'epoch_{}.png'.format(epoch))
    plt.savefig(path, bbox_inches='tight')

if __name__ == "__main__":
    args = parse_arguments()
    name = args.name
    os.makedirs(os.path.join(args.plot_dir, 'pngs'), exist_ok=True)
    main_image(args)
