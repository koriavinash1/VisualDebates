from inspect import ArgSpec
import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2

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
    arg.add_argument("--only_properties", type=bool, default=False, 
                    help='Only plot convergence and argument properties')
    arg.add_argument("--threshold", type=float, default=0.25, 
                    help='threhold value for extracting binary mask')
    return arg.parse_args()


normalize = lambda x: np.uint8(255*(x - x.min())/(x.max() - x.min()))


def get_arguments(x, z, arguments, threshold=0.25):
    size = np.asarray(x.shape[1:-1])

    return_arguments = [] #nargs x bs
    for ai in range(arguments.shape[1]):
        F = normalize(np.sum(z*arguments[ai], 1))
        Ftilde = cv2.resize(F, size, cv2.INTER_AREA)
        Bmask = Ftilde > threshold
        return_arguments.append(Bmask*x)
    return np.swapaxes(np.array(return_arguments), 0, 1) # bs x nargs


def main_image(args):
    plot_dir = args.plot_dir
    epoch = args.epoch 
    
    logs_data = {
                'images': [],
                'z': [],
                'z_idx': [],
                'jpred': [],
                'labels': [],
                'outcome': [],
                'arguments': {},
                'visual_arguments': {},
                'predictions':{},
            }

    for ai in range(args.nagents):
        name = str(ai) + '_' + args.name
        # read in pickle files
        with open(os.path.join(plot_dir, "{}_logs_{}.p".format(name, epoch)), "rb") as f:
            data = pickle.load(f)
            logs_data['images'] = data['imgs'].transpose(0, 2, 3, 1)
            logs_data['z'] = data['zs']
            logs_data['z_idx'] = data['zs_idx']
            logs_data['jpred'] = data['jpreds']
            logs_data['outcome'] = data['dpreds']
            logs_data['labels'] = data['Ys']
            logs_data['arguments'][ai] = data['arguments']
            logs_data['predictions'][ai] = data['preds']
            
        logs_data['visual_arguments'][ai] = get_arguments(logs_data['images'], 
                                                            data['zs'],
                                                            data['arguments'],
                                                            args.threshold)

    narguments = len(logs_data['arguments'][0][0])
    num_imgs = logs_data['images'][0].shape[0]
    img_shape = np.asarray([logs_data['images'][0][0].shape[1:]])
    
    # denormalize coordinates
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
        title = '$Y={}$, $\mathcal{J}(x)={}$, $Debate(x)={}$, '.format(logs_data['images'][imgidx],
                                                                    logs_data['jpred'][imgidx],
                                                                    logs_data['outcome'][imgidx])
        if logs_data['labels'][imgidx] != logs_data['jpred'][imgidx]: continue

        for aidx in range(args.nagents):
            title += '$\mathcal{P}^{}(x)={}$, '.format(aidx+1, 
                                            logs_data['predictions'][aidx][imgidx])
                                 
            for argidx in range(ncols):
                if argidx == 0:
                    ax = plt.subplot2grid(shape=(nrows, ncols), 
                                            loc=(imgidx, 0), 
                                            colspan=args.nagents,
                                            rowspan=args.nagents)
                    ax.imshow(normalize(logs_data['images'][imgidx]),cmap='gray')
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    ax.set_title(title)
                    
                elif argidx < args.nagents:
                    continue
                else:
                    # print (imgidx, aidx, argidx, imgidx//args.nagents + aidx)
                    ax = plt.subplot2grid(shape=(nrows, ncols), 
                                            loc=(imgidx + aidx, argidx))

                    ax.imshow(logs_data['visual_arguments'][aidx][imgidx, argidx],cmap='gray')
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    ax.set_title('$\mathcal{A}^{}_{}'.format(aidx +1, argidx - args.nagents + 1))

    # save as png
    path = os.path.join(args.plot_dir, 'pngs/')
    path = os.path.join(path, 'epoch_{}.png'.format(epoch))
    plt.savefig(path, bbox_inches='tight')

if __name__ == "__main__":
    args = parse_arguments()
    name = args.name
    os.makedirs(os.path.join(args.plot_dir, 'pngs'), exist_ok=True)
    main_image(args)
