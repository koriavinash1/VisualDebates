from cProfile import label
from inspect import ArgSpec
import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import stats
import matplotlib.animation as animation
from scipy.interpolate import interp1d
from utils import bounding_box

## python3 plot_glimpses.py --plot_dir=./plots/ram_6_8x8_2_2/ --epoch=111  -- use this to run and save gif
def parse_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument("--plot_dir", type=str,
                     help="path to directory containing pickle dumps")
    arg.add_argument("--start_epoch", type=int, default=0,
                     help="epoch of desired plot")
    arg.add_argument("--nagents", type=int, default=2,
                     help="Total number of agents in a debate")
    arg.add_argument("--name", type=str, default='', 
                    help='Name of the plots')
    arg.add_argument("--properties", type=bool, default=False, 
                    help='Only plot convergence and argument properties')
    arg.add_argument("--threshold", type=float, default=0.85, 
                    help='threhold value for extracting binary mask')
    return arg.parse_args()


normalize = lambda x: np.uint8(255*(x - x.min())/(x.max() - x.min()))



def get_properties(pred, y, arguments):

    def __get_unique__(arguments):
        class_arguments = -1*np.ones((arguments.shape[0], arguments.shape[1]))
        for i, argument in enumerate(np.argmax(arguments, -1)):
            unique_symbols = np.unique(argument)
            class_arguments[i, :len(unique_symbols)] = np.sort(unique_symbols)

        class_arguments = np.sort(class_arguments, 1)
        return class_arguments

    unique_classes = np.unique(y)
    zr_ = []
    for class_ in unique_classes:
        class_idx = y == class_

        p1_class_arguments = __get_unique__(arguments[0][class_idx])
        p2_class_arguments = __get_unique__(arguments[1][class_idx])
        # print("====================")
        # print (p1_class_arguments, p2_class_arguments)
        total_unqiue = len(np.unique(list(p1_class_arguments) + list(p2_class_arguments))) - 1
        d = (1./(total_unqiue*np.sum(class_idx))) * np.sum((p1_class_arguments != p2_class_arguments)*(p1_class_arguments > 0)*(p2_class_arguments > 0))
        zr_.append(d)
    zr_ = np.mean(zr_)
   
   # AH calculation
    # import pdb; pdb.set_trace()
    # print (arguments[0].shape, "*******************")
    ah_ = 0.5*(np.var(arguments[0], axis=-1) + \
                np.var(arguments[1], axis=-1))

   # AD calculation
    ad_ = np.mean((arguments[0] - arguments[1])**2, -1)**0.5
    # print (ah_, ad_)
    # import pdb;pdb.set_trace()

    ah_ = np.mean(ah_)*10**5
    ad_ = np.mean(ad_)*10**5
    # dacc
    dacc = np.sum(pred == y)*1.00/len(y)
    return zr_, ah_, ad_, dacc 



def get_arguments(x, z, arguments, threshold=0.25):
    size = np.asarray(x.shape[1:-1])
    return_arguments = [] #nargs x bs
    for ai in range(arguments.shape[1]):
        idx = np.array([np.random.choice(arguments.shape[-1], 1, p = arguments[ii, ai,:]) for ii in range(arguments.shape[0])])[:,0] # None, None]
        F = np.array([normalize(z[i, idx[i], ...]) for i in range(z.shape[0])])
        Bmask = F #np.uint8(F > (threshold*255))
        # import pdb; pdb.set_trace()
        Bmask = np.array([cv2.resize(Fd, tuple(size), interpolation = cv2.INTER_CUBIC) for Fd in Bmask])
        # print (wts[0], F, threshold*255, Bmask, "=============================")
        return_arguments.append(Bmask[...,None])

    return_arguments = np.array(return_arguments)
    return np.swapaxes(return_arguments, 0, 1) # bs x nargs


def main_image(args, epoch, plot=False):
    plot_dir = args.plot_dir
    
    logs_data = {
                'images': [],
                'z': [],
                'z_idx': [],
                'jpred': [],
                'labels': [],
                'outcome': [],
                'arguments': {},
                'arg_dist': {},
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
            # print("8888888888888888888888888888888888888888888888", logs_data['arguments'][ai].shape)
            # print (ai, (np.argmax(logs_data['arguments'][ai], -1)))
            logs_data['arg_dist'][ai] = data['arguments_dist']
            logs_data['predictions'][ai] = data['preds']

        # import pdb; pdb.set_trace()
        if plot:
            logs_data['visual_arguments'][ai] = get_arguments(logs_data['images'], 
                                                            data['zs'],
                                                            data['arguments'],
                                                            args.threshold)

    if plot:
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

        # plot base image
        color = ['r', 'b', 'g', 'c', 'm', 'y']
        for imgidx in range(0, nrows, args.nagents):
            title = '$Y={}$, $\mathcal{{J}}(x)={}$, $Debate(x)={}$, '.format(logs_data['labels'][imgidx],
                                                                        logs_data['jpred'][imgidx],
                                                                        logs_data['outcome'][imgidx])
            if logs_data['labels'][imgidx] != logs_data['jpred'][imgidx]: continue

            for aidx in range(args.nagents):
                title += '$\mathcal{{P}}^{}(x)={}$, '.format(aidx+1, 
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

                        ax.imshow(normalize(logs_data['images'][imgidx]), cmap='gray', alpha=0.5)
                        ax.imshow(logs_data['visual_arguments'][aidx][imgidx, argidx - args.nagents], cmap='coolwarm', alpha=0.5)
                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)
                        ax.set_title('$\mathcal{{A}}^{}_{}$'.format(aidx +1, argidx - args.nagents + 1))

        # save as png
        path = os.path.join(os.path.dirname(args.plot_dir), 'pngs/')
        path = os.path.join(path, 'epoch_{}.png'.format(epoch))
        plt.savefig(path, bbox_inches='tight')
    
    return logs_data



if __name__ == "__main__":
    args = parse_arguments()
    name = args.name
    os.makedirs(os.path.join(args.plot_dir, 'pngs'), exist_ok=True)
    max_epoch = 49 #TODO: update...

    ZR = []; AH = []; AD = []; epochs = []; accs = []; P1L = []; P2L = []
    for epoch_ in range(args.start_epoch, max_epoch):
        # try:
        data = main_image(args, epoch_, plot=False)
        # except:
            # exit()
        print ("epoch: ================", epoch_)
        zr_, ah_, ad_, dacc_ = get_properties(data['jpred'], data['labels'], data['arg_dist'])
        ZR.append(zr_)
        AH.append(ah_)
        AD.append(ad_)
        # P1L.append(data[0]['loss'])
        # P2L.append(data[1]['loss'])
        accs.append(dacc_)
        epochs.append(epoch_)

    normalize_plot = lambda x: (x - np.min(x))/(np.max(x) - np.min(x))

    def ma(a, n=4) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return (ret[n - 1:] / n)

    ## compute ema and plotting 
    xnew = np.linspace(0, max_epoch, num=100)
    print (max(xnew))
    zr_cubic = interp1d(epochs[:-3], ma(ZR), kind='cubic', fill_value="extrapolate")
    ad_cubic = interp1d(epochs[:-3], ma(AD), kind='cubic', fill_value="extrapolate")
    ah_cubic = interp1d(epochs[:-3], ma(AH), kind='cubic', fill_value="extrapolate")
    acc_cubic = interp1d(epochs[:-3], ma(accs), kind='cubic', fill_value="extrapolate")


    plt.clf()
    n = 10
    plt.plot(xnew[:-n], normalize_plot(zr_cubic(xnew))[:-n], c='b', label='$Z_R$')
    plt.plot(xnew[:-n], normalize_plot(ah_cubic(xnew))[:-n], c='r', label ='$\mathcal{{AH}}$' )
    plt.plot(xnew[:-n], normalize_plot(ad_cubic(xnew))[:-n], c='k', label = '$\mathcal{{AD}}$')
    plt.plot(xnew[:-n], normalize_plot(acc_cubic(xnew))[:-n], c='g', label='Debate accuracy')
    plt.axvline(x=25, color='orange', ls='--', label='Training mode')
    plt.legend()
    plt.xlabel('Training epochs')
    plt.ylabel('Properties')

    # plt.plot(epochs, P1L, c='b-')
    # plt.plot(epochs, P2L, c='r-')


    path = os.path.join(os.path.dirname(args.plot_dir), 'logs.png')
    plt.savefig(path, bbox_inches='tight')

    main_image(args, max_epoch, plot=True)

