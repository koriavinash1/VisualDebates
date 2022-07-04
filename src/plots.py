from cProfile import label
from inspect import ArgSpec
import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2, copy
from scipy import stats
import matplotlib.animation as animation
from scipy.interpolate import interp1d, make_interp_spline
from utils import bounding_box

## python3 plot_glimpses.py --plot_dir=./plots/ram_6_8x8_2_2/ --epoch=111  -- use this to run and save gif
def parse_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument("--plot_dir", type=str,
                     help="path to directory containing pickle dumps")
    arg.add_argument("--start_epoch", type=int, default=0,
                     help="epoch of desired plot")
    arg.add_argument("--stop_epoch", type=int, default=0,
                     help="epoch of desired plot")
    arg.add_argument("--split_epoch", type=int, default=0,
                     help="epoch of desired plot")
    arg.add_argument("--nagents", type=int, default=2,
                     help="Total number of agents in a debate")
    arg.add_argument("--quantize", type=str, default='channel',
                     help="Different quantization approaches: spatial or channel")
    arg.add_argument("--name", type=str, default='', 
                    help='Name of the plots')
    arg.add_argument("--properties", type=bool, default=False, 
                    help='Only plot convergence and argument properties')
    arg.add_argument("--threshold", type=float, default=0.95, 
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
        d = (1./(total_unqiue*np.sum(class_idx))) * (np.sum((p1_class_arguments != p2_class_arguments)))
        zr_.append(d)
    zr_ = np.mean(zr_)
   
   # AH calculation
    # import pdb; pdb.set_trace()
    # print (arguments[0].shape, "*******************")
    ah_ = 0.5*(stats.entropy(arguments[0], axis=-1) + stats.entropy(arguments[1], axis=-1))
    # ah_ = 0.5*(np.var(arguments[0], axis=-1) + \
    #             np.var(arguments[1], axis=-1))

    # import pdb;pdb.set_trace()
   # AD calculation
    ad_ = np.mean((arguments[0] - arguments[1])**2, -1)**0.5

    ah_ = np.mean(ah_)
    ad_ = np.mean(ad_)
    # dacc
    dacc = np.sum(pred == y)*1.00/len(y)
    return zr_, ah_, ad_, dacc 



def get_arguments(x, z, 
                    sampled_idx, 
                    arguments, 
                    arg_idx = [],
                    quantize = 'channel', 
                    threshold=0.85):
    size = np.asarray(x.shape[1:-1])
    return_arguments = [] #nargs x bs
    argument_idx = []

    # z = z/np.max(z)
    for ai in range(arguments.shape[1]):
        z_pertub = [];batch_arg = []
        for i in range(z.shape[0]):
            # sampling with poisioning
            if not len(arg_idx):
                probabilities = arguments[i, ai,:]
                noise = np.random.uniform(0, 0.25, probabilities.shape)
                probabilities += noise
                probabilities /= np.sum(probabilities)
                idx = np.random.choice(arguments.shape[-1], 1, p = probabilities)
            else:
                idx = np.argmax(arg_idx[i, ai])

            z_ = copy.deepcopy(z[i])
            if quantize == 'channel':
                z_[sampled_idx[i] == sampled_idx[i][idx]] = 0 
                batch_arg.append(sampled_idx[i][idx])
            else:
                z_[idx] = 0
                batch_arg.append(idx)

            z_pertub.append(z_)

        z_pertub = np.array(z_pertub)

        # F_cummilative = np.sum(z, 1)
        # Ft_cummilative = np.sum(z_pertub, 1)

        # import pdb;pdb.set_trace()
        F = z - z_pertub #F_cummilative - Ft_cummilative
        F = F/np.max(F)

        Bmask = F*(F > np.percentile(F, threshold, 0))
        Bmask = np.sum(Bmask, 1)

        Bmask = np.array([cv2.resize(Fd, tuple(size), \
                            interpolation = cv2.INTER_CUBIC) \
                            for Fd in Bmask])
        return_arguments.append(Bmask[...,None])
        argument_idx.append(batch_arg)

    return_arguments = np.array(return_arguments)
    return np.swapaxes(return_arguments, 0, 1), np.swapaxes(np.array(argument_idx), 0, 1) # bs x nargs


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
                'arguments_idx': {},
                'predictions':{},
                'loss':{}
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
            logs_data['arg_dist'][ai] = data['arguments_dist']
            logs_data['predictions'][ai] = data['preds']
            logs_data['loss'][ai] = data['loss']

        # import pdb; pdb.set_trace()
        if plot:
            visual_arguments, argument_idx = get_arguments(logs_data['images'], 
                                                            data['zs'],
                                                            data['zs_idx'],
                                                            data['arguments_dist'],
                                                            arg_idx = data['arguments'],
                                                            quantize=args.quantize,
                                                            threshold = args.threshold)
            logs_data['visual_arguments'][ai] = visual_arguments
            logs_data['arguments_idx'][ai] = argument_idx


    if plot:
        narguments = len(logs_data['arguments'][0][0])
        num_imgs = logs_data['images'].shape[0]
        img_shape = np.asarray([logs_data['images'][0].shape[1:]])
        
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
            title = '$Y={}$, $\mathcal{{J}}(x)={}$, $Debate(x)={}$, '.format(logs_data['labels'][imgidx//2],
                                                                        logs_data['jpred'][imgidx//2],
                                                                        logs_data['outcome'][imgidx//2])
            if logs_data['labels'][imgidx//2] != logs_data['jpred'][imgidx//2]: continue
            if logs_data['outcome'][imgidx//2] != logs_data['labels'][imgidx//2]: continue
            if logs_data['predictions'][0][imgidx//2] == logs_data['predictions'][1][imgidx//2]: continue
            # if logs_data['predictions'][0][imgidx//2] != logs_data['outcome'][imgidx//2]: continue

            for aidx in range(args.nagents):
                # aidx = 1 - aidx
                title += '$\mathcal{{P}}^{}(x)={}$, '.format(aidx+1, 
                                                logs_data['predictions'][aidx][imgidx//2])
                                    
                for argidx in range(ncols):
                    
                    if argidx == 0:
                        ax = plt.subplot2grid(shape=(nrows, ncols), 
                                                loc=(imgidx, 0), 
                                                colspan=args.nagents,
                                                rowspan=args.nagents)
                        ax.imshow(normalize(logs_data['images'][imgidx//2]),cmap='gray')
                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)
                        ax.set_title(title)
                        
                    elif argidx < args.nagents:
                        continue
                    else:
                        # print (imgidx, aidx, argidx, imgidx//args.nagents + aidx)
                        ax = plt.subplot2grid(shape=(nrows, ncols), 
                                                loc=(imgidx + aidx, argidx))

                        ax.imshow(normalize(logs_data['images'][imgidx//2]), cmap='gray', alpha=0.5)
                        ax.imshow(logs_data['visual_arguments'][aidx][imgidx//2, argidx - args.nagents], cmap='coolwarm', alpha=0.5)
                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)
                        ax.set_title('$\mathcal{{A}}^{}_{} = {}$'.format(aidx +1, 
                                                                argidx - args.nagents + 1,
                                                                logs_data['arguments_idx'][aidx][imgidx//2, argidx - args.nagents]))

        # save as png
        path = os.path.join(os.path.dirname(args.plot_dir), 'pngs/')
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'epoch_{}.png'.format(epoch))
        plt.savefig(path, bbox_inches='tight')
    
    return logs_data



if __name__ == "__main__":
    args = parse_arguments()
    name = args.name
    os.makedirs(os.path.join(args.plot_dir, 'pngs'), exist_ok=True)
    max_epoch = args.stop_epoch #TODO: update...

    ZR = []; AH = []; AD = []; epochs = []; accs = []; P1L = []; P2L = []
    for epoch_ in range(args.start_epoch, max_epoch):
        # try:
        data = main_image(args, epoch_, plot=True)
        # except:
            # exit()
        print ("epoch: ================", epoch_)
        zr_, ah_, ad_, dacc_ = get_properties(data['outcome'], data['jpred'], data['arg_dist'])
        ZR.append(zr_)
        AH.append(ah_)
        AD.append(ad_)
        P1L.append(np.mean(data['loss'][0]))
        P2L.append(np.mean(data['loss'][1]))
        accs.append(dacc_)
        epochs.append(epoch_+1)

    normalize_plot = lambda x: (x - np.min(x))/(np.max(x) - np.min(x))
    # max_loss = np.max((np.max(np.abs(P1L)), np.max(np.abs(P2L))))
    # P1L = P1L/max_loss
    # P2L = P2L/max_loss

    mn = 8
    def ma(a, n=mn+1) :
        a = list(a)
        for _ in range(n-1):
            a.append(a[-1])
        
        a = np.array(a)
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return (ret[n - 1:] / n)

    ## compute ema and plotting 
    xnew = np.linspace(0, max_epoch, num=100)
    zr_cubic = interp1d(epochs, ma(ZR), kind='cubic', fill_value="extrapolate")
    ad_cubic = interp1d(epochs, ma(AD), kind='cubic', fill_value="extrapolate")
    ah_cubic = interp1d(epochs, ma(AH), kind='cubic', fill_value="extrapolate")
    acc_cubic = interp1d(epochs, ma(accs), kind='cubic', fill_value="extrapolate")
    p1l_cubic = interp1d(epochs, ma(P1L), kind='cubic', fill_value="extrapolate")
    p2l_cubic = interp1d(epochs, ma(P2L), kind='cubic', fill_value="extrapolate")



    n = 1
    plt.clf()
    plt.figure(figsize=(6,6))
    # plt.plot(epochs, ma(ZR), c='b', label='$Z_R$')
    # plt.plot(epochs, normalize_plot(ma(AH)), c='m', label ='$\mathcal{{AH}}$' )
    # plt.plot(epochs, normalize_plot(ma(AD)), c='k', label = '$\mathcal{{AD}}$')
    # plt.plot(epochs, ma(accs), c='g', label='Debate accuracy')
    plt.plot(xnew[:-n], zr_cubic(xnew)[:-n], c='b', label='$Z_R$')
    plt.plot(xnew[:-n], normalize_plot(ah_cubic(xnew))[:-n], c='m', label ='$\mathcal{{AH}}$' )
    plt.plot(xnew[:-n], normalize_plot(ad_cubic(xnew))[:-n], c='k', label = '$\mathcal{{AD}}$')
    plt.plot(xnew[:-n], acc_cubic(xnew)[:-n], c='g', label='Debate accuracy')
    # plt.plot(xnew[:-n], (p1l_cubic(xnew))[:-n], c='c', label='$\mathcal{{P}}^1$ Loss Profile')
    # plt.plot(xnew[:-n], (p2l_cubic(xnew))[:-n], c='r', label='$\mathcal{{P}}^2$ Loss Profile')
    plt.axvline(x=args.split_epoch, color='orange', ls='--', label='Training mode')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Normalized Properties')


    path = os.path.join(os.path.dirname(args.plot_dir), 'logs.png')
    plt.savefig(path, bbox_inches='tight')

    main_image(args, max_epoch, plot=True)
    # print (ZR, zr_cubic(xnew))

