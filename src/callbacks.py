import warnings

import numpy as np

import pickle
import os
import torch
import shutil

from src.tflogger import TFLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau



class Callback(object):
    '''Abstract base class used to build new callbacks.
        on_batch_end: logs include `loss`, and optionally `acc`
            (if accuracy monitoring is enabled).
    '''
    def __init__(self, model):
        self.model = model

    def on_train_beg(self):
        pass

    def on_epoch_end(self, epoch, logs={}):
        pass

    def on_batch_end(self, epoch, batch, name='', logs={}):
        pass

    def on_train_end(self, logs={}):
        pass


class PlotCbk(Callback):
    def __init__(self, plot_dir, model, num_imgs, plot_freq, use_gpu):
        self.model = model
        self.num_imgs = num_imgs
        self.plot_freq = plot_freq
        self.use_gpu = use_gpu
        self.plot_dir = os.path.join(plot_dir, self.model.name+'/')
        # print (self.plot_dir)
        # print ('###################################')
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
    
    def plot(self, imgs, zs, zs_idx, arguments, argument_dist, dpreds, preds, jpreds, Ys, loss, epoch, batch_ind, name):
        if ((epoch % self.plot_freq == 0) and (batch_ind == 0)) or (epoch == -1):
            if self.use_gpu:
                imgs = imgs.detach().cpu()
                zs = zs.detach().cpu() 
                zs_idx = zs_idx.cpu()
                arguments = arguments.cpu()
                arguments_dist = argument_dist.detach().cpu()
                preds = preds.cpu()
                jpreds = jpreds.cpu()
                dpreds = dpreds.cpu()
                Ys = Ys.cpu()
                loss = loss.detach().cpu()


            pickle.dump(
                dict(
                    imgs = imgs.numpy(),
                    zs = zs.numpy(), 
                    zs_idx = zs_idx.numpy(),
                    arguments = arguments.numpy(),
                    arguments_dist = arguments_dist.numpy(),
                    preds = preds.numpy(),
                    dpreds = dpreds.numpy(),
                    jpreds = jpreds.numpy(),
                    Ys = Ys.numpy(),
                    loss = loss.numpy(),
                ),

                open(
                    self.plot_dir + "{}_logs_{}.p".format(name, epoch),
                    "wb"
                )
            )

    def on_batch_end(self, epoch, batch_ind, name='',logs={}):
        if isinstance(logs[list(logs.keys())[0]], dict):
            for key in logs.keys():
                imgs = logs[key]['x'][:self.num_imgs]
                zs = logs[key]['z'][:self.num_imgs]
                zs_idx = logs[key]['z_idx'][:self.num_imgs]
                arguments = logs[key]['arguments'][:self.num_imgs]
                argument_dist = logs[key]['argument_dist'][:self.num_imgs]
                preds = logs[key]['preds'][:self.num_imgs] 
                jpreds = logs[key]['jpred'][:self.num_imgs]
                dpreds = logs[key]['dpred'][:self.num_imgs]
                Ys = logs[key]['y'][:self.num_imgs]
                loss = logs[key]['loss']
                self.plot(imgs, zs, zs_idx, arguments, argument_dist, dpreds, preds, jpreds, Ys, loss, epoch, batch_ind, str(key) + '_' +name)
        else:
            imgs = logs['x'][:self.num_imgs]
            zs = logs['z'][:self.num_imgs]
            zs_idx = logs['z_idx'][:self.num_imgs]
            arguments = logs['arguments'][:self.num_imgs]
            argument_dist = logs['argument_dist'][:self.num_imgs]
            preds = logs['preds'][:self.num_imgs] 
            jpreds = logs['jpred'][:self.num_imgs] 
            dpreds = logs['dpred'][:self.num_imgs] 
            Ys = logs['y'][:self.num_imgs]
            loss = logs['loss']
            self.plot(imgs, zs, zs_idx, arguments, argument_dist, dpreds, preds, jpreds, Ys, loss, epoch, batch_ind, name)


class TensorBoard(Callback):
    def __init__(self, model, log_dir):
        self.model = model
        self.logger = TFLogger(log_dir)

    def to_np(self, x):
        return x.data.cpu().numpy()

    def on_epoch_end(self, epoch, logs, name=''):
        for tag in ['loss', 'acc']:
            self.logger.scalar_summary(tag, logs[tag], epoch)

        for tag, value in self.model.named_parameters():
            tag = tag.replace('.', '/')
            self.logger.histo_summary(tag, self.to_np(value), epoch)
            self.logger.histo_summary(tag+'/grad', self.to_np(value.grad), epoch)


class ModelCheckpoint(Callback):
    def __init__(self, model, ckpt_dir, monitor_val):
        self.model = model
        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)
        self.monitor_val = monitor_val
        suff = 'contrastive' if self.model.contrastive else 'supportive'

        filename = self.model.name + suff + '_ckpt'
        self.ckpt_path = os.path.join(ckpt_dir, filename)
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs={}):
        state = {'epoch': epoch}
        
        for k, v in logs.items():
            state[k] = v

        state.update(self.model.get_state_dict())
        torch.save(state, self.ckpt_path)

        if logs[self.monitor_val] > self.best_val_acc:
            shutil.copyfile(self.ckpt_path, self.ckpt_path + '_best')
            print ("Model saved: epoch: {}, acc:{}, last-max-acc: {}, path: {}".format(epoch, logs[self.monitor_val], self.best_val_acc, self.ckpt_path))
            self.best_val_acc = logs[self.monitor_val]


class LearningRateScheduler(Callback):
    def __init__(self, optimizer, factor, patience, mode, monitor_val,verbose=True):
        
        self.scheduler = None
        if not (optimizer is None):
            self.scheduler = ReduceLROnPlateau(optimizer,
                                            factor=factor,
                                            patience=patience,
                                            mode = mode,
                                            verbose=verbose)
        print("LR schedular included....")
        self.monitor_val = monitor_val
        self.best_val_loss = 1000.0
        self.counter = 0


    def on_epoch_end(self, epoch, logs):

        if not (self.scheduler is None):
            self.scheduler.step(logs[self.monitor_val])

        

class EarlyStopping(Callback):
    '''Stop training when a monitored quantity has stopped improving.
    '''
    def __init__(self, model, monitor='val_loss', patience=0, verbose=0, mode='auto'):
        '''
        @param monitor: str. Quantity to monitor.
        @param patience: number of epochs with no improvement after which training will be stopped.
        @param verbose: verbosity mode, 0 or 1.
        @param mode: one of {auto, min, max}. Decides if the monitored quantity improves. If set to `max`, increase of the quantity indicates improvement, and vice versa. If set to 'auto', behaves like 'max' if `monitor` contains substring 'acc'. Otherwise, behaves like 'min'.
        '''
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.model = model

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode), RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Early stopping requires {} available!'.format(self.monitor), RuntimeWarning)

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print('Epoch {}: early stopping'.format(epoch))

                self.model.stop_training = True
            self.wait += 1
