import os
import torch
import random
import logging
import numpy as np
import torch.nn.functional as F
from torchvision import transforms

from src.data_loader import get_test_loader, get_train_val_loader
from src.dataGenerator import DataGenerator
from src.debate import Debate
from src.callbacks import PlotCbk, ModelCheckpoint, LearningRateScheduler, EarlyStopping
from src.trainer import Trainer
import argparse


import sys
def parse_args():
    def str2bool(v):
        return v.lower() in ('true', '1')

    import sys
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--M', type=float, default=10, help='Monte Carlo sampling for valid and test sets')

    # common parameters
    common_arg = parser.add_argument_group('GlimpseNet Params')
    common_arg.add_argument('--glimpse_hidden', type=int, default=128, help='hidden size of glimpse fc')
    common_arg.add_argument('--patch_size', type=int, default=32, help='size of extracted patch at highest res')
    common_arg.add_argument('--img_size', type=int, default=32, help='size of extracted patch at highest res')
    common_arg.add_argument('--nglimpses', type=int, default=1, help='# of downscaled patches per glimpse')
    common_arg.add_argument('--glimpse_scale', type=int, default=2, help='scale of successive patches')
    common_arg.add_argument('--narguments', type=int, default=6, help='# of glimpses, i.e. BPTT iterations')
    common_arg.add_argument('--model', type=str, default='vanilla', help='Convolutional model to use. One of {"vanilla", "resnetX", "densenetX"}.')

    # core_network params
    core_network_arg = parser.add_argument_group('core_network Params')
    core_network_arg.add_argument('--rnn_hidden', type=int, default=256, help='hidden size of the rnn')  # on purpose set equal to glimpse_hidden + loc_hidden, can be changed
    core_network_arg.add_argument('--rnn_type', type=str, default='RNN', help='Type of RNN Cell to use, RNN/LSTM/GRU')  # on purpose set equal to glimpse_hidden + loc_hidden, can be changed

    # Parameters for confusion matrix
    cnf_matrix_arg = parser.add_argument_group('Params made for confusion matrix')
    cnf_matrix_arg.add_argument('--n_class', type=int, default=3, help='number of classes in the dataset ')

    # data params
    data_arg = parser.add_argument_group('Data Params')
    data_arg.add_argument('--val_split', type=float, default=0.1, help='Proportion of training set used for validation')
    data_arg.add_argument('--num_workers', type=int, default=4, help='# of subprocesses to use for data loading')
    data_arg.add_argument('--random_split', type=str2bool, default=True, help='Whether to randomly split the train and valid indices')
    data_arg.add_argument('--include_classes', type=str, default='all', help='include subset of classes for debate')
   
    # training params
    train_arg = parser.add_argument_group('Training Params')
    train_arg.add_argument('--is_train', type=str2bool, default=True, help='Whether to train or test the model')
    train_arg.add_argument('--batch_size', type=int, default=4, help='# of images in each batch of data')
    train_arg.add_argument('--epochs', type=int, default=25, help='# of epochs to train for')
    train_arg.add_argument('--patience', type=int, default=5, help='Max # of epochs to wait for no validation improv')
    train_arg.add_argument('--momentum', type=float, default=0.5, help='Nesterov momentum value')
    train_arg.add_argument('--init_lr', type=float, default=0.001, help='Initial learning rate value')
    train_arg.add_argument('--min_lr', type=float, default=0.000001, help='Min learning rate value')
    train_arg.add_argument('--saturate_epoch', type=int, default=150, help='Epoch at which decayed lr will reach min_lr')

    #Plotting
    plot_args = parser.add_argument_group('Plotting parameters')
    plot_args.add_argument('--is_plot', type=str2bool, default=False)
    plot_args.add_argument('--mode', type=str, default='train', help='train/valid/test')
    plot_args.add_argument('--num_plots', type=int, default=1, help='')
    plot_args.add_argument('--plot_name', type=str, default='plots', help='')


    # other params
    misc_arg = parser.add_argument_group('Misc.')
    misc_arg.add_argument('--use_gpu', type=str2bool, default=True, help="Whether to run on the GPU")
    misc_arg.add_argument('--device', type=int, default=1, help="GPU device to use")
    misc_arg.add_argument('--best', type=str2bool, default=False, help='Load best model or most recent for testing')
    misc_arg.add_argument('--random_seed', type=int, default=1, help='Seed to ensure reproducibility')
    misc_arg.add_argument('--data_dir', default='./data', help='Directory in which data is stored')
    misc_arg.add_argument('--ckpt_dir', default='./ckpt/conv_model', help='Directory in which to save model checkpoints')
    misc_arg.add_argument('--plot_dir', default='./plots/conv_model', help='Directory in which to save model checkpoints')
    misc_arg.add_argument('--log_dir', default='./logs/', help='Directory in which Tensorboard logs wil be stored')
    misc_arg.add_argument('--use_tensorboard', type=str2bool, default=False, help='Whether to use tensorboard for visualization')
    misc_arg.add_argument('--resume', type=str2bool, default=False, help='Whether to resume training from checkpoint')
    misc_arg.add_argument('--print_freq', type=int, default=10, help='How frequently to print training details')
    misc_arg.add_argument('--plot_freq', type=int, default=1, help='How frequently to plot glimpses')
    misc_arg.add_argument('--plot_num_imgs', type=int, default=4, help='How many imgs to plot glimpses animiation')
    

    # glimpse network parameters
    glimpse_arg = parser.add_argument_group('GlimpseNet Params')
    glimpse_arg.add_argument('--conv', type=str2bool, default=False, help='Use convolutional model')
    glimpse_arg.add_argument('--loc_hidden', type=int, default=128, help='hidden size of loc fc')


    # debate parameters
    debate_arg = parser.add_argument_group('Debate Params')
    debate_arg.add_argument('--nagents', type=int, default=2, help='# of agents in debate')
    debate_arg.add_argument('--fully_observable', type=bool, default=True, help='share locations across agents')
    debate_arg.add_argument('--contrastive', type=bool, default=False, help='fine tune supporter models')
    debate_arg.add_argument('--reward_weightage', type=float, default=1, help='weightage for reward')
    debate_arg.add_argument('--rl_weightage', type=float, default=0.5, help='weightage for rl loss terms')


    # LocationNet params
    location_arg = parser.add_argument_group('LocationNet Params')
    location_arg.add_argument('--std', type=float, default=1.0, help='gaussian policy standard deviation')

    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':

    logger = logging.getLogger('Debate')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', "%m-%d %H:%M")
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    args = parse_args()

    # ensure reproducibility
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    kwargs = {}

    transResize = args.img_size
    traintransformList = [
                    # transforms.RandomAffine(30, translate=(0.2, 0.2), scale=(0.7, 1.0), shear=0.0),
                    transforms.Resize(transResize),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]
    traintransformSequence=transforms.Compose(traintransformList)

    testtransformList = [
                    transforms.Resize(transResize),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]
    testtransformSequence=transforms.Compose(testtransformList)

    # make loc std dependent on img_size and patch_size
    args.std = args.std*args.patch_size*(args.glimpse_scale**args.nglimpses)*0.5/args.img_size
    
    if args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.device)
        torch.cuda.manual_seed(args.random_seed)
        kwargs = {'num_workers': 4, 'pin_memory': True}

    if args.is_train:
        
        train_dataset = DataGenerator(os.path.join(args.data_dir,'training'), 
                                                    traintransformSequence,
                                                    include_classes=args.include_classes)
        val_dataset = DataGenerator(os.path.join(args.data_dir,'testing'), 
                                                    traintransformSequence,
                                                    include_classes=args.include_classes)
        train_loader, val_loader = get_train_val_loader(
            train_dataset, val_dataset,
            val_split=args.val_split,
            random_split=args.random_split,
            batch_size=args.batch_size,
            **kwargs
        )

        args.num_class = train_loader.dataset.num_class
        args.num_channels = train_loader.dataset.num_channels

    else:
        test_dataset = DataGenerator(os.path.join(args.data_dir,'testing'), 
                                                testtransformSequence)
        test_loader = get_test_loader(test_dataset, args.batch_size, **kwargs)
        args.num_class = test_loader.dataset.num_class
        args.num_channels = test_loader.dataset.num_channels


    # build Debate model
    model = Debate(args)
    manipulator_prob = 0.6


    def get_reward(logits, y, attacker=False):
        pred = torch.max(logits, dim=1)[0]
        rand_prob = np.random.uniform(0,1)

        if (not attacker) or (not args.contrastive) or (rand_prob > manipulator_prob):
            reward = (pred == y).float()
        else:
            reward = -(pred == y).float()


        return args.reward_weightage*reward 

    
    reward_fns = [lambda p, y: get_reward(p, y),
                    lambda p, y: get_reward(p, y, True),]
    model.reward_fns = reward_fns


    # classifier_fns = [lambda p, jpred, y: F.nll_loss(p[-1], y),
    #                     lambda p, jpred, y: F.nll_loss(p[-1], y)]
    # model.classifier_fns = classifier_fns


    if args.use_gpu: model.cuda()

    logger.info('Number of model parameters: {:,}'.format(
                sum([p.data.nelement() for p in model.parameters()])))

    trainer = Trainer(model, 
                        watch=['acc', 'loss'], 
                        val_watch=['acc', 'loss'], 
                        logger=logger)


    if args.is_train:
        logger.info("Train on {} samples, validate on {} samples".format(len(train_loader.dataset), len(val_loader.dataset)))
        start_epoch = 0
        if args.resume:
            start_epoch = model.load_model(args.ckpt_dir, 
                                            best=args.best)


        # best model selection method is based on acc in callback.py
        # Need to fix that before changing monitor_val
        monitor_val = 'val_0_acc' 
        trainer.train(train_loader, val_loader,
                      start_epoch=start_epoch,
                      epochs=args.epochs,
                      callbacks=[
                          PlotCbk(args.plot_dir, model, 
                                      args.plot_num_imgs, 
                                    args.plot_freq, 
                                    args.use_gpu),
                          # TensorBoard(model, args.log_dir),
                          ModelCheckpoint(model, 
                                            args.ckpt_dir,
                                            monitor_val),
                          LearningRateScheduler(model, 
                                                factor = 0.1, 
                                                patience = 5, 
                                                mode = 'min', 
                                                monitor_val = monitor_val),
                          EarlyStopping(model, 
                                          monitor_val,
                                          patience=args.patience)
                      ])
    elif args.is_plot:
        dataset = DataGenerator(os.path.join(args.data_dir, 'testing'), 
                                    testtransformSequence)
        loader = get_test_loader(dataset, args.num_plots, **kwargs)
        logger.info("Plotting a random batch from the folder {}".format('testing'))
        print (args.contrastive)
        start_epoch = model.load_model(args.ckpt_dir, contrastive=args.contrastive, best=args.best)
        trainer.plot(loader, PlotCbk(args.plot_dir, model, args.num_plots, 1, args.use_gpu), args.plot_name)
    else:
        logger.info("Test on {} samples".format((len(test_loader))))
        print(args.ckpt_dir)
        epoch = model.load_model(args.ckpt_dir, contrastive=args.contrastive, best=args.best)
        print(epoch)
        trainer.test(test_loader, best=args.best)