
from tqdm import tqdm
from src.utils import AverageMeter
import torch



class Trainer(object):
    """
    Trainer encapsulates all the logic necessary for training.
    """
    def __init__(self, model, watch=[], val_watch=[], logger=None):
        self.model = model
        self.stop_training = False
        self.watch = watch
        self.val_watch = val_watch
        self.logger = logger
        if 'loss' not in watch:
            watch.insert(0, 'loss')
        if 'loss' not in val_watch:
            val_watch.insert(0, 'loss')

    def train(self, train_loader, val_loader, start_epoch=0, epochs=200, callbacks=[]):
        for epoch in range(start_epoch, epochs):
            if self.stop_training:
                return
            epoch_log = self.train_one_epoch(epoch, train_loader, callbacks=callbacks)
            val_log = self.validate(epoch, val_loader)

            msg = ' '.join(['{}: {:.3f}'.format(name, avg) for name, avg in epoch_log.items()])
            self.logger.info(msg)
            msg = ' '.join(['{}: {:.3f}'.format(name, avg) for name, avg in val_log.items()])
            self.logger.info(msg)
            epoch_log.update(val_log)

            for cbk in callbacks:
                cbk.on_epoch_end(epoch, epoch_log)

    def train_one_epoch(self, epoch, train_loader, callbacks=[]):
        """
        Train the model for 1 epoch of the training set.
        """
        epoch_log = {}
        self.model.train()
        for i, (x, y, lts) in enumerate(tqdm(train_loader, unit='batch', desc='Epoch {:>3}'.format(epoch))):
            metric = self.model.forward(x, y, lts, is_training=True, epoch=epoch+1)
            for name in self.watch:
                for key in metric.keys():
                    if isinstance(metric[key], dict):
                        for key_ in metric[key]:
                            if key_.__contains__(name):
                                if (i == 0): epoch_log[str(key) + '_' + key_] = AverageMeter()
                                epoch_log[str(key) + '_' + key_].update(metric[key][key_].item(), x.size()[0])
          
                    else:
                        if key.__contains__(name):
                            if (i == 0): epoch_log[name] = AverageMeter()
                            epoch_log[name].update(metric[key].item(), x.size()[0])

            for cbk in callbacks:
                cbk.on_batch_end(epoch, i, logs=metric)


        return {name: meter.avg for name, meter in epoch_log.items()}

    @torch.no_grad()
    def validate(self, epoch, val_loader):
        """
        Evaluate the model on the validation set.
        """
        val_log = {}
        self.model.eval()
        for i, (x, y, lts) in enumerate(tqdm(val_loader, unit='batch', desc='Epoch {:>3}'.format(epoch))):
            metric = self.model.forward(x, y, lts, is_training=False, epoch=epoch+1)
            for name in self.watch:
                for key in metric.keys():
                    if isinstance(metric[key], dict):
                        for key_ in metric[key]:
                            if key_.__contains__(name):
                                if (i == 0): val_log[str(key) + '_' + key_] = AverageMeter()
                                val_log[str(key) + '_' + key_].update(metric[key][key_].item(), x.size()[0])
          
                    else:
                        if key.__contains__(name):
                            if (i == 0): val_log[name] = AverageMeter()
                            val_log[name].update(metric[key].item(), x.size()[0])


        # TODO: remove the hard-coding here-----
        # cqloss = val_log['0_cqloss'].avg
        # p1loss = val_log['0_loss'].avg
        # p2loss = val_log['1_loss'].avg

        # if not (self.model.quantized_lr_sch is None):
        #     self.model.quantized_lr_sch.step(cqloss)
        # self.model.agents[0].lr_sch.step(p1loss)
        # self.model.agents[1].lr_sch.step(p2loss)

        #-----------------------------------------------
        return {'val_'+name: meter.avg for name, meter in val_log.items()}

    @torch.no_grad()
    def test(self, test_loader, best=True):
        """
        Test the model on the held-out test data.
        This function should only be called at the very
        end once the model has finished training.
        """
        # load the best checkpoint
        # self.load_checkpoint(best=best)

        accs = {} #AverageMeter()
        self.model.eval()
        for i, (x, y, lts) in enumerate(tqdm(test_loader, unit='batch')):
            # metric = self.model.forward(x, y, is_training=False)
            # metric,confusion_meter = self.model.forward(x, y, is_training=False)
            metric = self.model.forward(x, y, lts, is_training=False)
            if isinstance(metric[list(metric.keys())[0]], dict):
                for key in metric.keys():
                    if (i == 0): accs[key] = AverageMeter()
                    accs[key].update(metric[key]['acc'].item(), x.size()[0])
            else:
                if (i == 0): accs = AverageMeter()
                accs.update(metric['acc'].item(), x.size()[0])
            # confusion_meter = metric['con_mat']  
            # print(dir(acc))
            # assert False

            # for cbk in callbacks:
            #     cbk.on_batch_end(epoch, i, logs=metric)
        # print(accs)
        confusion_meter = metric['con_mat']
        print (confusion_meter.conf)
        self.logger.info('Test Acc: ({:.2f}%)'.format(accs.avg))

    @torch.no_grad()
    def plot(self, data_loader, PlotCallback, name):
        self.model.eval()
        x, y, _ = next(iter(data_loader))
        metric = self.model.forward(x, y, _, is_training=False)
        PlotCallback.on_batch_end(-1,0, name=name, logs=metric)
