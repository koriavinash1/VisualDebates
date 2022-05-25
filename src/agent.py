from torch.autograd import Variable
import torch.nn.functional as F

import torch
import torch.nn as nn
from src.modules import PlayerNet
torch.autograd.set_detect_anomaly(True)

class RecurrentAttentionAgent(nn.Module):
    def __init__(self, args, iagent):
        """
        Initialize the recurrent attention model and its different components.
        """
        super(RecurrentAttentionAgent, self).__init__()
        self.std = args.std
        self.use_gpu = args.use_gpu
        self.M = args.M

        self.ram_net = PlayerNet(args)

        self.name = 'Agent:{}_{}_{}_{}_{}x{}_{}_{}'.format(iagent,
                                        args.model, 
                                        args.rnn_type, 
                                        args.narguments, 
                                        args.patch_size, 
                                        args.patch_size, 
                                        args.glimpse_scale, 
                                        args.nglimpses)

        self.__init_optimizer(args.init_lr)

    def initLoc(self, batch_size):
        dtype = torch.cuda.FloatTensor if self.use_gpu else torch.FloatTensor
        l_t = torch.Tensor(batch_size, 2).uniform_(-1, 1)
        l_t = Variable(l_t).type(dtype)
        return l_t
    
    def init_rnn_hidden(self, batch_size):
        return self.ram_net.rnn.init_hidden(batch_size)

    def __init_optimizer(self, lr=1e-3, weight_decay = 1e-5):
        print("LearningRate: ", lr)
        self.optimizer = torch.optim.Adam (self.parameters(), 
                            lr=lr, 
                            weight_decay=weight_decay)

    def forwardStep(self, x, lt_agent, lt_other, h_t):
        return self.ram_net.step(x, lt_agent, lt_other, h_t)

    def optStep(self, loss):
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

    def classifier(self, h):
        return self.ram_net.classifier(h)