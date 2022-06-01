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
        self.use_gpu = args.use_gpu
        self.nfeatures = args.nfeatures
        self.M = args.M


        self.ram_net = PlayerNet(args)

        self.name = 'Agent:{}_{}_{}_{}'.format(iagent,
                                        args.rnn_type, 
                                        args.narguments, 
                                        args.nconcepts)

        self.__init_optimizer(args.init_lr)

    def initLoc(self, batch_size):
        dtype = torch.cuda.FloatTensor if self.use_gpu else torch.FloatTensor
        l_t = torch.Tensor(batch_size, self.nfeatures).uniform_(0, 1)
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
        loss.backward()
        self.optimizer.step()

    def classifier(self, h):
        return self.ram_net.classifier(h)