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

        self.name = 'Exp-{}-Agent:{}_{}_{}_{}'.format(args.name,
                                        iagent,
                                        args.rnn_type, 
                                        args.narguments, 
                                        args.nconcepts)

        self.__init_optimizer(args.init_lr)

    def init_argument(self, batch_size):
        dtype = torch.cuda.FloatTensor if self.use_gpu else torch.FloatTensor
        arg_0 = torch.Tensor(batch_size, self.nfeatures).uniform_(0, 1)
        arg_0 = Variable(arg_0).type(dtype)
        arg_0 = F.softmax(arg_0, dim=1)
        # return torch.argmax(arg_0, 1)
        return arg_0
    
    def init_rnn_hidden(self, batch_size):
        return self.ram_net.rnn.init_hidden(batch_size)

    def __init_optimizer(self, lr=1e-3, weight_decay = 1e-5):
        print("LearningRate: ", lr)
        self.optimizer = torch.optim.Adam (self.parameters(), 
                            lr=lr, 
                            weight_decay=weight_decay)

    def forwardStep(self, x, arg_agent, arg_other, h_t):
        return self.ram_net.step(x, arg_agent, arg_other, h_t)

    def optStep(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def classifier(self, *args):
        return self.ram_net.classifier(*args)