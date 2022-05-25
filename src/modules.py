import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from torch.autograd import Variable
import torchvision.models as models
import numpy as np

import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.hypersphere import HypersphereMetric
from einops import rearrange


class Clamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0.9, max=1.1) # the value in iterative = 2

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

class weightConstraint2(object):
    def __init__(self):
        pass

    def __call__(self,module):
        if hasattr(module,'weight'):
            w=module.weight.data
            w = torch.heaviside(w, torch.tensor([0.0]))
            x = w.shape[0]
            module.weight.data=w


class VectorQuantizer2DHS(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, args):
        super().__init__()
        '''
        n_e : total number of codebook vectors
        e_dim: codebook vector dimension
        beta: factor for legacy term
        '''

        # device,  
        # n_e = 128, 
        # e_dim = 16, 
        # beta = 0.9, 
        # disentangle = True,
        # remap=None, 
        # unknown_index="random",
        # legacy=True, 

        remap = args.remap
        unknown_index = args.unknown_index
        self.n_e = args.n_e
        self.e_dim = args.e_dim
        self.beta = args.beta
        self.legacy = args.legacy
        self.device = args.device
        self.disentangle = args.disentangle

        self.epsilon = 1e-4
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}
        self.min_norm = 1e-15
        # uniformly sampled initialization
        sphere = Hypersphere(dim=self.e_dim - 1)
        self.embedding = nn.Embedding(self.n_e, 
                                        self.e_dim)


        points_in_manifold = torch.Tensor(sphere.random_uniform(n_samples=self.n_e))
        self.embedding.weight.data.copy_(points_in_manifold).requires_grad=True


        self.hsreg = lambda x: [ torch.norm(x[i]) for i in range(x.shape[0])]
        self.r = torch.nn.Parameter(torch.ones(self.n_e)).to(self.device)
        self.ed = lambda x: [torch.norm(x[i]) for i in range(x.shape[0])]
        

        # remap
        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1

            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = self.n_e

        self.clamp_class = Clamp()


    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None, None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def HLoss(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z,
                    temp=None, 
                    rescale_logits=False, 
                    return_logits=False):
        
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        z_flattened = z.view(-1, self.e_dim)


        # intra distance (gdes-distance) between codebook vector 
        d1 = torch.einsum('bd,dn->bn', self.embedding.weight, rearrange(self.embedding.weight, 'n d -> d n'))
        ed1 = torch.tensor(self.ed(self.embedding.weight))
        ed1 = ed1.repeat(self.n_e, 1)
        ed2 = ed1.transpose(0,1)
        ed3 = ed1 * ed2

        edx = d1/ed3.to(self.device)
        edx = torch.clamp(edx, min=-0.99999, max=0.99999)
        d1 = torch.acos(edx)
        

        min_distance = torch.kthvalue(d1, 2, 0)
        total_min_distance = torch.mean(min_distance[0])
        codebookvariance = torch.mean(torch.var(d1, 1))

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))
        
        min_encoding_indices = torch.argmin(d, dim=1)
    

        z_q = self.embedding(min_encoding_indices).view(z.shape)

        hsw = torch.Tensor(self.hsreg(self.embedding.weight)).to(self.device)
        hsw = torch.mean(torch.square(self.r - hsw))

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) 
            loss += torch.mean((z_q - z.detach()) ** 2) 
        else:
            loss = torch.mean((z_q.detach() - z) ** 2)  
            loss += self.beta * torch.mean((z_q - z.detach()) ** 2)


        disentanglement_loss = codebookvariance - total_min_distance
        if self.disentangle:
            loss += hsw
            loss += disentanglement_loss


        # preserve gradients
        z_q = z + (z_q - z).detach()


        sampled_idx = torch.zeros(z.shape[0]*self.n_e).to(z.device)
        sampled_idx[min_encoding_indices] = 1
        sampled_idx = sampled_idx.view(z.shape[0], self.n_e)


        return (z_q, loss,
                    (sampled_idx, min_encoding_indices.view(z.shape[0], -1)), 
                    codebookvariance, 
                    total_min_distance,  
                    hsw, 
                    torch.mean(self.r))



class RevLSTMCell(nn.Module):
    """ Defining Network Completely along with gradients to Variables """
    def __init__(self, input_size, hidden_size):
        super(RevLSTMCell, self).__init__()
        self.f_layer= nn.Linear(input_size + hidden_size, hidden_size) 
        self.i_layer= nn.Linear(input_size + hidden_size, hidden_size) 
        self.uc_layer= nn.Linear(input_size + hidden_size, hidden_size) 
        self.u_layer= nn.Linear(input_size + hidden_size, hidden_size) 
        self.r_layer= nn.Linear(input_size + hidden_size, hidden_size) 
        self.uh_layer= nn.Linear(input_size + hidden_size, hidden_size) 
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.hidden_size = hidden_size
        self.init_weights()
        
    def forward(self, x, state):
        c, h = state
        #import pdb; pdb.set_trace()
        concat1 = torch.cat((x, h), dim=-1)
        f = self.sigmoid(self.f_layer(concat1))
        i = self.sigmoid(self.i_layer(concat1))
        c_ = self.tanh(self.uc_layer(concat1))
        new_c = (f*c + i*c_)/2
        
        concat2 = torch.cat((x, new_c), dim=-1)
        u = self.sigmoid(self.u_layer(concat2))
        r = self.sigmoid(self.r_layer(concat2))
        h_ = self.tanh(self.uh_layer(concat2))
        new_h = (r*h + u*h_)/2
        return new_h, (new_c, new_h)
    
    def reconstruct(self, x, state):
        new_c, new_h = state
        
        concat2 = torch.cat((x, new_c), dim=-1)
        u = self.sigmoid(self.u_layer(concat2))
        r = self.sigmoid(self.r_layer(concat2))
        h_ = self.tanh(self.uh_layer(concat2))
        h = (2*new_h - u*h_)/(r+1e-64)
        
        
        concat1 = torch.cat((x, h), dim=-1)
        f = self.sigmoid(self.f_layer(concat1))
        i = self.sigmoid(self.i_layer(concat1))
        c_ = self.tanh(self.uc_layer(concat1))
        c = (2*new_c - i*c_)/(f+1e-64)
        
        return h, (c, h)
    
    def init_weights(self):
        for parameter in self.parameters():
            if parameter.ndimension() == 2:
                nn.init.xavier_uniform(parameter, gain=0.01)
    
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(bsz, self.hidden_size).zero_()),
                Variable(weight.new(bsz, self.hidden_size).zero_()))


class core_network(nn.Module):
    """
    An RNN that maintains an internal state that integrates
    information extracted from the history of past observations.
    It encodes the agent's knowledge of the environment through
    a state vector `h_t` that gets updated at every time step `t`.

    Concretely, it takes the glimpse representation `g_t` as input,
    and combines it with its internal state `h_t_prev` at the previous
    time step, to produce the new internal state `h_t` at the current
    time step.

    In other words:

        `h_t = relu( fc(h_t_prev) + fc(g_t) )`

    Args
    ----
    - input_size: input size of the rnn.
    - rnn_hidden: hidden size of the rnn.
    - g_t: a 2D tensor of shape (B, rnn_hidden). The glimpse
      representation returned by the glimpse network for the current timestep `t`.
    - h_t_prev: a 2D tensor of shape (B, rnn_hidden). The
      hidden state vector for the previous timestep `t-1`.

    Returns
    -------
    - h_t: a 2D tensor of shape (B, rnn_hidden). The hidden
      state vector for the current timestep `t`.
    """
    def __init__(self, input_size, 
                    rnn_hidden, 
                    use_gpu, 
                    rnn_type='RNN'):
        super(core_network, self).__init__()
        self.input_size = input_size
        self.rnn_hidden = rnn_hidden
        self.use_gpu = use_gpu
        self.rnn_type = rnn_type
        if rnn_type=='RNN':
            self.rnn = nn.RNNCell(input_size, rnn_hidden, bias=True, nonlinearity='relu')
        if rnn_type=='LSTM':
            self.rnn = nn.LSTMCell(input_size, rnn_hidden, bias=True)
        if rnn_type=='GRU':
            self.rnn = nn.GRUCell(input_size, rnn_hidden, bias=True)
        if rnn_type=='REV':
            self.rnn = RevLSTMCell(input_size, rnn_hidden)

        if use_gpu:
            self.cuda()


    def forward(self, g_t, h_t_prev):
        if self.rnn_type == 'RNN' or self.rnn_type == 'GRU':
            h = self.rnn(g_t, h_t_prev[0])
            h_t = (h, 0)
        if self.rnn_type == 'LSTM' or self.rnn_type == 'REV':
            h_t = self.rnn(g_t, h_t_prev)
        return h_t

    def init_hidden(self, batch_size, use_gpu=False):
        """
        Initialize the hidden state of the core network
        and the location vector.

        This is called once every time a new minibatch
        `x` is introduced.
        """
        dtype = torch.cuda.FloatTensor if self.use_gpu else torch.FloatTensor
        if self.rnn_type == 'RNN' or self.rnn_type == 'GRU':
            h = torch.zeros(batch_size, self.rnn_hidden)
            h = Variable(h).type(dtype)
            h_t = (h, 0)
        elif self.rnn_type == 'LSTM' or self.rnn_type == 'REV':
            h = torch.zeros(batch_size, self.rnn_hidden)
            h = Variable(h).type(dtype)
            c = torch.zeros(batch_size, self.rnn_hidden)
            c = Variable(c).type(dtype)
            h_t = (h, c)
        return h_t



class ActionNet(nn.Module):
    def __init__(self, input_size, output_size):
        """
        @param input_size: input size of the fc layer.
        @param output_size: output size of the fc layer.
        """
        super(PolicyNet, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        """
        Uses the last output of the decoder to output the final classification.
        @param h_t: (batch, rnn_hidden)
        @return a_t: (batch, output_size)
        """
        a_t = F.log_softmax(self.fc(h_t), dim=1)
        return a_t


class PolicyNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, use_gpu):
        """
        @param input_size: input size of the fc layer.
        @param output_size: output size of the fc layer.
        @param std: standard deviation of the normal distribution.
        """
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)

        self.combine = nn.Linear(2*hidden_size, output_size)

        if use_gpu:
            self.cuda()

    def forward(self, z, w0, w1, h_t):

        z = F.adaptive_avg_pool2d(z , (1, 1)).squeeze()
        z1 = F.relu(self.fc1(z*w0))
        z2 = F.relu(self.fc2(z*w1))

        logits = self.combine(torch.cat((z1, z2), dim=1))
        return F.softmax(logits)



class ModulatorNet(nn.Module):
    def __init__(self, input_size, output_size, use_gpu):
        """
        @param input_size: input size of the fc layer, total number of sampled symbols.
        @param output_size: output size of the fc layer, hidden vector dimension.
        """
        super(ModulatorNet, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        if use_gpu:
            self.cuda()

    def forward(self, z, w):
        z = F.adaptive_avg_pool2d(z , (1, 1)).squeeze()

        # attention
        z = z*w

        return self.fc(z)


class BaselineNet(nn.Module):
    """
    Regresses the baseline in the reward function
    to reduce the variance of the gradient update.

    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - h_t: the hidden state vector of the core network
      for the current time step `t`.

    Returns
    -------
    - b_t: a 2D vector of shape (B, 1). The baseline
      for the current time step `t`.
    """
    def __init__(self, input_size, output_size):
        super(BaselineNet, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        b_t = self.fc(h_t)
        return b_t




class PlayerNet(nn.Module):
    def __init__(self, args):
        """
        Initialize the recurrent attention model and its different components.
        """
        super(PlayerNet, self).__init__()
        rnn_inp_size = args.glimpse_hidden + args.loc_hidden
        self.narguments = args.narguments
        self.std = args.std

        self.rnn = core_network(rnn_inp_size, args.rnn_hidden, args.use_gpu)
        self.modulator_net = ModulatorNet(args.rnn_hidden, 2, 
                                        args.std, 
                                        args.use_gpu)

        self.policy_net = PolicyNet(args.rnn_hidden, 2, 
                                        args.std, 
                                        args.use_gpu)

        self.classifier = ActionNet(args.rnn_hidden, args.num_class)
        self.baseline_net = BaselineNet(args.rnn_hidden, 1)


        if args.use_gpu:
            self.cuda()


    def step(self, x, lt_agent, lt_other, h_t):
        """
        @param x: image. (batch, channel, height, width)
        @param l_t: location trial. (batch, 2)
        @param h_t: last hidden state. (batch, rnn_hidden)
        @return h_t: next hidden state. (batch, rnn_hidden)
        @return l_t: next location trial. (batch, 2)
        @return b_t: baseline for step t. (batch)
        @return log_pi: probability for next location trial. (batch)
        """
        glimpse = self.glimpse_net(x, lt_agent, lt_other)
        h_t = self.rnn(glimpse, h_t)
        mu, l_t = self.location_net(h_t[0])
        b_t = self.baseline_net(h_t[0]).squeeze()

        log_pi = torch.log(l_t)
        # Note: log(p_y*p_x) = log(p_y) + log(p_x)
        log_pi = log_pi.sum(dim=1)

        return h_t, l_t, b_t, log_pi

    def forward(self, x, l_t):
        """
        @param x: image. (batch, channel, height, width)
        @param l_t: initial location. (batch, 2)

        @return hiddens: hidden states (output) of rnn. (batch, narguments, rnn_hidden)
        @return locs: locations. (batch, 2)*narguments
        @return baselines: (batch, narguments)
        @return log_pi: probabilities for each location trial. (batch, narguments)
        """
        batch_size = x.shape[0]
        h_t = self.rnn.init_hidden(batch_size)

        locs = []
        baselines = []
        log_pi = []
        for t in range(self.narguments):
            h_t, l_t, b_t, p_t = self.step(x, l_t, h_t)
            locs.append(l_t)
            baselines.append(b_t)
            log_pi.append(p_t)

        log_probas = self.classifier(h_t[0])
        baselines = torch.stack(baselines).transpose(1, 0)
        log_pi = torch.stack(log_pi).transpose(1, 0)
        return locs, baselines, log_pi #, log_probas