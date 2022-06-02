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

from torch.distributions import Categorical


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
        self.n_e = args.nconcepts
        self.e_dim = args.cdim
        self.beta = args.beta
        self.legacy = args.legacy
        self.use_gpu = args.use_gpu
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


        self.hsreg = lambda x: torch.cat([ torch.norm(x[i]).unsqueeze(0) for i in range(x.shape[0])], 0)
        self.r = torch.nn.Parameter(torch.ones(self.n_e)).to(self.embedding.weight.device)
        self.ed = lambda x: torch.cat([torch.norm(x[i]).unsqueeze(0) for i in range(x.shape[0])], 0)
        

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
            r = torch.randint(0, self.re_embed,size=new[unknown].shape)
            if self.use_gpu:
                r.cuda()

            new[unknown] = r
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
        edx = d1/ed3
        if self.use_gpu: edx = edx.cuda()
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

        hsw = self.hsreg(self.embedding.weight)
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
        super(ActionNet, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        """
        Uses the last output of the decoder to output the final classification.
        @param h_t: (batch, rnn_hidden)
        @return a_t: (batch, output_size)
        """
        h_t = F.relu6(h_t)
        a_t = F.log_softmax(self.fc(h_t), dim=1)
        return a_t


class PolicyNet(nn.Module):
    def __init__(self, concept_size, input_size, output_size, hidden_size, use_gpu):
        """
        @param input_size: total number of sampled symbols from a codebook.
        @param concept_size: dimension of an individual sampled symbol.
        @param hidden_size: hidden unit size of core recurrent/GRU network.
        @param output_size: output dimension of core recurrent/GRU network.
        @param std: standard deviation of the normal distribution.
        """
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(concept_size, output_size//2)
        self.fc2 = nn.Linear(concept_size, output_size//2)
        self.fc3 = nn.Linear (hidden_size, output_size//2)
        # self.fc4 = nn.Linear (input_size, output_size//2)

        self.combine = nn.Linear(output_size//2, output_size)

        if use_gpu:
            self.cuda()

    def forward(self, z,  arg1, arg2, h):
        
        batch_size = z.shape[0]
        # z = F.adaptive_avg_pool2d(z , (1, 1)).squeeze()

        # z_info = F.relu6(self.fc4(z))
        arg1_info = F.relu6(self.fc1(arg1.view(batch_size, -1)))
        arg2_info = F.relu6(self.fc2(arg2.view(batch_size, -1)))
        history = F.relu6(self.fc3(h))

        # logits = self.combine(torch.cat((z_info, 
        #                                     arg1_info, 
        #                                     arg2_info, 
        #                                     history), dim=1))
        logits = self.combine(arg1_info + 
                                arg2_info + 
                                history)
        return F.softmax(logits)



class ModulatorNet(nn.Module):
    def __init__(self, concept_size, input_size, output_size, use_gpu):
        """
        @param concept_size: dimension of an individual sampled symbol.
        @param output_size: output size of the fc layer, hidden vector dimension.
        """
        super(ModulatorNet, self).__init__()
        self.fc = nn.Linear(concept_size, output_size)
        # self.fc2 = nn.Linear (input_size, output_size)

        self.final = nn.Linear (output_size, output_size)

        if use_gpu:
            self.cuda()

    def forward(self, z, arg_idx):
        batch_size = z.shape[0]

        arg = torch.cat([z[i, _idx_].unsqueeze(0) \
                        for i, _idx_ in enumerate(arg_idx.detach())], 0)
        arg_info = F.relu6(self.fc(arg.view(batch_size, -1)))

        # z = F.adaptive_avg_pool2d(z , (1, 1)).squeeze()
        # z_info = F.relu6(self.fc2(z))

        # return F.relu6(self.final(arg_info + z_info))
        return F.relu6(self.final(arg_info))


class BaselineNet(nn.Module):
    """
    Regresses the baseline in the reward function
    to reduce the variance of the gradient update.

    Args
    ----
    @param input_size: input size of the fc layer.
    @param output_size: output size of the fc layer.
    @param h_t: the hidden state vector of the core network
                for the current time step `t`.

    Returns
    -------
    @param b_t: a 2D vector of shape (B, 1). The baseline
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
        self.narguments = args.narguments

        self.rnn = core_network(args.rnn_input_size, 
                                    args.rnn_hidden, 
                                    args.use_gpu)
        self.modulator_net = ModulatorNet(concept_size = args.cdim,
                                        input_size = args.nfeatures, 
                                        output_size = args.rnn_input_size,
                                        use_gpu = args.use_gpu)
        self.policy_net = PolicyNet(concept_size = args.cdim, 
                                    input_size = args.nfeatures, 
                                    output_size = args.nfeatures, 
                                    hidden_size = args.rnn_hidden,
                                    use_gpu = args.use_gpu)


        self.classifier = ActionNet(args.rnn_hidden, args.num_class)
        self.baseline_net = BaselineNet(args.rnn_hidden, 1)


        if args.use_gpu:
            self.cuda()


    def step(self, z, arg1_t, arg2_t, h_t):
        """
        @param z: image. (batch, channel, height, width)
        @param arg1_t:
        @param arg2_t:
        @param h_t:
        """

        argument_prob = self.policy_net(z, arg1_t, arg2_t, h_t[0])
        argument_dist = Categorical(argument_prob)
        arg_current = argument_dist.sample()

        z_current = self.modulator_net(z, arg_current)
        h_t = self.rnn(z_current, h_t)
        b_t = self.baseline_net(h_t[0]).squeeze()

        log_pi = argument_dist.log_prob(arg_current)
        # Note: log(p_y*p_x) = log(p_y) + log(p_x)
        # log_pi = log_pi.sum(dim=1)

        return h_t, arg_current, b_t, log_pi, argument_dist