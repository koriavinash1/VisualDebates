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

from src.hpenalty import hessian_penalty
from torch.distributions import Categorical



class GumbelQuantizer(nn.Module):
    """
    credit to @karpathy: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py (thanks!)
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """
    def __init__(self, args):
        super().__init__()

        # n_e = 128, 
        # e_dim = 16, 
        # beta = 0.9, 
        # ignorezq = False,
        # disentangle = True,
        # remap=None, 
        # unknown_index="random",
        # sane_index_shape=False, 
        # legacy=True, 
        # sigma = 0.1,
        # straight_through=True,
        # kl_weight=5e-4, 
        # temp_init=1.0


        self.n_e = args.nconcepts
        self.e_dim = args.cdim
        self.beta = args.beta
        self.legacy = args.legacy
        self.sigma = args.sigma
        self.use_gpu = args.use_gpu
        self.temperature = args.temperature
        self.modulated_channels = args.modulated_channels
        self.straight_through = True

        # self.nfeatures = args.nfeatures
        # self.modulated_channels = args.modulated_channels


        # self.dis_modulator = torch.nn.Conv2d(self.nfeatures,
        #                                self.modulated_channels,
        #                                kernel_size=1,
        #                                stride=1,
        #                                )

        self.proj = nn.Linear(self.e_dim, self.n_e, 1)
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)


    def forward(self, z,
                    temp=None, 
                    rescale_logits=False, 
                    return_logits=False):
        # force hard = True when we are in eval mode, as we must quantize. 
        # actually, always true seems to work


        # z = self.dis_modulator(z)
        batch_size = z.shape[0]

        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        
        z_flattened = z.view(-1, self.e_dim)
        cb = self.embedding.weight

        hard = self.straight_through if self.training else True
        temp = self.temperature if temp is None else temp

        logits = self.proj(z_flattened)

        soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=hard)
        

        z_q = torch.einsum('b n, n d -> b d', soft_one_hot, cb)
        z_q = z_q.view(z.shape)

        # + kl divergence to the prior loss
        qy = F.softmax(logits, dim=1)
        kl_loss = torch.sum(qy * torch.log(qy + 1e-10), dim=1).mean()

        ind = soft_one_hot.argmax(dim=1)

        sampled_idx = torch.zeros(batch_size*self.n_e).to(z.device)
        sampled_idx[ind] = 1
        sampled_idx = sampled_idx.view(batch_size, self.n_e)

        return (z_q, kl_loss,
                    (sampled_idx, ind.view(batch_size, -1)))




class VectorQuantizer(nn.Module):
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

        # remap = args.remap
        # unknown_index = args.unknown_index
        self.n_e = args.nconcepts
        self.e_dim = args.cdim
        self.beta = args.beta
        self.use_gpu = args.use_gpu
        self.legacy = args.legacy
        self.nfeatures = args.nfeatures
        self.modulated_channels = args.modulated_channels
        self.temperature = args.temperature

        self.dis_modulator = torch.nn.Conv2d(self.nfeatures,
                                       self.modulated_channels,
                                       kernel_size=1,
                                       stride=1,
                                       )
        self.norm = nn.BatchNorm2d(self.modulated_channels, affine=True)

        print ("USING CB MODULATOR..............")

        # self.epsilon = 1e-4
        # self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}
        # self.min_norm = 1e-15


        # uniformly sampled initialization
        self.embedding = nn.Embedding(self.n_e, 
                                        self.e_dim)
        self.embedding.weight.data.uniform_(-1 / self.n_e, 1 / self.n_e)


        # contrastive loss
        # self.contrastive_loss = SimCLR_Loss()

        # points_in_manifold = torch.Tensor(np.random.uniform(0, 1, (self.n_e, self.edim)))
        # self.embedding.weight.data.copy_(points_in_manifold).requires_grad=True
        # self.hsreg = lambda x: torch.cat([ torch.norm(x[i]).unsqueeze(0) for i in range(x.shape[0])], 0)
        # self.r = torch.nn.Parameter(torch.ones(self.n_e)).to(self.embedding.weight.device)
        self.ed = lambda x: torch.cat([torch.norm(x[i]).unsqueeze(0) for i in range(x.shape[0])], 0)
        



        # remap
        # self.remap = remap
        # if self.remap is not None:
        #     self.register_buffer("used", torch.tensor(np.load(self.remap)))
        #     self.re_embed = self.used.shape[0]
        #     self.unknown_index = unknown_index # "random" or "extra" or integer
        #     if self.unknown_index == "extra":
        #         self.unknown_index = self.re_embed
        #         self.re_embed = self.re_embed+1

        #     print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
        #           f"Using {self.unknown_index} for unknown indices.")
        # else:
        #     self.re_embed = self.n_e
        # self.clamp_class = Clamp()


    def HLoss(self, x):
        x = x.view(-1, self.nfeatures, x.shape[-1]).mean(-1)
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(dim = 1)
        return torch.mean(b)

    def forward(self, z_,
                    temp=None, 
                    rescale_logits=False, 
                    return_logits=False):
        
        z = self.dis_modulator(z_)
        # z = self.norm(z)


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
        cosine_distance = d1/ed3

        # numerator = torch.exp(cosine_distance / self.temperature)        
        # denominator = torch.mean(torch.exp(cosine_distance/ self.temperature))    
            
        contrastive_loss = torch.mean(cosine_distance)
        # if self.use_gpu: edx = edx.cuda()
        # edx = torch.clamp(edx, min=-0.99999, max=0.99999)
        # d1 = torch.acos(edx)
        

        # min_distance = torch.kthvalue(d1, 2, 0)
        # total_min_distance = torch.mean(min_distance[0])
        # codebookvariance = torch.mean(torch.var(d1, 1))

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))
        
        min_encoding_indices = torch.argmin(d, dim=1)
    

        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # hsw = self.hsreg(self.embedding.weight)
        # hsw = torch.mean(torch.square(self.r - hsw))

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) 
            loss += torch.mean((z_q - z.detach()) ** 2) 
        else:
            loss = torch.mean((z_q.detach() - z) ** 2)  
            loss += self.beta * torch.mean((z_q - z.detach()) ** 2)


        loss += contrastive_loss
        loss += hessian_penalty(self.dis_modulator, z=z_, G_z = z)


        # contrastive_sum = 0
        # for i in range(0, z.shape[1]):
        #     for j in range(i, z.shape[1]): # symmetric loss function
        #         contrastive_sum += F.cosine_similarity(z[:, i, ...].view(-1, self.e_dim), 
        #                                                     z[:, j, ...].view(-1, self.e_dim)).mean()
        # loss += contrastive_sum
        # loss = self.HLoss(z_q) #self.embedding.weight)

        # disentanglement_loss = codebookvariance - total_min_distance
        # if self.disentangle:
        #     loss += hsw
        #     loss += disentanglement_loss


        # preserve gradients
        z_q = z + (z_q - z).detach()


        sampled_idx = torch.zeros(z.shape[0]*self.n_e).to(z.device)
        sampled_idx[min_encoding_indices] = 1
        sampled_idx = sampled_idx.view(z.shape[0], self.n_e)


        return (z_q, loss,
                    (sampled_idx, min_encoding_indices.view(z.shape[0], -1)))



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



class PlayerClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        @param input_size: input size of the fc layer.
        @param output_size: output size of the fc layer.
        """
        super(PlayerClassifier, self).__init__()
        self.fc = nn.Linear(hidden_size, output_size)
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(output_size, output_size)

    def forward(self, z, arg, h_t):
        """
        Uses the last output of the decoder to output the final classification.
        @param h_t: (batch, rnn_hidden)
        @return a_t: (batch, output_size)
        """

        # mask z
        arg = torch.clip(torch.sum(arg, 1), 0, 1)  
        z = z * arg

        #==================
        h_t = F.relu(self.fc(h_t))
        zd = F.relu(self.fc1(z))

        f = F.relu(self.fc2(h_t + zd))
        a_t = F.log_softmax(f, dim=1)
        return a_t



class PolicyNet(nn.Module):
    def __init__(self, concept_size, 
                        input_size, 
                        output_size, 
                        hidden_size, 
                        nclasses, 
                        use_gpu,
                        temperature=3):
        """
        @param input_size: total number of sampled symbols from a codebook.
        @param concept_size: dimension of an individual sampled symbol.
        @param hidden_size: hidden unit size of core recurrent/GRU network.
        @param output_size: output dimension of core recurrent/GRU network.
        @param std: standard deviation of the normal distribution.
        """
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size//2)
        self.fc2 = nn.Linear(input_size, output_size//2)
        self.fc3 = nn.Linear (hidden_size, output_size//2)
        self.fc4 = nn.Linear(nclasses, output_size//2)

        self.combine = nn.Linear(2*output_size, output_size)

        self.temperature = temperature
        if use_gpu:
            self.cuda()

    def forward(self, z,  y, arg1, arg2, h):  
            
        arg1 = z*arg1
        arg2 = z*arg2

        arg1_info = F.relu(self.fc1(arg1))
        arg2_info = F.relu(self.fc2(arg2))
        decision_info = F.relu(self.fc4(y))
        history = F.relu(self.fc3(h))

        logits = self.combine(torch.cat((arg1_info, 
                                            arg2_info,
                                            decision_info, 
                                            history), dim=1))

        return F.softmax(logits/self.temperature, -1)



class ModulatorNet(nn.Module):
    def __init__(self, input_size, output_size, use_gpu):
        """
        @param concept_size: dimension of an individual sampled symbol.
        @param output_size: output size of the fc layer, hidden vector dimension.
        """
        super(ModulatorNet, self).__init__()
        self.fc = nn.Linear (input_size, output_size)

        self.final = nn.Linear (output_size, output_size)

        if use_gpu:
            self.cuda()

    def forward(self, z, arg):

        arg = arg*z
        arg_info = F.relu(self.fc(arg))

        return F.relu(self.final(arg_info))


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
        b_t = self.fc(F.relu(h_t))
        return b_t


class QClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        """
        @param input_size: input size of the fc layer.
        @param output_size: output size of the fc layer.
        """
        super(QClassifier, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        """
        Uses the last output of the decoder to output the final classification.
        @param h_t: (batch, rnn_hidden)
        @return a_t: (batch, output_size)
        """
        h_t = F.relu(h_t)
        a_t = F.log_softmax(self.fc(h_t), dim=1)
        return a_t




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
        self.modulator_net = ModulatorNet(input_size = args.modulated_channels, 
                                        output_size = args.rnn_input_size,
                                        use_gpu = args.use_gpu)
        self.policy_net = PolicyNet(concept_size = args.cdim, 
                                    input_size = args.modulated_channels, 
                                    output_size = args.modulated_channels, 
                                    hidden_size = args.rnn_hidden,
                                    nclasses = args.n_class,
                                    use_gpu = args.use_gpu,
                                    temperature=args.softmax_temperature)


        self.classifier = PlayerClassifier(args.modulated_channels, args.rnn_hidden, args.num_class)
        self.baseline_net = BaselineNet(args.rnn_hidden, 1)


        if args.use_gpu:
            self.cuda()


    def step(self, z, z_idxs, y, arg1_t, arg2_t, h_t):
        """
        @param z: image. (batch, channel, height, width)
        @param arg1_t:
        @param arg2_t:
        @param h_t:
        """

        argument_prob = self.policy_net(z, y, arg1_t, arg2_t, h_t[0])


        # poisioning
        # argument_prob_copy = argument_prob.clone().detach().cpu().numpy()
        # scale = 1 - torch.min(torch.min(argument_prob_copy - arg1_t),
        #                         torch.min(argument_prob_copy - arg2_t))
        # scale =  scale.detach().cpu().numpy()
        # scale = 0.1
        # noise = torch.from_numpy(np.random.normal(
        #                     scale=scale, 
        #                     size=argument_prob.shape))
        # noise = Variable(noise.float()).type_as(argument_prob).to(argument_prob.device)


        # update logits with noise vectors
        # for encouraging exploration
        # argument_prob += noise 
        # argument_prob = torch.clip(argument_prob, 0.0, 1.0)


        argument_dist = Categorical(argument_prob)
        arg_current = argument_dist.sample()

        # Note: log(p_y*p_x) = log(p_y) + log(p_x)
        # log_pi = torch.log(torch.clip(0.0001 + argument_prob, 0.0, 1.0))
        # log_pi = log_pi.sum(dim=1)
        log_pi = argument_dist.log_prob(arg_current)


        arg_current_one_hot = 1.0*F.one_hot(arg_current, num_classes=argument_prob.shape[-1])


        for i, _ in enumerate(z.clone()):
            arg_current_one_hot[i, z_idxs[i] == z_idxs[i][arg_current[i]]] = 1


        z_current = self.modulator_net(z, arg_current_one_hot)
        h_t = self.rnn(z_current, h_t)
        b_t = self.baseline_net(h_t[0]).squeeze()

        return h_t, arg_current_one_hot, b_t, log_pi, argument_prob


# loss functions
# directly taken from https://zablo.net/blog/post/understanding-implementing-simclr-guide-eli5-pytorch/
class ContrastiveLossELI5(nn.Module):
    def __init__(self, temperature=0.5, verbose=False):
        super().__init__()
        self.register_buffer("temperature", torch.tensor(temperature))
        self.verbose = verbose
            
    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        batch_size = emb_i.shape[0]
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        if self.verbose: print("Similarity matrix\n", similarity_matrix, "\n")
            
        def l_ij(i, j):
            z_i_, z_j_ = representations[i], representations[j]
            sim_i_j = similarity_matrix[i, j]
            if self.verbose: print(f"sim({i}, {j})={sim_i_j}")
                
            numerator = torch.exp(sim_i_j / self.temperature)
            one_for_not_i = torch.ones((2 * batch_size, )).to(emb_i.device).scatter_(0, torch.tensor([i]).to(emb_i.device), 0.0)
            if self.verbose: print(f"1{{k!={i}}}",one_for_not_i)
            
            denominator = torch.sum(
                one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature)
            )    
            if self.verbose: print("Denominator", denominator)
                
            loss_ij = torch.log(numerator / denominator)
            # loss_ij = -torch.log(numerator / denominator)
            if self.verbose: print(f"loss({i},{j})={loss_ij}\n")
                
            return loss_ij.squeeze(0)

        N = batch_size
        loss = 0.0
        for k in range(0, N):
            loss += l_ij(k, k + N) + l_ij(k + N, k)
        return 1.0 / (2*N) * loss



class SimCLR_Loss(nn.Module):
    def __init__(self, max_batch_size=64, temperature=0.5):
        super(SimCLR_Loss, self).__init__()
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(max_batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        
        batch_size = z_i.shape[0]


        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        
        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        
        #SIMCLR
        labels = torch.from_numpy(np.array([0]*N)).reshape(-1).to(positive_samples.device).long() #.float()
        
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        
        return loss