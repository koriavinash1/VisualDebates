from argparse import Action
from torch.autograd import Variable
import torch.nn.functional as F

import torch
import copy
import torch.nn as nn
from src.agent import RecurrentAttentionAgent
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os, json
import torchnet as tnt
from src.clsmodel import afhq, mnist, stl10
from src.modules import ActionNet, VectorQuantizer2DHS

class Debate(nn.Module):
    def __init__(self, args):
        """
        Initialize the recurrent attention model and its different components.
        """
        super(Debate, self).__init__()
        self.nagents = 2
        self.M = args.M
        self.narguments = args.narguments
        self.discretizer = VectorQuantizer2DHS(args)
        self.agents = [RecurrentAttentionAgent(args, i) \
                            for i in range(self.nagents)]

        self.use_gpu = args.use_gpu
        self.contrastive = args.contrastive
        self.rl_weightage = args.rl_weightage

        self.name = 'Exp-{}-Debate: {}_{}_{}_{}_{}_{}'.format(
                                        args.name,
                                        args.rnn_type, 
                                        args.narguments, 
                                        args.nconcepts,
                                        args.rnn_hidden,
                                        args.rnn_input_size,
                                        args.reward_weightage)

        self.reward_fns = [None for _ in range(self.nagents)]


        if args.data_dir.lower().__contains__('mnist'):
            self.judge = mnist(32, pretrained=True)
        elif args.data_dir.lower().__contains__('stl10'):
            self.judge = stl10(32, pretrained=True)
        elif args.data_dir.lower().__contains__('afhq'):
            self.judge = afhq(3, pretrained=True)
        else:
            raise ValueError('Unknown dataset found')


        self.judge.eval()
        if self.use_gpu: self.judge = self.judge.cuda()

        print ("Judge Netwrok loaded...")

        os.makedirs(args.ckpt_dir, exist_ok=True)
        json.dump(vars(args), 
                open(os.path.join(args.ckpt_dir, 'args-{}.json'.format(self.name)), 'w'), 
                indent=4)


        self.confusion_meters = []
        for _ in range(self.nagents):
            confusion_meter = tnt.meter.ConfusionMeter(args.n_class, normalized=True)
            confusion_meter.reset()
            self.confusion_meters.append(confusion_meter)


        self.quantized_classifier = ActionNet(args.nfeatures, args.n_class)
        self.quantized_optimizer = torch.optim.Adam (list(self.discretizer.parameters()) + \
                                                    list(self.quantized_classifier.parameters()), 
                                                    lr=args.init_lr, weight_decay=1e-5)


    def fext(self, x):
        z, loss, (onehot_symbols, symbol_idx), \
            cbvar,  tdis, hsw, r = self.discretizer(self.judge.features(x))
        
        return z.detach(), symbol_idx, loss, cbvar, tdis, hsw, r


    def step(self, z):
        batch_size = z.shape[0]
        
        
        # init lists for location, hidden vector, baseline and log_pi
        # dimensions:    (narguments + 1, nagents, *)
        bs_t = [[ None for _ in range(self.nagents)] for _ in range(self.narguments + 1)]
        hs_t = [ agent.init_rnn_hidden(batch_size) for agent in self.agents ]
        args_idx_t = [[ agent.init_argument(batch_size) for agent in self.agents ] for _ in range(self.narguments + 1)]
        arg_dists_t = [[ None for agent in self.agents ] for _ in range(self.narguments + 1)]
        logpis_t = [[ None for _ in range(self.nagents)] for _ in range(self.narguments + 1)]

        # process
        for t in range(1, self.narguments + 1):
            for ai, agent in enumerate(self.agents):
                args_idx_t_const = copy.deepcopy(args_idx_t[t-1])

                # mapping idx to args---------------------
                args_t_const = [torch.cat([z[i, _idx_, ...].unsqueeze(0) \
                                for i, _idx_ in enumerate(idx_)], 0) \
                                for idx_ in args_idx_t_const]
                args_t_agent = args_t_const[ai]
                del args_t_const[ai]
                
                if len(args_t_const) > 0:
                    args_t_const = torch.cat([arg_.unsqueeze(1) for arg_ in args_t_const], dim=1) # batch_size, nagents -1, ...
                    args_t_const = args_t_const.squeeze()
                    args_t_const.requires_grad = False


                hs_t[ai], arg_idx_t, b_t, log_pi, dist = agent.forwardStep(z, 
                                                            args_t_agent, 
                                                            args_t_const, 
                                                            hs_t[ai])

                args_idx_t[t][ai] = arg_idx_t; 
                arg_dists_t[t][ai] = dist; 
                logpis_t[t][ai] = log_pi
                bs_t[t][ai] = b_t; 

        # remove first time stamp 
        bs_t = bs_t[1:]; logpis_t = logpis_t[1:]; 
        args_idx_t = args_idx_t[1:]; arg_dists_t = arg_dists_t[1:]

        return args_idx_t, arg_dists_t, bs_t, logpis_t, hs_t


    def reformat(self, x, ai, dist=False):
        """
        @param x: data. [narguments, [nagents, (batch_size, ...)]]
        returns x (batch_size, narguments, ...)
        """
        agents_data = []
        for t in range(self.narguments):
            if not dist: agents_data.append(x[t][ai].unsqueeze(0))
            else: agents_data.append(x[t][ai].probs.unsqueeze(0))

        agent_data = torch.cat(agents_data, dim=0)
        return torch.transpose(agent_data, 0, 1)

    def HLoss(self, x):
        """
        @param x: probability vector (probabilities of categorical disributions)
        """
        x = x.reshape(x.shape[0], -1)
        b = -1*F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        return torch.sum(b)


    def DDistance(self, dists):
        """
        @param dists: list of list of categorical distribtuions
        """
        dist0 =  self.reformat(dists, 0, True).transpose(0, 1)
        dist1 =  self.reformat(dists, 1, True).transpose(0, 1)

        sum_dist = 0
        for wv0 in dist0:
            dist = 1000
            for wv1 in dist1:
                dist_ = 0.5*(torch.norm(wv0 - wv1.detach()) +  torch.norm(wv0.detach() - wv1))
                if dist_ < dist:
                    dist = dist_
            sum_dist += dist

        return 0 #sum_dist*1.0/dist0.shape[0]



    def forward(self, x, y, lts, is_training=False, epoch=1):
        """
        @param x: image. (batch, channel, height, width)
        @param y: word indices. (batch, seq_len)
        """
        if self.use_gpu:
            x = x.cuda(); y = y.cuda()
        x = Variable(x); y = Variable(y)


        if not is_training:
            return self.forward_test(x, y, epoch)

        z, symbol_idxs, qloss, cbvar, dis, hsw, r = self.fext(x)

        # Judge prediction
        with torch.no_grad():
            jpred = torch.max(self.judge(x), 1)[1].detach()


        # quantized distillation
        self.quantized_optimizer.zero_grad()
        cqlog_probs = self.quantized_classifier(F.adaptive_avg_pool2d(z, (1,1)).squeeze())
        cq_loss = F.nll_loss(cqlog_probs, jpred)
        cq_loss = cq_loss + qloss
        cq_loss.backward()
        self.quantized_optimizer.step()



        args_idx_t, arg_dists_t, b_ts, log_pis, h_t = self.step(z.detach().clone())


        # TODO: include accurracy metric...



        # individual agent optimizer
        logs = {}
        inter_dis = self.DDistance(arg_dists_t)

        for ai, agent in enumerate(self.agents):
            baselines = self.reformat(b_ts, ai)
            args_idx =  self.reformat(args_idx_t, ai)
            args_dist = self.reformat(arg_dists_t, ai, True)
            log_pi = self.reformat(log_pis, ai)


            # Classifier Loss -> distillation loss
            log_probs_agent = agent.classifier(h_t[ai][0])
            preds_agent = torch.max(log_probs_agent, 1)[1]
            loss_classifier = F.nll_loss(log_probs_agent, jpred)


            # Baseline Loss
            # reward:          (batch, num_glimpses)
            log_probs = F.softmax(log_probs_agent).detach()

            reward = self.reward_fns[ai](z, symbol_idxs, args_idx, log_probs, jpred)
            reward = reward.unsqueeze(1).repeat(1, self.narguments)
            loss_baseline = F.mse_loss(baselines, reward)


            # Reinforce Loss
            # TODO: thought https://github.com/kevinzakka/recurrent-visual-attention/issues/10#issuecomment-378692338
            adjusted_reward = reward - baselines.detach()
            loss_reinforce = torch.mean(-log_pi*adjusted_reward)
            # loss_reinforce = torch.sum(-logpi_*adjusted_reward, dim=1)
            # loss_reinforce = torch.mean(loss_reinforce)


            # sum up into a hybrid loss
            intra_loss = self.HLoss(args_dist)
            
            regularization_loss = -1*(intra_loss + inter_dis)
            loss = self.rl_weightage*(loss_reinforce + loss_baseline) +\
                     loss_classifier + regularization_loss

            correct = (preds_agent == jpred).float()
            acc = 100 * (correct.sum() / len(y))

            agent.optStep(loss)


            # Logs record
            logs[ai] = {}
            logs[ai]['x'] = x
            logs[ai]['z'] = z
            logs[ai]['y'] = y
            logs[ai]['acc'] = acc
            logs[ai]['loss'] = loss
            logs[ai]['jpred'] = jpred
            logs[ai]['preds'] = preds_agent
            logs[ai]['arguments'] = args_idx

        return logs


    @torch.no_grad()
    def forward_test(self, x_orig, y, epoch):

        # duplicate M times
        x = x_orig.clone().repeat(self.M, 1, 1, 1)
        
        # Main forwarding step
        # locs:             (batch*M, 2)*num_glimpses
        # baselines:        (batch*M, num_glimpses)
        # log_pi:           (batch*M, num_glimpses)
        # log_probas:       (batch*M, num_class)



        z, sampled_idxs, qloss, cbvar, dis, hsw, r = self.fext(x)
        args_idx_t, arg_dists_t, b_ts, log_pis, h_t = self.step(z)
        z = z[0]; sampled_idxs = sampled_idxs[0]


        # Judge prediction
        jpred = torch.max(self.judge(x_orig), 1)[1].detach()

        logs = {}
        inter_dis = self.DDistance(arg_dists_t)


        for ai, agent in enumerate(self.agents):
            baselines = self.reformat(b_ts, ai)
            args_idx =  self.reformat(args_idx_t, ai)
            args_dist = self.reformat(arg_dists_t, ai, True)
            log_pi = self.reformat(log_pis, ai)


            # Average     
            args_idx = args_idx.contiguous().view((self.M, x_orig.shape[0]) + args_idx.shape[1:])
            args_idx = torch.mode(args_idx, dim = 0) # index max voting for M montecarlo iterations
             
            log_probs_agent = agent.classifier(h_t[ai][0])
            log_probs_agent = log_probs_agent.contiguous().view(self.M, -1, log_probs_agent.shape[-1])
            log_probs_agent = torch.mean(log_probs_agent, dim=0)
            preds_agent = torch.max(log_probs_agent, 1)[1]


            baselines = baselines.contiguous().view(self.M, -1, baselines.shape[-1])
            baselines = torch.mean(baselines, dim=0)


            log_pi = log_pi.contiguous().view(self.M, -1, log_pi.shape[-1])
            log_pi = torch.mean(log_pi, dim=0)


            # classifier loss -> distillation
            classifier_loss = F.nll_loss(log_probs_agent, jpred)
            
            
            # Prediction Loss & Reward
            # preds:    (batch)
            # reward:   (batch)
            log_probs = F.softmax(log_probs_agent).detach()
            reward = self.reward_fns[ai](z, symbol_idxs, args_idx, log_probs, jpred)

            # Baseline Loss
            # reward:          (batch, num_glimpses)
            reward = reward.unsqueeze(1).repeat(1, self.narguments)
            loss_baseline = F.mse_loss(baselines, reward)

            # Reinforce Loss
            # TODO: thought https://github.com/kevinzakka/recurrent-visual-attention/issues/10#issuecomment-378692338
            adjusted_reward = reward - baselines.detach()
            loss_reinforce = torch.mean(-log_pi*adjusted_reward)
            # loss_reinforce = torch.sum(-log_pi*adjusted_reward, dim=1)
            # loss_reinforce = torch.mean(loss_reinforce)


            # sum up into a hybrid loss
            intra_loss = self.HLoss(args_dist)
            regularization_loss = -1*(intra_loss + inter_dis)
            loss = self.rl_weightage*(loss_reinforce + loss_baseline) +\
                     classifier_loss + regularization_loss


             # calculate accuracy
            correct = (preds_agent == jpred).float()
            acc = 100 * (correct.sum() / len(y))

            self.confusion_meters[ai].add(preds_agent.data.view(-1), jpred.data.view(-1))
            
            logs[ai] = {}
            logs[ai]['x'] = x
            logs[ai]['z'] = z
            logs[ai]['y'] = y
            logs[ai]['acc'] = acc
            logs[ai]['loss'] = loss
            logs[ai]['preds'] = preds_agent
            logs[ai]['jpred'] = jpred
            logs[ai]['arguments'] = args_idx
            logs[ai]['con_mat'] = self.confusion_meters[ai]

        return logs


    def load_model(self, ckpt_dir, contrastive=False, best=False):
        suff = 'supportive' if not contrastive else 'contrastive'
        if best:
            print ('model name',self.name)
            path = os.path.join(ckpt_dir, self.name + suff + '_ckpt_best')
        else:
            path = os.path.join(ckpt_dir, self.name + suff + '_ckpt')
        
        print ('Model loaded from: {}'.format(path))
        ckpt = torch.load(path)
        for ai, agent in enumerate(self.agents):
            agent.load_state_dict(ckpt[ai]['model_state_dict'])
            agent.optimizer.load_state_dict(ckpt[ai]['optim_state_dict'])
        return ckpt['epoch']


    def get_state_dict(self):
        state = {}
        for ai, agent in enumerate(self.agents):
            state[ai] = {} 
            state[ai]['model_state_dict'] = agent.state_dict()
            state[ai]['optim_state_dict'] = agent.optimizer.state_dict()

        return state


    def lr_schedular(self, factor, patience, mode):
        return [ReduceLROnPlateau(agent.optimizer,
                                            factor = factor,
                                            patience=patience,
                                            mode = mode) for agent in self.agents]
