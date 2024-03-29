from argparse import Action
from torch.autograd import Variable
import torch.nn.functional as F

import torch
import copy
import torch.nn as nn
from src.agent import RecurrentAttentionAgent

import os, json
import torchnet as tnt
from src.clsmodel import afhq, mnist, stl10, shapes
from src.modules import QClassifier, VectorQuantizer, GumbelQuantizer, ContrastiveLossELI5

def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool
    return model
    

class Debate(nn.Module):
    def __init__(self, args):
        """
        Initialize the recurrent attention model and its different components.
        """
        super(Debate, self).__init__()
        self.nagents = args.nagents
        self.M = args.M
        self.narguments = args.narguments
        self.agents = [RecurrentAttentionAgent(args, i) \
                            for i in range(self.nagents)]

        self.use_gpu = args.use_gpu
        self.contrastive = args.contrastive
        self.rl_weightage = args.rl_weightage

        self.name = 'Exp-{}-Debate:{}_{}_{}_{}_{}_{}'.format(
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
        elif args.data_dir.lower().__contains__('shapes'):
            self.judge = shapes(16, pretrained=True)
        elif args.data_dir.lower().__contains__('stl10'):
            self.judge = stl10(32, pretrained=True)
        elif args.data_dir.lower().__contains__('afhq'):
            self.judge = afhq(3, pretrained=True)
        else:
            raise ValueError('Unknown dataset found')


        self.judge.eval()
        if self.use_gpu: self.judge = self.judge.cuda()

        print ("Judge Netwrok loaded...")

        os.makedirs(os.path.join(args.ckpt_dir, self.name), exist_ok=True)
        json.dump(vars(args), 
                open(os.path.join(os.path.join(args.ckpt_dir, self.name), 'args-{}.json'.format(self.name)), 'w'), 
                indent=4)


        self.confusion_meters = []
        for _ in range(self.nagents):
            confusion_meter = tnt.meter.ConfusionMeter(args.n_class, normalized=True)
            confusion_meter.reset()
            self.confusion_meters.append(confusion_meter)


        # grad setting...
        set_requires_grad(self.judge, False)

        # common model def.============================
        if not args.gumbel:
            self.discretizer = VectorQuantizer(args)
            print ('Using traditional VQ')
        else:
            self.discretizer = GumbelQuantizer(args)
            print ('Using Gumbel VQ sampler')

        self.quantized_classifier = QClassifier(args.modulated_channels, 
                                                    args.n_class)

        self.contrastive_loss = ContrastiveLossELI5()


        # ==============================================
        self.quantized_optimizer = None
        if self.contrastive:
            self.quantized_classifier.eval()
            self.discretizer.eval()
            set_requires_grad(self.discretizer, False)
            set_requires_grad(self.quantized_classifier, False)

            for i in range(self.nagents):
                set_requires_grad(self.agents[i].classifier, False)
        else:
            self.quantized_optimizer = torch.optim.Adam (list(self.discretizer.parameters()) + \
                                                    list(self.quantized_classifier.parameters()), 
                                                    lr=1e-3, weight_decay=1e-5)


    def fext(self, x):
        z, loss, (onehot_symbols, symbol_idx) = self.discretizer(self.judge.features(x))
        return z, symbol_idx, loss, 


    def step(self, z, symbol_idxs, y):
        batch_size = z.shape[0]
        
        
        # init lists for location, hidden vector, baseline and log_pi
        # dimensions:    (narguments + 1, nagents, *)
        bs_t = [[ None for _ in range(self.nagents)] for _ in range(self.narguments + 1)]
        args_idx_t = [[ agent.init_argument(batch_size) for agent in self.agents ] for _ in range(self.narguments + 1)]
        arg_dists_t = [[ None for _ in self.agents ] for _ in range(self.narguments + 1)]
        logpis_t = [[ None for _ in range(self.nagents)] for _ in range(self.narguments + 1)]
        hs = [ agent.init_rnn_hidden(batch_size) for agent in self.agents ]

        # process
        argument_history = torch.zeros_like(args_idx_t[0][0])
        for t in range(1, self.narguments + 1):
            args_idx_t_const = [_arg_.clone() for _arg_ in args_idx_t[t-1]]
            for ai, agent in enumerate(self.agents):

                args_t_current = args_idx_t_const[ai]
                args_t_opponent = args_idx_t_const[1 - ai]

                argument_history += args_t_current
                argument_history += args_t_opponent
                argument_history[argument_history > 0] = 1
                # args_t_opponent.requires_grad = False
                # print("*********************")
                # import pdb;pdb.set_trace()

                hs[ai], arg_idx_t, b_t, log_pi, dist = agent.forwardStep(z, 
                                                                    symbol_idxs, y,
                                                                    argument_history,
                                                                    args_t_current, 
                                                                    args_t_opponent, 
                                                                    hs[ai])

                # import pdb;pdb.set_trace()

                args_idx_t[t][ai] = arg_idx_t
                if ai == 0: 
                    args_idx_t_const[ai] = arg_idx_t
                arg_dists_t[t][ai] = dist;
                logpis_t[t][ai] = log_pi
                bs_t[t][ai] = b_t


        # remove first time stamp 
        bs_t = bs_t[1:]; logpis_t = logpis_t[1:]; 
        args_idx_t = args_idx_t[1:]; arg_dists_t = arg_dists_t[1:]

        return args_idx_t, arg_dists_t, bs_t, logpis_t, hs


    def reformat(self, x, ai, dist=False):
        """
        @param x: data. [narguments, [nagents, (batch_size, ...)]]
        returns x (batch_size, narguments, ...)
        """
        agents_data = []
        for t in range(self.narguments):
            agents_data.append(x[t][ai].unsqueeze(0))
        agent_data = torch.cat(agents_data, dim=0)
        return torch.transpose(agent_data, 0, 1)


    def HLoss(self, x):
        """
        @param x: probability vector (probabilities of categorical disributions)
        """
        b = -1*(F.softmax(x, dim=-1) * F.log_softmax(x, dim=-1)).sum(-1)
        # b = -1*(F.softmax(b, dim=-1) * F.log_softmax(b, dim=-1)).sum(-1)
        # b = torch.max(F.softmax(x, dim=1), dim=1)[0] 
        return torch.mean(b)


    def DDistance(self, dists, arguments, ai):
        """
        @param dists: list of list of categorical distribtuions
        """
        dist0 =  torch.transpose(self.reformat(dists, 0, True), 0, 1)
        dist1 =  torch.transpose(self.reformat(dists, 1, True), 0, 1)
        
        dist0tilde = torch.mean(dist0, 0)
        dist1tilde = torch.mean(dist1, 0)
        
        dist0tilde = dist0tilde/torch.norm(dist0tilde, 1)
        dist1tilde = dist1tilde/torch.norm(dist1tilde, 1)


        if ai == 0: 
            euclidian_distance = torch.norm(dist0tilde - dist1tilde.clone().detach(), 1)
            cosine_distance = self.contrastive_loss(dist0tilde, dist1tilde.clone().detach())
        else:
            euclidian_distance = torch.norm(dist0tilde.clone().detach() - dist1tilde, 1)
            cosine_distance = self.contrastive_loss(dist0tilde.clone().detach(), dist1tilde)


        cosine_distance_intra = 0
        for ii in range(1, dist0.shape[0]):
            if ai == 0:
                cosine_distance_intra += torch.mean(F.cosine_similarity(dist0[ii-1], dist0[ii]))
            else:
                cosine_distance_intra += torch.mean(F.cosine_similarity(dist1[ii-1], dist1[ii]))


        intra_argument_distance = (1.0/dist0.shape[0]) * cosine_distance_intra
        inter_argument_distance = euclidian_distance  - cosine_distance

        return inter_argument_distance, intra_argument_distance


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

        
        # ================================= prelim.....
        # Judge prediction
        with torch.no_grad():
            jpred_probs = self.judge(x).detach()
            jpred = torch.argmax(jpred_probs, 1)
            # jpred = y


        # quantized distillation
        if not self.contrastive:
            self.quantized_optimizer.zero_grad()
            z_orig, symbol_idxs, qloss = self.fext(x)
            z = F.adaptive_avg_pool2d(z_orig , (1, 1)).squeeze()
            cqlog_probs = self.quantized_classifier(z)
            cq_loss = F.nll_loss(cqlog_probs, jpred)
            cq_loss_ = cq_loss + qloss
            cq_loss_.backward()
            self.quantized_optimizer.step()
        else:
            with torch.no_grad():
                z_orig, symbol_idxs, qloss = self.fext(x)
                z = F.adaptive_avg_pool2d(z_orig , (1, 1)).squeeze()
                cqlog_probs = self.quantized_classifier(z)
                cq_loss = F.nll_loss(cqlog_probs, jpred)
                
        cq_preds = torch.argmax(cqlog_probs, 1)
        cq_correct = (cq_preds == jpred).float()
        cq_acc = 100 * (cq_correct.sum() / len(y))
        
        # ==========================================

        z = z.clone().detach()

        args_idx_t, arg_dists_t, b_ts, log_pis, h_t = self.step(z, symbol_idxs, jpred_probs)


        logs = {}

        # zerosum-component:
        arguments = []; log_prob_agents = []; claims = []
        for ai, agent in enumerate(self.agents):
            argument  = self.reformat(args_idx_t, ai)

            if not self.contrastive:
                log_prob_agent = agent.classifier(z, argument, h_t[ai][0])
            else:
                with torch.no_grad():
                    log_prob_agent = agent.classifier(z, argument, h_t[ai][0])

            claims.append(torch.argmax(log_prob_agent.clone().detach(), 1))
            log_prob_agents.append(log_prob_agent)
            arguments.append(argument)


        # individual agent optimizer

        for ai, agent in enumerate(self.agents):
            baselines = self.reformat(b_ts, ai)
            log_pi = self.reformat(log_pis, ai)
            args_dist = self.reformat(arg_dists_t, ai, True)


            # Classifier Loss ---> distillation loss
            if self.contrastive:
                loss_classifier = 0 # torch.mean(F.cosine_similarity(h_t[ai][0], 
                                    #  h_t[1-ai][0].clone().detach()))
            else:
                loss_classifier = F.nll_loss(log_prob_agents[ai], jpred) # - \
                                    # torch.mean(F.cosine_similarity(h_t[ai][0], 
                                            # h_t[1-ai][0].clone().detach()))



            # Baseline Loss
            # reward:          (batch, num_glimpses)
            # import pdb;pdb.set_trace()
            reward, dpred = self.reward_fns[ai](z, symbol_idxs, arguments, claims, jpred)
            reward = reward.unsqueeze(1).repeat(1, self.narguments)
            loss_baseline = F.mse_loss(baselines, reward)


            # Reinforce Loss
            # TODO: thought https://github.com/kevinzakka/recurrent-visual-attention/issues/10#issuecomment-378692338
            # loss_reinforce = torch.mean(-log_pi*adjusted_reward)
            adjusted_reward = reward - baselines.detach()
            loss_reinforce = torch.sum(-log_pi*adjusted_reward, dim=1)
            loss_reinforce = torch.mean(loss_reinforce)


            # sum up into a hybrid loss
            intra_argument_entropy = self.HLoss(args_dist)
            inter_argument_distance, intra_argument_distance = self.DDistance(arg_dists_t, arguments, ai)
            
            # maximize intra argument cosine distance and entropy
            regularization_loss = intra_argument_entropy + intra_argument_distance

            if self.contrastive:
                regularization_loss -= inter_argument_distance
            else:
                regularization_loss += inter_argument_distance 
            

            loss = (loss_reinforce + loss_baseline) +\
                            loss_classifier + 0.1*regularization_loss



            correct = (claims[ai] == jpred).float()
            acc = 100 * (correct.sum() / len(y))

            dcorrect = (dpred == jpred).float()
            dacc = 100 * (dcorrect.sum() / len(y))


            agent.optStep(loss)


            # Logs record
            logs[ai] = {}
            logs[ai]['x'] = x
            logs[ai]['y'] = y
            logs[ai]['acc'] = acc
            logs[ai]['z'] = z_orig
            logs[ai]['dacc'] = dacc
            logs[ai]['loss'] = loss
            logs[ai]['jpred'] = jpred
            logs[ai]['dpred'] = dpred
            logs[ai]['cqacc'] = cq_acc
            logs[ai]['cqloss'] = cq_loss
            logs[ai]['preds'] = claims[ai]
            logs[ai]['z_idx'] = symbol_idxs
            logs[ai]['argument_dist'] = args_dist
            logs[ai]['arguments'] = arguments[ai]

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


        # ======================================= prelim........
        # Judge prediction
        jpred_probs_ = self.judge(x).detach()
        # jpred_probs_ = F.softmax(jpred_probs_, 1)
        jpred_probs = jpred_probs_.contiguous().view((self.M, x_orig.shape[0]) + jpred_probs_.shape[1:])
        jpred_probs = torch.mean(jpred_probs, 0)
        jpred = torch.argmax(jpred_probs, 1)

        z_orig, symbol_idxs_, qloss = self.fext(x)

        z_ = F.adaptive_avg_pool2d(z_orig, (1,1)).squeeze()
        
        cqlog_probs = self.quantized_classifier(z_)
        cq_loss = F.nll_loss(cqlog_probs, torch.argmax(cqlog_probs, 1))
        cq_preds = torch.argmax(cqlog_probs, 1)
        cq_correct = (cq_preds == torch.argmax(cqlog_probs, 1)).float()
        cq_acc = 100 * (cq_correct.sum() / len(cq_preds))



        z = z_.contiguous().view((self.M, x_orig.shape[0]) + z_.shape[1:])
        z = z[0]


        z_orig = z_orig.contiguous().view((self.M, x_orig.shape[0]) + z_orig.shape[1:])
        z_orig = z_orig[0]


        symbol_idxs = symbol_idxs_.contiguous().view((self.M, x_orig.shape[0]) + symbol_idxs_.shape[1:])
        symbol_idxs = symbol_idxs[0]

        args_idx_t, arg_dists_t, b_ts, log_pis, h_t = self.step(z_, symbol_idxs_, jpred_probs_)



        # zerosum-component:
        arguments = []; log_prob_agents = []; claims = []
        for ai, agent in enumerate(self.agents):
            argument_ = self.reformat(args_idx_t, ai)
            _argument_ = argument_.contiguous().view((self.M, x_orig.shape[0]) + argument_.shape[1:])
            _argument_ = torch.clip(torch.sum(_argument_, dim = 0), 0, 1)


            log_prob_agent = agent.classifier(z_, argument_, h_t[ai][0])
            log_prob_agent = log_prob_agent.contiguous().view(self.M, -1, log_prob_agent.shape[-1])
            log_prob_agent = torch.mean(log_prob_agent, dim=0)

            claims.append(torch.argmax(log_prob_agent, 1))
            log_prob_agents.append(log_prob_agent)
            arguments.append(_argument_)       

        
        # individual optimization
        logs = {}

        for ai, agent in enumerate(self.agents):
            baselines = self.reformat(b_ts, ai)
            log_pi = self.reformat(log_pis, ai)
            args_dist = self.reformat(arg_dists_t, ai, True)


            # Average     
            baselines = baselines.contiguous().view(self.M, -1, baselines.shape[-1])
            baselines = torch.mean(baselines, dim=0)

            log_pi = log_pi.contiguous().view(self.M, -1, log_pi.shape[-1])
            log_pi = torch.mean(log_pi, dim=0)



            # classifier loss -> distillation
            if self.contrastive:
                loss_classifier = 0 # torch.mean(F.cosine_similarity(h_t[ai][0], 
                                    #  h_t[1-ai][0].clone().detach()))
            else:
                loss_classifier = F.nll_loss(log_prob_agents[ai], jpred) # - \
                                    # torch.mean(F.cosine_similarity(h_t[ai][0], 
                                            # h_t[1-ai][0].clone().detach()))

            # Prediction Loss & Reward
            # preds:    (batch)
            # reward:   (batch)
            reward, dpred = self.reward_fns[ai](z, symbol_idxs, arguments, claims, jpred)

            # Baseline Loss
            # reward:          (batch, num_glimpses)
            reward = reward.unsqueeze(1).repeat(1, self.narguments)
            loss_baseline = F.mse_loss(baselines, reward)

            # Reinforce Loss
            # TODO: thought https://github.com/kevinzakka/recurrent-visual-attention/issues/10#issuecomment-378692338
            # loss_reinforce = torch.mean(-log_pi*adjusted_reward)
            adjusted_reward = reward - baselines.detach()
            loss_reinforce = torch.sum(-log_pi*adjusted_reward, dim=1)
            loss_reinforce = torch.mean(loss_reinforce)


            # sum up into a hybrid loss
            intra_argument_entropy = self.HLoss(args_dist)
            inter_argument_distance, intra_argument_distance = self.DDistance(arg_dists_t, arguments, ai)
            
            # maximize intra argument entropy and minimizing cosine distance
            regularization_loss = intra_argument_entropy + intra_argument_distance

            if self.contrastive:
                regularization_loss -= inter_argument_distance
            else:
                regularization_loss += inter_argument_distance 
            
            loss = (loss_reinforce + loss_baseline) +\
                            loss_classifier + 0.1*regularization_loss



             # calculate accuracy
            correct = (claims[ai] == jpred).float()
            acc = 100 * (correct.sum() / len(y))

            dcorrect = (dpred == jpred).float()
            dacc = 100 * (dcorrect.sum() / len(y))

            self.confusion_meters[ai].add(claims[ai].data.view(-1), jpred.data.view(-1))
            
            logs[ai] = {}
            logs[ai]['x'] = x
            logs[ai]['y'] = y
            logs[ai]['acc'] = acc
            logs[ai]['z'] = z_orig
            logs[ai]['dacc'] = dacc
            logs[ai]['loss'] = loss
            logs[ai]['jpred'] = jpred
            logs[ai]['dpred'] = dpred
            logs[ai]['cqacc'] = cq_acc
            logs[ai]['cqloss'] = cq_loss
            logs[ai]['preds'] = claims[ai]
            logs[ai]['z_idx'] = symbol_idxs
            logs[ai]['arguments'] = arguments[ai]
            logs[ai]['argument_dist'] = args_dist
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
            if not self.contrastive:
                agent.load_state_dict(ckpt[ai]['model_state_dict'])
                agent.optimizer.load_state_dict(ckpt[ai]['optim_state_dict'])
            else:
                agent.load_state_dict(ckpt[1]['model_state_dict'])

        self.quantized_classifier.load_state_dict(ckpt['qclassifier_state_dict'])
        self.discretizer.load_state_dict(ckpt['discretizer'])
        if not self.contrastive:
            self.quantized_optimizer.load_state_dict(ckpt['qclassifier_optim_state_dict'])
        return ckpt['epoch']


    def get_state_dict(self):
        state = {}
        for ai, agent in enumerate(self.agents):
            state[ai] = {} 
            state[ai]['model_state_dict'] = agent.state_dict()
            state[ai]['optim_state_dict'] = agent.optimizer.state_dict()

        state['qclassifier_state_dict'] = self.quantized_classifier.state_dict()
        state['discretizer'] = self.discretizer.state_dict()
        if not self.contrastive:
            state['qclassifier_optim_state_dict'] = self.quantized_optimizer.state_dict()
        return state
