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
from src.modules import VectorQuantizer2DHS

class Debate(nn.Module):
    def __init__(self, args):
        """
        Initialize the recurrent attention model and its different components.
        """
        super(Debate, self).__init__()
        self.nagents = args.nagents
        self.M = args.M
        self.narguments = args.narguments
        self.discretizer = VectorQuantizer2DHS(args)
        self.agents = [RecurrentAttentionAgent(args, i) \
                            for i in range(self.nagents)]

        self.use_gpu = args.use_gpu
        self.contrastive = args.contrastive
        self.rl_weightage = args.rl_weightage

        self.name = '{}_{}_{}_{}x{}_{}_{}_{}_{}_{}_{}'.format(args.model, 
                                        args.rnn_type, 
                                        args.narguments, 
                                        args.patch_size, 
                                        args.patch_size, 
                                        args.glimpse_scale, 
                                        args.nglimpses,
                                        args.glimpse_hidden,
                                        args.loc_hidden,
                                        args.rnn_hidden,
                                        args.reward_weightage)

        self.reward_fns = [None for _ in range(self.nagents)]

        # self.classifier = ActionNet(args.rnn_hidden, args.num_class)

        # debate_paramaeters = list(self.classifier.parameters())
        # for i in range(self.nagents):
        #     debate_paramaeters += list(self.agents[i].parameters())

        # self.optimizer = torch.optim.Adam (debate_paramaeters, 
        #                         lr=args.init_lr, weight_decay=1e-5)
       

            

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


    def step(self, x):
        batch_size = x.shape[0]

        # init lists for location, hidden vector, 
        # baseline and log_pi
        # dimensions:    (narguments + 1, nagents, *)
        b_ts = [[ None for _ in range(self.nagents)] for _ in range(self.narguments + 1)]
        log_pis = [[ None for _ in range(self.nagents)] for _ in range(self.narguments + 1)]
        l_ts = [[ agent.initLoc(batch_size) for agent in self.agents] for _ in range(self.narguments + 1)]
        h_t = [agent.init_rnn_hidden(batch_size) for agent in self.agents]

        # process
        for t in range(1, self.narguments + 1):
            for ai, agent in enumerate(self.agents):
                lt_const = copy.deepcopy(l_ts[t-1])
                lt_agent = lt_const[ai]
                del lt_const[ai]
                
                lt_agent = lt_agent
                if len(lt_const) > 0:
                    lt_const = torch.cat([lt_.detach().unsqueeze(1) for lt_ in lt_const], dim=1) # batch_size, nagents -1, ...
                    lt_const.requires_grad = False
                h_t[ai], l_t, b_t, log_pi = agent.forwardStep(x, 
                                                            lt_agent, 
                                                            lt_const, 
                                                            h_t[ai])

                l_ts[t][ai] = l_t; 
                b_ts[t][ai] = b_t; 
                log_pis[t][ai] = log_pi

        # remove first time stamp 
        b_ts = b_ts[1:]; log_pis = log_pis[1:]; l_ts = l_ts[1:]
        return l_ts, b_ts, log_pis, h_t


    def reformat(self, x, ai):
        """
        @param x: data. [narguments, [nagents, (batch_size, ...)]]
        returns x (batch_size, narguments, ...)
        """
        agents_data = []
        for t in range(self.narguments):
            agents_data.append(x[t][ai].unsqueeze(0))
        agent_data = torch.cat(agents_data, dim=0)
        return torch.transpose(agent_data, 0, 1)


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

        l_ts, b_ts, log_pis, h_t = self.step(x)

        # Judge prediction
        with torch.no_grad():
            jpred = torch.max(self.judge(x), 1)[1].detach()

        # Prediction Loss common classifier
        # preds:    (batch)
        # ht = 0
        # for ai in range(self.nagents):
        #     ht += h_t[ai][0]
        # ht = ht*1.0/self.nagents

        # if self.contrastive:
        #     with torch.no_grad():
        #         log_probs = self.classifier(ht)
        #     loss_pred = 0
        # else:
        #     log_probs = self.classifier(ht)

        # preds = torch.max(log_probs, 1)[1]


        # individual agent optimizer
        logs = {}
        for ai, agent in enumerate(self.agents):
            baselines = self.reformat(b_ts, ai)
            locations =  self.reformat(l_ts, ai)
            log_pi = self.reformat(log_pis, ai)


            # Classifier Loss
            log_probs_agent = agent.classifier(h_t[ai][0])
            preds_agent = torch.max(log_probs_agent, 1)[1]
            loss_classifier = F.nll_loss(log_probs_agent, y)


            # Baseline Loss
            # reward:          (batch, num_glimpses)
            log_probs = log_probs_agent
            log_probs = F.softmax(log_probs).detach()
            values, preds = torch.max(log_probs, 1)
            reward = self.reward_fns[ai](log_probs, jpred)
            reward = reward.unsqueeze(1).repeat(1, self.narguments)
            loss_baseline = F.mse_loss(baselines, reward)


            # Reinforce Loss
            # TODO: thought https://github.com/kevinzakka/recurrent-visual-attention/issues/10#issuecomment-378692338
            adjusted_reward = reward - baselines.detach()
            loss_reinforce = torch.mean(-log_pi*adjusted_reward)
            # loss_reinforce = torch.sum(-logpi_*adjusted_reward, dim=1)
            # loss_reinforce = torch.mean(loss_reinforce)


            # sum up into a hybrid loss
            loss = self.rl_weightage*(loss_reinforce + loss_baseline) + loss_classifier

            # calculate accuracy wrt judge network
            # with torch.no_grad():
            #     preds_ = torch.max(self.classifier(h_t[ai][0]).detach(), 1)[1]

            correct = (preds_agent == jpred).float()
            acc = 100 * (correct.sum() / len(y))

            agent.optStep(loss)
            logs[ai] = {}
            logs[ai]['x'] = x
            logs[ai]['y'] = y
            logs[ai]['acc'] = acc
            logs[ai]['loss'] = loss
            logs[ai]['preds'] = preds_agent
            logs[ai]['dpreds'] = preds
            logs[ai]['jpred'] = jpred
            logs[ai]['locs'] = locations
           


        # For train individual agents for backprop
        # above optimizer step updates parameters with 
        # obtained gradients, so need to reinit this for 
        # joint classifier training            
        # l_ts, b_ts, log_pis, h_t = self.step(x)
        # ht = 0
        # for ai in range(self.nagents):
        #     ht += h_t[ai][0]

        # log_probs = self.classifier(ht)
        # loss_pred = F.nll_loss(log_probs, jpred)
        # self.optimizer.zero_grad()
        # loss_pred.backward(retain_graph=True)
        # self.optimizer.step()

        return logs


    @torch.no_grad()
    def forward_test(self, x_orig, y, epoch):
        # duplicate 10 times
        x = x_orig.clone().repeat(self.M, 1, 1, 1)
        
        # Main forwarding step
        # locs:             (batch*M, 2)*num_glimpses
        # baselines:        (batch*M, num_glimpses)
        # log_pi:           (batch*M, num_glimpses)
        # log_probas:       (batch*M, num_class)
        l_ts, b_ts, log_pis, h_t = self.step(x)

        # Judge prediction
        jpred = torch.max(self.judge(x_orig), 1)[1].detach()

        # Prediction Loss common classifier
        # preds:    (batch)
        # ht = 0
        # for ai in range(self.nagents):
        #     ht += h_t[ai][0]
        # ht = ht*1.0/self.nagents

        # log_probs = self.classifier(ht)
        # log_probs = log_probs.view(self.M, -1, log_probs.shape[-1])
        # log_probs = torch.mean(log_probs, dim=0)
        # preds = torch.max(log_probs, 1)[1]

        logs = {}
        for ai, agent in enumerate(self.agents):
            baselines = self.reformat(b_ts, ai)
            locations =  self.reformat(l_ts, ai)
            log_pi = self.reformat(log_pis, ai)


            # Average           
            log_probs_agent = agent.classifier(h_t[ai][0])
            log_probs_agent = log_probs_agent.contiguous().view(self.M, -1, log_probs_agent.shape[-1])
            log_probs_agent = torch.mean(log_probs_agent, dim=0)
            preds_agent = torch.max(log_probs_agent, 1)[1]

            baselines = baselines.contiguous().view(self.M, -1, baselines.shape[-1])
            baselines = torch.mean(baselines, dim=0)

            log_pi = log_pi.contiguous().view(self.M, -1, log_pi.shape[-1])
            log_pi = torch.mean(log_pi, dim=0)


            # classifier loss
            classifier_loss = F.nll_loss(log_probs_agent, y)
            
            
            # Prediction Loss & Reward
            # preds:    (batch)
            # reward:   (batch)
            log_probs = log_probs_agent
            log_probs = F.softmax(log_probs).detach()
            values, preds = torch.max(log_probs, 1)
            reward = self.reward_fns[ai](log_probs, jpred)

            # Baseline Loss
            # reward:          (batch, num_glimpses)
            reward = reward.unsqueeze(1).repeat(1, self.narguments)
            loss_baseline = F.mse_loss(baselines, reward)

            # Reinforce Loss
            adjusted_reward = reward - baselines.detach()
            loss_reinforce = torch.mean(-log_pi*adjusted_reward)

            # TODO: thought https://github.com/kevinzakka/recurrent-visual-attention/issues/10#issuecomment-378692338
            # loss_reinforce = torch.sum(-log_pi*adjusted_reward, dim=1)
            # loss_reinforce = torch.mean(loss_reinforce)

            # sum up into a hybrid loss
            loss = self.rl_weightage*(loss_reinforce + loss_baseline) + classifier_loss

            # with torch.no_grad():
            #     preds_ = torch.max(torch.mean(self.classifier(h_t[ai][0]).detach().view(self.M, \
            #                                     -1, log_probs.shape[-1]), 0), 1)[1]

            # calculate accuracy
            correct = (preds_agent == jpred).float()
            acc = 100 * (correct.sum() / len(y))

            self.confusion_meters[ai].add(preds_agent.data.view(-1), jpred.data.view(-1))
            
            logs[ai] = {}
            logs[ai]['x'] = x
            logs[ai]['y'] = y
            logs[ai]['acc'] = acc
            logs[ai]['loss'] = loss
            logs[ai]['preds'] = preds_agent
            logs[ai]['dpreds'] = preds
            logs[ai]['jpred'] = jpred
            logs[ai]['locs'] = locations
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
        # self.classifier.load_state_dict(ckpt['classifier'])
        return ckpt['epoch']


    def get_state_dict(self):
        state = {}
        for ai, agent in enumerate(self.agents):
            state[ai] = {} 
            state[ai]['model_state_dict'] = agent.state_dict()
            state[ai]['optim_state_dict'] = agent.optimizer.state_dict()

        # state['classifier'] = self.classifier.state_dict()
        return state


    def lr_schedular(self, factor, patience, mode):
        return [ReduceLROnPlateau(agent.optimizer,
                                            factor = factor,
                                            patience=patience,
                                            mode = mode) for agent in self.agents]
