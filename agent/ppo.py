import os
import pdb
from torch.distributions import Categorical
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import pandas as pd
from collections import deque, namedtuple
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import numpy as np
import gym
import sys
sys.path.append(sys.path[0]+"/..")
from utils.experience import *
from torch.utils.tensorboard import SummaryWriter
import time
import shutil
from torch import Tensor
import math
# import roboschool
# import pybullet

# print(sys.path)

cur_path = os.path.split(__file__)[0]
EPS = 1e-10

class args(object):
    # path
    save_model_path = './data/model/latest/'
    save_log_path = './data/log/latest/'
    memory_save_path = cur_path+"/../data/latest_mem.npy"
    memory_load_path = cur_path+"/../data/latest_mem.npy"
    tensorboard_log_path = cur_path+"/../data/log/tensorboard_log/latest"
    model_save_dir = cur_path+"/../data/model/latest/"
    model_load_dir = cur_path+"/../data/model/latest/"

    # env_name = 'LunarLanderContinuous-v2'
    # env_name = 'MountainCarContinuous-v0'
    # env_name = 'Ant-v2'
    # env_name = 'Swimmer-v2'
    env_name = 'InvertedPendulum-v2'
    seed = 1234

    log_num_round = 1
    model_save_num_round = 10

    update_round = 2000 # 2000
    num_epoch = 10 # 10
    minibatch_size = 256 # 256
    max_step_per_round = 2000
    end_trans = 1000000 # 1000000 
    batch_size = 2048 # 2048
    clip = 0.2 # 0.2
    gamma = 0.995 # 0.995
    lamda = 0.97 # 0.97
    loss_coeff_value = 0.5
    loss_coeff_entropy = 0.01
    lr = 3e-4
    # tricks
    layer_norm = True
    state_norm = True
    advantage_norm = True
    lossvalue_norm = True
    schedule_adam = 'linear' # 'linear'
    schedule_clip = 'linear' # 'linear'
    # layer_norm = False
    # state_norm = False
    # advantage_norm = False
    # lossvalue_norm = False
    # schedule_adam = '' # 'linear'
    # schedule_clip = '' # 'linear
    
'''class'''
Transition = namedtuple('Transition', ('state', 'value', 'action', 'logproba', 'mask', 'next_state', 'reward'))
class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, do_reverse=True):
        if do_reverse:
            return Transition(*zip(*reversed(self.memory)))
        else:
            return Transition(*zip(*self.memory))
            
    def sample_zip(self, do_reverse=True):
        if do_reverse:
            batch = Transition(*zip(*reversed(self.memory)))
        else:
            batch = Transition(*zip(*self.memory))
        s = Tensor(batch.state)
        value = Tensor(batch.value)
        a = Tensor(batch.action)
        logproba = Tensor(batch.logproba)
        mask = Tensor(batch.mask)
        s_ = Tensor(batch.next_state)
        r = Tensor(batch.reward)
        return s, value, a, logproba, mask, s_, r
        

    def __len__(self):
        return len(self.memory)

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, layer_norm=True):
        super(ActorCritic, self).__init__()
        
        self.actor_fc1 = nn.Linear(num_inputs, 64)
        self.actor_fc2 = nn.Linear(64, 64)
        self.actor_fc3 = nn.Linear(64, num_outputs)
        self.actor_logstd = nn.Parameter(torch.zeros(1, num_outputs))

        self.critic_fc1 = nn.Linear(num_inputs, 64)
        self.critic_fc2 = nn.Linear(64, 64)
        self.critic_fc3 = nn.Linear(64, 1)

        if layer_norm:
            self.layer_norm(self.actor_fc1, std=1.0)
            self.layer_norm(self.actor_fc2, std=1.0)
            self.layer_norm(self.actor_fc3, std=0.01)

            self.layer_norm(self.critic_fc1, std=1.0)
            self.layer_norm(self.critic_fc2, std=1.0)
            self.layer_norm(self.critic_fc3, std=1.0)

    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)

    def forward(self, states):
        """
        run policy network (actor) as well as value network (critic)
        :param states: a Tensor2 represents states
        :return: 3 Tensor2
        """
        action_mean, action_logstd = self._forward_actor(states)
        critic_value = self._forward_critic(states)
        return action_mean, action_logstd, critic_value

    def _forward_actor(self, states):
        x = torch.tanh(self.actor_fc1(states))
        x = torch.tanh(self.actor_fc2(x))
        action_mean = self.actor_fc3(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        return action_mean, action_logstd

    def _forward_critic(self, states):
        x = torch.tanh(self.critic_fc1(states))
        x = torch.tanh(self.critic_fc2(x))
        critic_value = self.critic_fc3(x)
        return critic_value

    def select_action(self, action_mean, action_logstd, return_logproba=True):
        """
        given mean and std, sample an action from normal(mean, std)
        also returns probability of the given chosen
        """
        action_std = torch.exp(action_logstd)
        action = torch.normal(action_mean, action_std)
        if return_logproba:
            logproba = self._normal_logproba(action, action_mean, action_logstd, action_std)
        return action, logproba

    @staticmethod
    def _normal_logproba(x, mean, logstd, std=None):
        if std is None:
            std = torch.exp(logstd)

        std_sq = std.pow(2)
        logproba = - 0.5 * math.log(2 * math.pi) - logstd - (x - mean).pow(2) / (2 * std_sq)
        return logproba.sum(1)

    def get_logproba(self, states, actions):
        """
        return probability of chosen the given actions under corresponding states of current network
        :param states: Tensor
        :param actions: Tensor
        """
        action_mean, action_logstd = self._forward_actor(states)
        logproba = self._normal_logproba(actions, action_mean, action_logstd)
        return logproba

class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape

class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape


'''main'''
def main(args):
    env = gym.make(args.env_name)
    dim_states = env.observation_space.shape[0]
    dim_actions = env.action_space.shape[0]

    env.seed(args.seed)
    torch.manual_seed(args.seed)

    # actor = Actor(dim_states, dim_actions)
    # baseline = Baseline(dim_states)
    network = ActorCritic(dim_states, dim_actions, layer_norm=args.layer_norm)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    writer = SummaryWriter(args.tensorboard_log_path)
    running_state = ZFilter((dim_states,), clip=5.0)

    global_trans = 0
    global_epis = 0
    global_update_epoch = 0
    lr_now = args.lr
    clip_now = args.clip
    r_trans = []

    for update_round_cnt in range(args.update_round): # loop[update]
        # 1. rollout data
        memory = Memory()
        num_trans = 0 # count number of steps, when reachs args.batch_size, finish rolling out, start updating
        while num_trans < args.batch_size: # loop[episode]
            s = env.reset()
            r_sum = 0
            # TODO add running state trick
            if args.state_norm:
                s = running_state(s)
            for t in range(args.max_step_per_round): # loop[step]
                a_mean, a_logstd, value = network(Tensor(s).unsqueeze(0))
                a, logproba = network.select_action(a_mean, a_logstd)
                a = a.data.numpy()[0]
                logproba = logproba.data.numpy()[0]
                s_, r, is_done, _ = env.step(a)
                mask = 0 if is_done else 1 # TODO if not finished, mask is not 0?
                if args.state_norm:
                    s_ = running_state(s_)
                memory.push(Transition(
                    s, value, a, logproba, mask, s_, r)
                )
                s = s_
                r_sum += r
                if is_done: break

            writer.add_scalar('Episode/Reward', r_sum, global_epis)
            writer.add_scalar('Episode/Reward-Trans', r_sum, global_trans)
            writer.add_scalar('Episode/Epis Length', t, global_epis)
            writer.add_scalar('Episode/Reward_Ave_Step', r_sum/t, global_epis)
            r_trans.append((r_sum, global_trans))
            num_trans += (t+1)
            global_trans += (t+1)
            global_epis += 1
            if global_trans > args.end_trans: return {
                'env': args.env_name,
                'r_trans': r_trans
            }
        # 2. prepare data
        # s, a, a_mean, a_logstd, mask, s_, r = memory.sample_zip()
        s, value, a, oldlogproba, mask, s_, r = memory.sample_zip(do_reverse=False) # TODO reverse
        batch_size = len(memory)
        # GAE
        returns = Tensor(batch_size)
        deltas = Tensor(batch_size)
        advantages = Tensor(batch_size)
        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(batch_size)):
            returns[i] = r[i] + args.gamma * prev_return * mask[i]
            deltas[i] = r[i] + args.gamma * prev_value * mask[i] - value[i]
            # ref: https://arxiv.org/pdf/1506.02438.pdf (generalization advantage estimate)
            advantages[i] = deltas[i] + args.gamma * args.lamda * prev_advantage * mask[i]
            prev_return = returns[i]
            prev_value = value[i]
            prev_advantage = advantages[i]
        writer.add_histogram("update_epoch/return", returns, update_round_cnt)
        writer.add_histogram("update_epoch/reward", r, update_round_cnt)
        writer.add_histogram("update_epoch/advantages", advantages, update_round_cnt)
        writer.add_histogram("update_epoch/value", value, update_round_cnt)
        writer.add_histogram("update_epoch/deltas", deltas, update_round_cnt)
        
        if args.advantage_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)

        # 3. update
        for i_epoch in range(int(args.num_epoch * batch_size / args.minibatch_size)):
            global_update_epoch += 1
            # sample from current batch
            minibatch_ind = np.random.choice(batch_size, args.minibatch_size, replace=False)
            minibatch_states = s[minibatch_ind]
            minibatch_actions = a[minibatch_ind]
            minibatch_oldlogproba = oldlogproba[minibatch_ind]
            minibatch_newlogproba = network.get_logproba(minibatch_states, minibatch_actions)
            minibatch_advantages = advantages[minibatch_ind]
            minibatch_returns = returns[minibatch_ind]
            minibatch_newvalues = network._forward_critic(minibatch_states).flatten()

            ratio =  torch.exp(minibatch_newlogproba - minibatch_oldlogproba)
            surr1 = ratio * minibatch_advantages
            surr2 = ratio.clamp(1 - clip_now, 1 + clip_now) * minibatch_advantages
            loss_surr = - torch.mean(torch.min(surr1, surr2))

            # not sure the value loss should be clipped as well 
            # clip example: https://github.com/Jiankai-Sun/Proximal-Policy-Optimization-in-Pytorch/blob/master/ppo.py
            # however, it does not make sense to clip score-like value by a dimensionless clipping parameter
            # moreover, original paper does not mention clipped value 
            if args.lossvalue_norm:
                minibatch_return_6std = 6 * minibatch_returns.std()
                loss_value = torch.mean((minibatch_newvalues - minibatch_returns).pow(2)) / minibatch_return_6std
            else:
                loss_value = torch.mean((minibatch_newvalues - minibatch_returns).pow(2))

            loss_entropy = torch.mean(torch.exp(minibatch_newlogproba) * minibatch_newlogproba)

            total_loss = loss_surr + args.loss_coeff_value * loss_value + args.loss_coeff_entropy * loss_entropy
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # writer.add_scalar('Loss/', torch.mean(total_loss).detach().numpy(), global_update_epoch)
            writer.add_scalar('Loss/total_loss', torch.mean(total_loss).detach().numpy(), global_update_epoch)
            writer.add_scalar('Loss/loss_surr', torch.mean(loss_surr).detach().numpy(), global_update_epoch)
            writer.add_scalar('Loss/loss_value', torch.mean(loss_value).detach().numpy(), global_update_epoch)
            writer.add_scalar('Loss/loss_entropy', torch.mean(loss_entropy).detach().numpy(), global_update_epoch)

        # change some parameters
        if args.schedule_clip == 'linear':
            ep_ratio = 1 - (update_round_cnt / args.update_round)
            clip_now = args.clip * ep_ratio
        if args.schedule_adam == 'linear':
                ep_ratio = 1 - (update_round_cnt / args.update_round)
                lr_now = args.lr * ep_ratio
                # set learning rate
                # ref: https://stackoverflow.com/questions/48324152/
                for g in optimizer.param_groups:
                    g['lr'] = lr_now

        
        # log
        if update_round_cnt % args.log_num_round == 0:
            print("test log batch_size: {}".format(batch_size))

        # save model
        if update_round_cnt % args.model_save_num_round == 0:
            path = args.save_model_path
            if os.path.exists(path):
                shutil.rmtree(path)
                print("Model saved path already exists, rewrite with new: "+path)
            os.mkdir(path)
            torch.save(network, path+"ppo_net")
            test = torch.load(path+"ppo_net")
            print("Model saved at dir {0}".format(path))

        
def save_res(data, path):
    np.save(path, data, allow_pickle=True)

def test_on_envs():
    envs = ['HalfCheetah-v2', 'Hopper-v2', 'InvertedDoublePendulum-v2',
    'InvertedPendulum-v2', 'Reacher-v2', 'Swimmer-v2',
    'Walker2d-v2']
    folder = './data/res/'
    for env in envs:
        arg = args()
        arg.env_name = env
        res = main(arg)
        save_res(res, folder + arg.env_name)
        print('Saved: '+folder + arg.env_name)

test_on_envs()