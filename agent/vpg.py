"""[summary]

all state, action are saved as np instead of tensor
tensor are saved in extra, which is [output] in this file
all var saved as numpy as default instead of list or tensor
TODO: different output in select_action and update

"""


import os
import pdb
from torch.distributions import Categorical
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import pandas as pd
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

# print(sys.path)

cur_path = os.path.split(__file__)[0]

class st(object):
    mode = 'train'
    learning_rate = 0.001  # optimizer learning rate
    gamma = 0.99  # reward discount
    memory_capacity = 100000
    save_model_path = './data/model/latest'
    save_log_path = './data/log/latest'
    epis_num = 2000
    max_step_num = 500
    log_epis = 50
    memory_save_path = cur_path+"/../data/latest_mem.npy"
    memory_load_path = cur_path+"/../data/latest_mem.npy"
    tensorboard_log_path = cur_path+"/../data/log/tensorboard_log/latest"
    model_save_dir = cur_path+"/../data/model/latest/"
    model_load_dir = cur_path+"/../data/model/latest/"
    test_epis_num = 2


'''Net'''


class PolicyNet(nn.Module):
    def __init__(self, input_num, output_num):
        super(PolicyNet, self).__init__()
        self.state_space = input_num
        self.action_space = output_num
        # TODO: 改一下名，然后对hidden layer数量进行修改
        self.l1 = nn.Linear(self.state_space, 128, bias=False)
        self.l2 = nn.Linear(128, self.action_space, bias=False)

    def forward(self, x):
        model = torch.nn.Sequential(
            self.l1,
            nn.Dropout(p=0.6),
            nn.ReLU(),
            self.l2,
            nn.Softmax(dim=-1)
        )
        return model(x)


'''Policy Gradient Agent'''

class Agent:
    def __init__(self, state_num, action_num, memory_capacity=10000, gamma=0.99,
                 name="PolicyGradient"
                 ):
        self.gamma = gamma
        self.state_num = state_num
        self.memory = Experience(memory_capacity)
        self.name = name
        # TODO: select action net
        self.net = PolicyNet(input_num=state_num, output_num=action_num)
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=st.learning_rate)

    def select_action(self, s):
        '''select_action'''
        s = torch.from_numpy(s).type(torch.FloatTensor)
        output = self.net(Variable(s))
        c = Categorical(output)
        a = c.sample()
        c_log = c.log_prob(a).reshape(1)
        return a.data.item(), output

    def update(self):
        '''optimize_model with episode data.'''
        def get_c_log(s, a):
            s = torch.tensor(s).type(torch.FloatTensor)
            output = self.net(s)  # TODO
            # output_ = torch.tensor([each[0] for each in extra]) # TODO
            c = Categorical(output)
            c_log = c.log_prob(torch.tensor(a))
            return c_log
        self.memory.calculate_gain(self.gamma)
        epis = self.memory[-1]
        s, a, r, s_, is_done, extra = zip(*self.memory[-1])
        gains = [each['gain'] for each in extra]
        gains = torch.tensor(gains)
        gain_norm = (gains - gains.mean()) / (gains.std() +
                                              np.finfo(np.float32).eps)  # eps是一个精度要求下的最小值，防止出现分母为0
        c_log = get_c_log(s, a)
        loss = torch.sum(
            torch.mul(c_log, Variable(gain_norm)).mul(-1)
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path, print_log=False):
        if os.path.exists(path): 
            shutil.rmtree(path)
            print("Model saved path already exists, rewrite with new: "+path)
        os.mkdir(path)
        torch.save(self.net.state_dict(), path+"policy_net")
        if print_log:
            print("Model saved at dir {0}".
                format(path)
                )
        
    def load(self, path, print_log=False):
        if not os.path.exists(path): 
            print("Model load path not exsit: "+path)
            exit()
        self.net.load_state_dict(torch.load(path+"policy_net"))
        if print_log:
            print("Model load from dir {0}".
                  format(path)
                  )

    # utils

if __name__ == "__main__":

    '''init'''
    env = gym.make('CartPole-v1')
    # env = gym.make('MountainCarContinuous-v0')
    # action_num = 10
    # action_list = [(i*np.array(env.action_space.low) + (action_num-i)*np.array(env.action_space.high))/action_num for i in range(action_num)] # use action_list[action_idx] to turn num into float
    # action_list = [(i**3)/(18**2) for i in action_list]
    state_num = env.observation_space.shape[0]
    action_num = env.action_space.n
    writer = SummaryWriter(st.tensorboard_log_path)
    agent = Agent(
        state_num=state_num,
        action_num=action_num,
        memory_capacity=st.memory_capacity,
        gamma=st.gamma
    )
    
    '''main'''
    if st.mode == 'train': # train
        for epis_cnt in range(st.epis_num):
            s = env.reset()
            is_done = False
            step_cnt = 0
            # for step_cnt in range(st.max_step):
            while not is_done:
                a, output = agent.select_action(s)
                s_, r, is_done, _ = env.step(a) 
                # s_, r, is_done, _ = env.step(action_list[a]) # for regression task
                if step_cnt > st.max_step_num or is_done:  # end of episode
                    is_done = True
                agent.memory.push_trans(Transition(
                    s, a, r, s_, is_done, extra={'output':output}))
                s = s_
                step_cnt += 1
            writer.add_scalar('Reward', agent.memory[-1].total_reward, epis_cnt)
            agent.update()
            if epis_cnt % st.log_epis == 0 and epis_cnt != 0:
                print("Episode {:>5}/{:<5}: mean_reward = {:.2f} rooling_mean_reward = {:.2f}".\
                format(
                    epis_cnt,
                    st.epis_num, 
                    agent.memory.total_reward_mean, 
                    sum(agent.memory.total_reward_list[-st.log_epis:])/st.log_epis
                ))
        agent.save(st.model_save_dir, print_log=True) # save model
    elif st.mode == 'test': # test
        agent.load(st.model_load_dir, print_log=True)
        for epis_cnt in range(st.test_epis_num):
            s = env.reset()
            is_done = False
            step_cnt = 0
            while not is_done:
                env.render()
                time.sleep(1.0/24.0)
                a, output = agent.select_action(s)
                s_, r, is_done, _ = env.step(a)
                # s_, r, is_done, _ = env.step(action_list[a]) # for regression task
                if step_cnt > st.max_step_num or is_done:
                    is_done = True
                agent.memory.push_trans(Transition(
                    s, a, r, s_, is_done, extra={'output':output}))
                s = s_
                step_cnt += 1
            print("Test episode {:>5}/{:<5}: total_reward = {:.2f}".\
            format(epis_cnt+1,st.test_epis_num, agent.memory[-1].total_reward))
    

    '''note area'''
    # print(agent.memory.describe())
    # print(agent.memory.df)
    # agent.memory.save(st.memory_save_path, print_log=True)
    # agent.memory.load(st.memory_load_path, print_log=True)
    # agent.save(st.model_save_dir, print_log=True)
    # agent.load(st.model_load_dir, print_log=True)
    # print(agent.memory.info())


