"""[summary]

all state, action are saved as np instead of tensor
tensor are saved in extra, which is [output] in this file
all var saved as numpy as default instead of list or tensor
TODO: change to hydra

TODO: different output in select_action and update
"""

import hydra
import pyrootutils
from pprint import pprint


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
import gymnasium as gym
import sys
sys.path.append(sys.path[0]+"/..")
from utils.experience import *
from utils.delay import DelayedRoboticEnv
import time
import shutil



# print(sys.path)

cur_path = os.path.split(__file__)[0]


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
    def __init__(self, env, cfg, name="PolicyGradient"):
        state_num = env.observation_space.shape[0]
        action_num = env.action_space.n
        memory_capacity = cfg.memory_capacity
        gamma = cfg.gamma
        self.gamma = gamma
        self.state_num = state_num
        self.memory = Experience(memory_capacity)
        self.name = name
        # TODO: select action net
        self.net = PolicyNet(input_num=state_num, output_num=action_num)
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=cfg.learning_rate)

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


root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)

@hydra.main(version_base=None, config_path=root / "configs", config_name="vpg.yaml")
def main(cfg):
    pprint(cfg)
    print("finished!") # TODO print nicely
    '''init'''
    from utils.wandb import init_wandb
    wandb = init_wandb(cfg)
    env = gym.make(cfg.env_name)
    if cfg.delay_steps > 0: # apply delay
        env = DelayedRoboticEnv(env, delay_steps=cfg.delay_steps)
    agent = Agent(env, cfg)
    if cfg.mode == 'train': # train
        for epis_cnt in range(cfg.epis_num):
            s, info = env.reset()
            is_done = False
            step_cnt = 0
            # for step_cnt in range(cfg.max_step):
            while not is_done:
                a, output = agent.select_action(s)
                s_, r, is_done, _, info = env.step(a) 
                if step_cnt > cfg.max_step_num or is_done:  # end of episode
                    is_done = True
                agent.memory.push_trans(Transition(
                    s, a, r, s_, is_done, extra={'output':output}))
                s = s_
                step_cnt += 1
            wandb.log({'Reward': agent.memory[-1].total_reward})
            agent.update()
            if epis_cnt % cfg.log_epis == 0 and epis_cnt != 0:
                print("Episode {:>5}/{:<5}: mean_reward = {:.2f} rooling_mean_reward = {:.2f}".\
                format(
                    epis_cnt,
                    cfg.epis_num, 
                    agent.memory.total_reward_mean, 
                    sum(agent.memory.total_reward_list[-cfg.log_epis:])/cfg.log_epis
                ))
        agent.save(cfg.model_save_dir, print_log=True) # save model
    elif cfg.mode == 'test': # test
        agent.load(cfg.model_load_dir, print_log=True)
        for epis_cnt in range(cfg.test_epis_num):
            s = env.reset()
            is_done = False
            step_cnt = 0
            while not is_done:
                env.render()
                time.sleep(1.0/24.0)
                a, output = agent.select_action(s)
                s_, r, is_done, _ = env.step(a)
                # s_, r, is_done, _ = env.step(action_list[a]) # for regression task
                if step_cnt > cfg.max_step_num or is_done:
                    is_done = True
                agent.memory.push_trans(Transition(
                    s, a, r, s_, is_done, extra={'output':output}))
                s = s_
                step_cnt += 1
            print("Test episode {:>5}/{:<5}: total_reward = {:.2f}".\
            format(epis_cnt+1,cfg.test_epis_num, agent.memory[-1].total_reward))
    '''note area'''
    print(agent.memory.describe())
    print(agent.memory.df)
    agent.memory.save(cfg.memory_save_path, print_log=True)
    agent.memory.load(cfg.memory_load_path, print_log=True)
    agent.save(cfg.model_save_dir, print_log=True)
    agent.load(cfg.model_load_dir, print_log=True)
    print(agent.memory.info())
    
if __name__ == "__main__":
    main()