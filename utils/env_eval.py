import torch
import gym
import time
from torch import nn
from torch.distributions import Categorical
from torch.autograd import Variable
import sys
sys.path.append(sys.path[0]+"/..")


class args(object):
    # env_name = "MountainCarContinuous-v0"
    env_name = 'LunarLanderContinuous-v2'
    model_path = "./data/model/share/actor_net"
    round_num = 2
    max_step_per_round = 2000
    
class Actor(nn.Module):
    def __init__(self, dim_states, dim_actions):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(dim_states, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc_mean = nn.Linear(args.hidden_size, dim_actions)
        self.fc_logstd = nn.Parameter(args.initial_policy_logstd * torch.ones(1, dim_actions), requires_grad=False)

    def forward(self, states):
        """
        given a states returns the action distribution (gaussian) with mean and logstd 
        :param states: a Tensor2 represents states
        :return: Tensor2 action mean and logstd  
        """
        x = torch.relu(self.fc1(states))
        x = torch.relu(self.fc2(x))
        action_mean = self.fc_mean(x)
        action_logstd = self.fc_logstd.expand_as(action_mean)
        return action_mean, action_logstd

    @ staticmethod
    def select_action(action_mean, action_logstd):
        """
        given mean and std, sample an action from normal(mean, std)
        also returns probability of the given chosen
        :param action_mean: Tensor2
        :param action_logstd: Tensor2
        :return: Tensor2 action
        """
        action_std = torch.exp(action_logstd)
        action = torch.normal(action_mean, action_std)
        return action

    @staticmethod
    def normal_logproba(x, mean, logstd, std=None):
        if std is None:
            std = torch.exp(logstd)

        std_sq = std.pow(2)
        logproba = - 0.5 * math.log(2 * math.pi) - logstd - (x - mean).pow(2) / (2 * std_sq)
        
        return logproba.sum(1).view(-1, 1)


if __name__ == "__main__":
    actor = torch.load(args.model_path)# load model
    env = gym.make(args.env_name)
    
    for round_cnt in range(args.round_num):
        s = env.reset()
        for step_cnt in range(args.max_step_per_round):
            time.sleep(0.01)
            a_mean, a_logstd = actor(torch.Tensor(s).unsqueeze(0))
            a = actor.select_action(a_mean, a_logstd)
            a = a.data.numpy()[0]
            s, r, is_done, _ = env.step(a)
            env.render()
            print("Epis:{}, Step:{}, Reward:{}".format(round_cnt, step_cnt, r))
            if is_done: break
            
        