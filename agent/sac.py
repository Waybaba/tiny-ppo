import pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
from typing import Any, Dict, List, Optional, Type, Union, Tuple
import numpy as np
import torch
import hydra
from pprint import pprint
import torch
import wandb
import numpy as np
import gymnasium as gym
# import gym
import sys
from utils.delay import DelayedRoboticEnv
from tianshou.data import Batch, ReplayBuffer
from tianshou.data import Batch, ReplayBuffer, to_torch, to_torch_as
from tianshou.policy import BasePolicy, PGPolicy, SACPolicy
from tianshou.utils import RunningMeanStd
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import Actor, ActorProb, Critic
import tianshou
from torch.utils.tensorboard import SummaryWriter
import rich
import utils
from functools import partial


import warnings
warnings.filterwarnings('ignore')
from tianshou.policy import DDPGPolicy
from tianshou.exploration import BaseNoise
from torch.distributions import Independent, Normal
from copy import deepcopy
from tianshou.trainer import OffpolicyTrainer


root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
@hydra.main(version_base=None, config_path=str(root / "configs"), config_name="train.yaml")	
def main(cfg):
	def make_env(env_cfg):
		env = gym.make(env_cfg.name)
		env = DelayedRoboticEnv(env, env_cfg.delay)
		return env
	# init
	utils.print_config_tree(cfg, resolve=True)
	wandb.init(project=cfg.task_name, tags=cfg.tags, config=utils.config_format(cfg),dir=cfg.output_dir)
	cfg = hydra.utils.instantiate(cfg)
	utils.seed_everything(cfg.seed) # TODO add env seed
	# env & not & policy
	train_envs = tianshou.env.DummyVectorEnv([partial(make_env, cfg.env) for _ in range(cfg.env.train_num)])
	test_envs = tianshou.env.DummyVectorEnv([partial(make_env, cfg.env) for _ in range(cfg.env.test_num)])
	env = make_env(cfg.env)
	if hasattr(cfg, "actor_use_rnn") and cfg.actor_use_rnn == True: # use rnn for actor or not
		assert cfg.net is None, "actor_use_rnn == True, net should be None"
		actor = cfg.actor(state_shape=env.observation_space.shape, action_shape=env.action_space.shape).to(cfg.device)
	else:
		net = cfg.net(env.observation_space.shape)
		actor = cfg.actor(net, env.action_space.shape).to(cfg.device)
	actor_optim = cfg.actor_optim(actor.parameters())
	net_c1 = cfg.net_c1(env.observation_space.shape, action_shape=env.action_space.shape)
	critic1 = cfg.critic1(net_c1).to(cfg.device)
	critic1_optim = cfg.critic1_optim(critic1.parameters())
	net_c2 = cfg.net_c2(env.observation_space.shape, action_shape=env.action_space.shape)
	critic2 = cfg.critic2(net_c2).to(cfg.device)
	critic2_optim = cfg.critic2_optim(critic2.parameters())
	policy = cfg.policy(
		actor, actor_optim,
		critic1, critic1_optim,
		critic2, critic2_optim,
		action_space=env.action_space,
	)
	# collector
	train_collector = cfg.collector.train_collector(policy, train_envs)
	test_collector = cfg.collector.test_collector(policy, test_envs)
	# train
	logger = tianshou.utils.WandbLogger(config=cfg)
	logger.load(SummaryWriter(cfg.output_dir))
	trainer = cfg.trainer(
		policy, train_collector, test_collector, 
		stop_fn=lambda mean_reward: mean_reward >= 10000,
		logger=logger,
	)
	for epoch, epoch_stat, info in trainer:
		to_log = {
			"key/reward": epoch_stat["test_reward"],
			# "key/length": epoch_stat["test/episode_length"],
		}
		to_log.update(epoch_stat)
		to_log.update(info)
		wandb.log(to_log)
	wandb.finish()
	
	
if __name__ == "__main__":
	main()
