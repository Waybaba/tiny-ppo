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


class DefaultRLRunner:
	""" Runner for RL algorithms
	Work flow: 
	
	"""
	def __init__(self, cfg):
		print("RLRunner init start ...")
		self.cfg = cfg
		self.env = cfg.env # ! TODO replace cfg to self.cfg
		# init
		wandb.init(project=cfg.task_name, tags=cfg.tags, config=utils.config_format(cfg),dir=cfg.output_dir)
		utils.seed_everything(cfg.seed) # TODO add env seed
		self.train_envs = tianshou.env.DummyVectorEnv([partial(utils.make_env, cfg.env) for _ in range(cfg.env.train_num)])
		self.test_envs = tianshou.env.DummyVectorEnv([partial(utils.make_env, cfg.env) for _ in range(cfg.env.test_num)])
		self.env = utils.make_env(cfg.env) # to get obs_space and act_space
		print("RLRunner init end!")
		
	def run(self):
		raise NotImplementedError("RLRunner.run() not implemented!")

	def __exit__(self):
		wandb.close()
		print("RLRunner exit!")
		
class SACRunner(DefaultRLRunner):
	def __init__(self, cfg):
		print("SACRunner init start ...")
		super().__init__(cfg)
		print("SACRunner init end!")
