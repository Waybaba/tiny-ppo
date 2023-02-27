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
	# print config
	utils.print_config_tree(cfg, resolve=True)
	# initialize hydra cfg
	cfg = hydra.utils.instantiate(cfg)
	runner = cfg.runner(cfg)
	runner.run()

	
if __name__ == "__main__":
	main()
