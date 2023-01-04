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
# import gymnasium as gym
import gym
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
from tianshou.exploration import BaseNoise, GaussianNoise
from torch.distributions import Independent, Normal
from copy import deepcopy


class DDPGPolicy(BasePolicy):
    """Implementation of Deep Deterministic Policy Gradient. arXiv:1509.02971.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic: the critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic_optim: the optimizer for critic network.
    :param float tau: param for soft update of the target network. Default to 0.005.
    :param float gamma: discount factor, in [0, 1]. Default to 0.99.
    :param BaseNoise exploration_noise: the exploration noise,
        add to the action. Default to ``GaussianNoise(sigma=0.1)``.
    :param bool reward_normalization: normalize the reward to Normal(0, 1),
        Default to False.
    :param int estimation_step: the number of steps to look ahead. Default to 1.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action) or empty string for no bounding.
        Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        actor: Optional[torch.nn.Module],
        actor_optim: Optional[torch.optim.Optimizer],
        critic: Optional[torch.nn.Module],
        critic_optim: Optional[torch.optim.Optimizer],
        tau: float = 0.005,
        gamma: float = 0.99,
        exploration_noise: Optional[BaseNoise] = GaussianNoise(sigma=0.1),
        reward_normalization: bool = False,
        estimation_step: int = 1,
        action_scaling: bool = True,
        action_bound_method: str = "clip",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            **kwargs
        )
        assert action_bound_method != "tanh", "tanh mapping is not supported" \
            "in policies where action is used as input of critic , because" \
            "raw action in range (-inf, inf) will cause instability in training"
        if actor is not None and actor_optim is not None:
            self.actor: torch.nn.Module = actor
            self.actor_old = deepcopy(actor)
            self.actor_old.eval()
            self.actor_optim: torch.optim.Optimizer = actor_optim
        if critic is not None and critic_optim is not None:
            self.critic: torch.nn.Module = critic
            self.critic_old = deepcopy(critic)
            self.critic_old.eval()
            self.critic_optim: torch.optim.Optimizer = critic_optim
        assert 0.0 <= tau <= 1.0, "tau should be in [0, 1]"
        self.tau = tau
        assert 0.0 <= gamma <= 1.0, "gamma should be in [0, 1]"
        self._gamma = gamma
        self._noise = exploration_noise
        # it is only a little difference to use GaussianNoise
        # self.noise = OUNoise()
        self._rew_norm = reward_normalization
        self._n_step = estimation_step

    def set_exp_noise(self, noise: Optional[BaseNoise]) -> None:
        """Set the exploration noise."""
        self._noise = noise

    def train(self, mode: bool = True) -> "DDPGPolicy":
        """Set the module in training mode, except for the target network."""
        self.training = mode
        self.actor.train(mode)
        self.critic.train(mode)
        return self

    def sync_weight(self) -> None:
        """Soft-update the weight for the target network."""
        self.soft_update(self.actor_old, self.actor, self.tau)
        self.soft_update(self.critic_old, self.critic, self.tau)

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]  # batch.obs_next: s_{t+n}
        target_q = self.critic_old(
            batch.obs_next,
            self(batch, model='actor_old', input='obs_next').act
        )
        return target_q

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        batch = self.compute_nstep_return(
            batch, buffer, indices, self._target_q, self._gamma, self._n_step,
            self._rew_norm
        )
        return batch

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "actor",
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which has 2 keys:

            * ``act`` the action.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        model = getattr(self, model)
        obs = batch[input]
        actions, hidden = model(obs, state=state, info=batch.info)
        return Batch(act=actions, state=hidden)

    @staticmethod
    def _mse_optimizer(
        batch: Batch, critic: torch.nn.Module, optimizer: torch.optim.Optimizer
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A simple wrapper script for updating critic network."""
        weight = getattr(batch, "weight", 1.0)
        current_q = critic(batch.obs, batch.act).flatten()
        target_q = batch.returns.flatten()
        td = current_q - target_q
        # critic_loss = F.mse_loss(current_q1, target_q)
        critic_loss = (td.pow(2) * weight).mean()
        optimizer.zero_grad()
        critic_loss.backward()
        optimizer.step()
        return td, critic_loss

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        # critic
        td, critic_loss = self._mse_optimizer(batch, self.critic, self.critic_optim)
        batch.weight = td  # prio-buffer
        # actor
        actor_loss = -self.critic(batch.obs, self(batch).act).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.sync_weight()
        return {
            "loss/actor": actor_loss.item(),
            "loss/critic": critic_loss.item(),
        }

    def exploration_noise(self, act: Union[np.ndarray, Batch],
                          batch: Batch) -> Union[np.ndarray, Batch]:
        if self._noise is None:
            return act
        if isinstance(act, np.ndarray):
            return act + self._noise(act.shape)
        warnings.warn("Cannot add exploration noise to non-numpy_array action.")
        return act


class AsyncACDDPGPolicy(DDPGPolicy):
	@staticmethod
	def _mse_optimizer(
		batch: Batch, critic: torch.nn.Module, optimizer: torch.optim.Optimizer
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""A simple wrapper script for updating critic network."""
		weight = getattr(batch, "weight", 1.0)
		# current_q = critic(batch.obs, batch.act).flatten()
		current_q = critic(batch.info["obs_cur"], batch.act).flatten()
		target_q = batch.returns.flatten()
		td = current_q - target_q
		# critic_loss = F.mse_loss(current_q1, target_q)
		critic_loss = (td.pow(2) * weight).mean()
		optimizer.zero_grad()
		critic_loss.backward()
		optimizer.step()
		return td, critic_loss

	def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
		# critic
		td, critic_loss = self._mse_optimizer(batch, self.critic, self.critic_optim)
		batch.weight = td  # prio-buffer
		# actor
		# actor_loss = -self.critic(batch.obs, self(batch).act).mean()
		actor_loss = -self.critic(batch.info["obs_cur"], self(batch).act).mean()
		self.actor_optim.zero_grad()
		actor_loss.backward()
		self.actor_optim.step()
		self.sync_weight()
		return {
			"loss/actor": actor_loss.item(),
			"loss/critic": critic_loss.item(),
		}

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
@hydra.main(version_base=None, config_path=str(root / "configs"), config_name="ddpg.yaml")
def main(cfg):
	def make_env(env_cfg):
		env = gym.make(env_cfg.name)
		env = DelayedRoboticEnv(env, env_cfg.delay)
		return env
	# init
	utils.print_config_tree(cfg, resolve=True)
	wandb.init(project=cfg.task_name, tags=cfg.tags, config=utils.config_format(cfg))
	cfg = hydra.utils.instantiate(cfg)
	# env & not & policy
	train_envs = tianshou.env.DummyVectorEnv([partial(make_env, cfg.env) for _ in range(cfg.env.train_num)])
	test_envs = tianshou.env.DummyVectorEnv([partial(make_env, cfg.env) for _ in range(cfg.env.test_num)])
	env = make_env(cfg.env)
	net = cfg.net(env.observation_space.shape)
	actor = cfg.actor(net, env.action_space.shape).to(cfg.device)
	actor_optim = cfg.actor_optim(actor.parameters())
	net_c1 = cfg.net_c1(env.observation_space.shape, env.action_space.shape)
	critic1 = cfg.critic1(net_c1).to(cfg.device)
	critic1_optim = cfg.critic1_optim(critic1.parameters())
	net_c2 = cfg.net_c2(env.observation_space.shape, env.action_space.shape)
	critic2 = cfg.critic2(net_c2).to(cfg.device)
	critic2_optim = cfg.critic2_optim(critic2.parameters())
	policy = cfg.policy(
		actor, actor_optim,
		critic1, critic1_optim,
		# critic2, critic2_optim,
		action_space=env.action_space,
	)
	# collector
	train_collector = cfg.train_collector(policy, train_envs)
	test_collector = cfg.test_collector(policy, test_envs)
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
