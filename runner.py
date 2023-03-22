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
from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union
import numpy as np
import torch
from torch import nn
from tianshou.utils.net.common import MLP
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

class SACPolicy(DDPGPolicy):
	"""Implementation of Soft Actor-Critic. arXiv:1812.05905.

	:param torch.nn.Module actor: the actor network following the rules in
		:class:`~tianshou.policy.BasePolicy`. (s -> logits)
	:param torch.optim.Optimizer actor_optim: the optimizer for actor network.
	:param torch.nn.Module critic1: the first critic network. (s, a -> Q(s, a))
	:param torch.optim.Optimizer critic1_optim: the optimizer for the first
		critic network.
	:param torch.nn.Module critic2: the second critic network. (s, a -> Q(s, a))
	:param torch.optim.Optimizer critic2_optim: the optimizer for the second
		critic network.
	:param float tau: param for soft update of the target network. Default to 0.005.
	:param float gamma: discount factor, in [0, 1]. Default to 0.99.
	:param (float, torch.Tensor, torch.optim.Optimizer) or float alpha: entropy
		regularization coefficient. Default to 0.2.
		If a tuple (target_entropy, log_alpha, alpha_optim) is provided, then
		alpha is automatically tuned.
	:param bool reward_normalization: normalize the reward to Normal(0, 1).
		Default to False.
	:param BaseNoise exploration_noise: add a noise to action for exploration.
		Default to None. This is useful when solving hard-exploration problem.
	:param bool deterministic_eval: whether to use deterministic action (mean
		of Gaussian policy) instead of stochastic action sampled by the policy.
		Default to True.
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
		actor: torch.nn.Module,
		actor_optim: torch.optim.Optimizer,
		critic1: torch.nn.Module,
		critic1_optim: torch.optim.Optimizer,
		critic2: torch.nn.Module,
		critic2_optim: torch.optim.Optimizer,
		tau: float = 0.005, # TODO use hyperparameter in the paper
		gamma: float = 0.99,
		alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
		reward_normalization: bool = False,
		estimation_step: int = 1,
		exploration_noise: Optional[BaseNoise] = None,
		deterministic_eval: bool = True,
		**kwargs: Any,
	) -> None:
		super().__init__(
			None, None, None, None, tau, gamma, exploration_noise,
			reward_normalization, estimation_step, **kwargs
		)
		self.actor, self.actor_optim = actor, actor_optim
		self.critic1, self.critic1_old = critic1, deepcopy(critic1)
		self.critic1_old.eval()
		self.critic1_optim = critic1_optim
		self.critic2, self.critic2_old = critic2, deepcopy(critic2)
		self.critic2_old.eval()
		self.critic2_optim = critic2_optim

		self._is_auto_alpha = False
		self._alpha: Union[float, torch.Tensor]
		if isinstance(alpha, tuple):
			self._is_auto_alpha = True
			self._target_entropy, self._log_alpha, self._alpha_optim = alpha
			if type(self._target_entropy) == str and self._target_entropy == "neg_act_num":
				if hasattr(self.actor, "mu"): # get act dim TODO improvement here
					if hasattr(self.actor.mu, "output_dim"):
						act_num = self.actor.mu.output_dim
					elif hasattr(self.actor.mu, "out_features"):
						act_num = self.actor.mu.out_features
					else:
						raise ValueError("Can not get actor output dim.")
				elif hasattr(self.actor, "output_dim"):
					act_num = self.actor.output_dim
				else:
					raise ValueError("Invalid actor type.")
				self._target_entropy = - act_num
			elif type(self._target_entropy) == float:
				pass
			else: 
				raise ValueError("Invalid target entropy type.")
			assert alpha[1].shape == torch.Size([1]) and alpha[1].requires_grad
			self._alpha_optim = self._alpha_optim([alpha[1]])
			self._alpha = self._log_alpha.detach().exp()
		else:
			self._alpha = alpha

		self._deterministic_eval = deterministic_eval
		self.__eps = np.finfo(np.float32).eps.item()

	def train(self, mode: bool = True) -> "SACPolicy":
		self.training = mode
		self.actor.train(mode)
		self.critic1.train(mode)
		self.critic2.train(mode)
		return self

	def sync_weight(self) -> None:
		self.soft_update(self.critic1_old, self.critic1, self.tau)
		self.soft_update(self.critic2_old, self.critic2, self.tau)

	def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
		batch = buffer[indices]  # batch.obs: s_{t+n}
		obs_next_result = self(batch, input="obs_next")
		act_ = obs_next_result.act
		target_q = torch.min(
			self.critic1_old(batch.obs_next, act_),
			self.critic2_old(batch.obs_next, act_),
		) - self._alpha * obs_next_result.log_prob
		return target_q

	def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
		# critic 1&2
		td1, critic1_loss = self._mse_optimizer(
			batch, self.critic1, self.critic1_optim
		)
		td2, critic2_loss = self._mse_optimizer(
			batch, self.critic2, self.critic2_optim
		)
		batch.weight = (td1 + td2) / 2.0  # prio-buffer

		# actor
		obs_result = self(batch)
		act = obs_result.act
		current_q1a = self.critic1(batch.obs, act).flatten()
		current_q2a = self.critic2(batch.obs, act).flatten()
		actor_loss = (
			self._alpha * obs_result.log_prob.flatten() -
			torch.min(current_q1a, current_q2a)
		).mean()
		self.actor_optim.zero_grad()
		actor_loss.backward()
		self.actor_optim.step()

		if self._is_auto_alpha: # TODO auto alpha
			log_prob = obs_result.log_prob.detach() + self._target_entropy
			# please take a look at issue #258 if you'd like to change this line
			alpha_loss = -(self._log_alpha * log_prob).mean()
			self._alpha_optim.zero_grad()
			alpha_loss.backward()
			self._alpha_optim.step()
			self._alpha = self._log_alpha.detach().exp()

		self.sync_weight()

		result = {
			"loss/actor": actor_loss.item(),
			"loss/critic1": critic1_loss.item(),
			"loss/critic2": critic2_loss.item(),
		}
		if self._is_auto_alpha:
			result["loss/alpha"] = alpha_loss.item()
			result["alpha"] = self._alpha.item()  # type: ignore

		return result

class CustomSACPolicy_(SACPolicy):

	def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
		batch = buffer[indices]  # batch.obs: s_{t+n}
		next_batch = buffer[buffer.next(indices)]  # next_batch.obs: s_{t+n+1}
		obs_next_result = self(batch, input="obs_next")
		act_ = obs_next_result.act
		target_q = torch.min(
			self.critic1_old(next_batch.obs, act_),
			self.critic2_old(next_batch.obs, act_),
		) - self._alpha * obs_next_result.log_prob
		return target_q

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
		# critic 1&2
		batch_for_critic = Batch(batch, copy=True)
		if "current_obs" in batch.info.keys(): # use non-delayed obs for critic if available
			batch_for_critic.obs = batch_for_critic.info.current_obs
		td1, critic1_loss = self._mse_optimizer(
			batch_for_critic, self.critic1, self.critic1_optim
		)
		td2, critic2_loss = self._mse_optimizer(
			batch_for_critic, self.critic2, self.critic2_optim
		)
		batch_for_critic.weight = (td1 + td2) / 2.0  # prio-buffer

		# actor
		obs_result = self(batch_for_critic)
		act = obs_result.act
		current_q1a = self.critic1(batch_for_critic.obs, act).flatten()
		current_q2a = self.critic2(batch_for_critic.obs, act).flatten()
		actor_loss = (
			self._alpha * obs_result.log_prob.flatten() -
			torch.min(current_q1a, current_q2a)
		).mean()
		self.actor_optim.zero_grad()
		actor_loss.backward()
		self.actor_optim.step()

		if self._is_auto_alpha:
			log_prob = obs_result.log_prob.detach() + self._target_entropy
			# please take a look at issue #258 if you'd like to change this line
			alpha_loss = -(self._log_alpha * log_prob).mean()
			self._alpha_optim.zero_grad()
			alpha_loss.backward()
			self._alpha_optim.step()
			self._alpha = self._log_alpha.detach().exp()

		self.sync_weight()

		result = {
			"loss/actor": actor_loss.item(),
			"loss/critic1": critic1_loss.item(),
			"loss/critic2": critic2_loss.item(),
		}
		if self._is_auto_alpha:
			result["loss/alpha"] = alpha_loss.item()
			result["alpha"] = self._alpha.item()  # type: ignore

		return result

class AsyncACDDPGPolicy(DDPGPolicy):
	@staticmethod
	def _mse_optimizer(
		batch: Batch, critic: torch.nn.Module, optimizer: torch.optim.Optimizer
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""A simple wrapper script for updating critic network."""
		weight = getattr(batch, "weight", 1.0)
		# current_q = critic(batch.obs, batch.act).flatten()
		current_q = critic(batch.info["obs_nodelay"], batch.act).flatten()
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
		actor_loss = -self.critic(batch.info["obs_nodelay"], self(batch).act).mean()
		self.actor_optim.zero_grad()
		actor_loss.backward()
		self.actor_optim.step()
		self.sync_weight()
		return {
			"loss/actor": actor_loss.item(),
			"loss/critic": critic_loss.item(),
		}

	def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
		batch = buffer[indices]  # batch.obs: s_{t+n}
		next_batch = buffer[indices + 1]  # next_batch.obs: s_{t+n+1}
		next_batch.obs_cur = next_batch.info["obs_cur"]
		# obs_next_result = self(batch, input="obs_cur")
		obs_next_result = self(next_batch, input="obs_cur")
		act_ = obs_next_result.act
		target_q = torch.min(
			# self.critic1_old(batch.obs_next, act_),
			# self.critic2_old(batch.obs_next, act_),
			self.critic1_old(next_batch.obs_cur, act_),
			self.critic2_old(next_batch.obs_cur, act_),
		) - self._alpha * obs_next_result.log_prob
		return target_q

class CustomSACPolicy(SACPolicy):
	def __init__(
		self,
		actor: torch.nn.Module,
		actor_optim: torch.optim.Optimizer,
		critic1: torch.nn.Module,
		critic1_optim: torch.optim.Optimizer,
		critic2: torch.nn.Module,
		critic2_optim: torch.optim.Optimizer,
		tau: float = 0.005, # TODO use hyperparameter in the paper
		gamma: float = 0.99,
		alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
		reward_normalization: bool = False,
		estimation_step: int = 1,
		exploration_noise: Optional[BaseNoise] = None,
		deterministic_eval: bool = True,
		**kwargs: Any,
	) -> None:
		self.critic_use_oracle_obs = kwargs.pop("critic_use_oracle_obs")
		self.actor_rnn = kwargs.pop("actor_rnn")
		self.actor_input_act = kwargs.pop("actor_input_act")
		return super().__init__(
			actor,	
			actor_optim,
			critic1,
			critic1_optim,
			critic2,
			critic2_optim,
			tau=tau,
			gamma=gamma,
			alpha=alpha,
			reward_normalization=reward_normalization,
			estimation_step=estimation_step,
			exploration_noise=exploration_noise,
			deterministic_eval=deterministic_eval,
			**kwargs
		)
	
	def _mse_optimizer(self,
		batch: Batch, critic: torch.nn.Module, optimizer: torch.optim.Optimizer
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""A simple wrapper script for updating critic network."""
		weight = getattr(batch, "weight", 1.0)
		if not self.actor_rnn: # ! TODO check order, whether -1 is correct or 0
			if self.critic_use_oracle_obs:
				current_q = critic(batch.info["obs_nodelay"], batch.act).flatten()
			else:
				current_q = critic(batch.obs, batch.act).flatten()
		else:
			if self.critic_use_oracle_obs:
				current_q = critic(batch.info["obs_nodelay"][:,-1], batch.act).flatten()
			else:
				current_q = critic(batch.obs[:,-1], batch.act).flatten()
		target_q = batch.returns.flatten()
		td = current_q - target_q
		critic_loss = (td.pow(2) * weight).mean()
		optimizer.zero_grad()
		critic_loss.backward()
		optimizer.step()
		return td, critic_loss
	
	def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
		# critic 1&2
		td1, critic1_loss = self._mse_optimizer(
			batch, self.critic1, self.critic1_optim
		)
		td2, critic2_loss = self._mse_optimizer(
			batch, self.critic2, self.critic2_optim
		)
		batch.weight = (td1 + td2) / 2.0  # prio-buffer

		# actor
		if not self.actor_rnn:
			if not self.actor_input_act:
				obs_result = self(batch)
			else:
				batch.obs_act = np.concatenate([batch.obs, batch.info["act_prev"]], axis=1)
				obs_result = self(batch, input="obs_act")
		else:
			# ! TODO check order, whether -1 is correct or 0
			# ! TODO check step of obs, ...
			if not self.actor_input_act: 
				obs_result = self(batch)
			else:
				batch.obs_act = np.concatenate([batch.obs, batch.info["stacked_act_prev"]], axis=2) # B,L,... use (S_t,a_t-1) # ! TODO check order, whether -1 is correct or 0
				obs_result = self(batch, input="obs_act") # actor use delayed obs
		act = obs_result.act
		if not self.actor_rnn:
			if self.critic_use_oracle_obs:
				current_q1a = self.critic1(batch.info["obs_nodelay"], act).flatten()
				current_q2a = self.critic2(batch.info["obs_nodelay"], act).flatten()
			else:
				current_q1a = self.critic1(batch.obs, act).flatten()
				current_q1a = self.critic1(batch.obs, act).flatten()
		else:
			if self.critic_use_oracle_obs:
				current_q1a = self.critic1(batch.info["obs_nodelay"][:,-1], act).flatten()
				current_q2a = self.critic2(batch.info["obs_nodelay"][:,-1], act).flatten()
			else:
				current_q1a = self.critic1(batch.obs[:,-1], act).flatten()
				current_q2a = self.critic2(batch.obs[:,-1], act).flatten()

		actor_loss = (
			self._alpha * obs_result.log_prob.flatten() -
			torch.min(current_q1a, current_q2a)
		).mean()
		self.actor_optim.zero_grad()
		actor_loss.backward()
		self.actor_optim.step()

		if self._is_auto_alpha:
			log_prob = obs_result.log_prob.detach() + self._target_entropy
			# please take a look at issue #258 if you'd like to change this line
			alpha_loss = -(self._log_alpha * log_prob).mean()
			self._alpha_optim.zero_grad()
			alpha_loss.backward()
			self._alpha_optim.step()
			self._alpha = self._log_alpha.detach().exp()

		self.sync_weight()

		result = {
			"loss/actor": actor_loss.item(),
			"loss/critic1": critic1_loss.item(),
			"loss/critic2": critic2_loss.item(),
		}
		if self._is_auto_alpha:
			result["target_entropy"] = self._target_entropy
			result["loss/alpha"] = alpha_loss.item()
			result["_log_alpha"] = self._log_alpha.item()
			result["alpha"] = self._alpha.item()  # type: ignore

		return result
	
	def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
		batch = buffer[indices]  # batch.obs: s_{t+n}
		if not self.actor_rnn:
			if not self.actor_input_act: 
				obs_next_result = self(batch, input="obs_next") # ! CHECK
			else:
				obs_next_result = self(batch, input="obs_next") # actor use delayed obs
		else:
			if not self.actor_input_act:
				obs_next_result = self(batch)
				obs_next_result = self(batch, input="obs_next") # actor use delayed obs
			else:
				act = buffer.get(indices, "act")
				batch.obs_act = np.concatenate([batch.obs_next,act], axis=2) # B,L,... use (S_t+1,a_t)# ! TODO check order, whether -1 is correct or 0
				obs_next_result = self(batch, input="obs_act") # actor use delayed obs

		act_ = obs_next_result.act
		if not self.actor_rnn:
			target_q = torch.min(
				self.critic1_old(batch.info["obs_next_nodelay"], act_) \
				if self.critic_use_oracle_obs else \
				self.critic1_old(batch.obs_next, act_),
				self.critic2_old(batch.info["obs_next_nodelay"], act_) \
				if self.critic_use_oracle_obs else \
				self.critic2_old(batch.obs_next, act_)
			) - self._alpha * obs_next_result.log_prob
		else:
			target_q = torch.min(
				self.critic1_old(batch.info["obs_next_nodelay"][:,-1], act_) \
				if self.critic_use_oracle_obs else \
				self.critic1_old(batch.obs_next[:,-1], act_),
				self.critic2_old(batch.info["obs_next_nodelay"][:,-1], act_) \
				if self.critic_use_oracle_obs else \
				self.critic2_old(batch.obs_next[:,-1], act_)
			) - self._alpha * obs_next_result.log_prob

		return target_q

	def process_fn(
		self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
	) -> Batch:
		super().process_fn(batch, buffer, indices)
		if self.critic_use_oracle_obs:
			prev_batch = buffer[buffer.prev(indices)]
			batch.info["obs_nodelay"] = prev_batch.info["obs_next_nodelay"]
		if not self.actor_rnn:
			if self.actor_input_act:
				batch.info["act_prev"] = buffer[buffer.prev(indices)].act
		else:
			if self.actor_input_act:
				batch.info["stacked_act"] = buffer.get(indices, "act")
				batch.info["stacked_act_prev"] = buffer.get(buffer.prev(indices), "act")
		return batch

	def forward(  # type: ignore
		self,
		batch: Batch,
		state: Optional[Union[dict, Batch, np.ndarray]] = None,
		input: str = "obs",
		**kwargs: Any,
	) -> Batch:
		obs = batch[input]
		if self.actor_rnn:
			if not self.actor_input_act:
				obs = obs
			else:
				if len(obs.shape) == 2: # (B, s_dim) online 
					if len(batch.act.shape) == 0: # first step, when act is not available
						obs_act = np.zeros([obs.shape[0], 1, self.actor.nn.input_size])
					else:
						obs_act = np.concatenate([obs, batch.act], axis=1)
				else: # (B, L, s_dim) offline self.learn
					obs_act = obs
		else:
			if not self.actor_input_act:
				obs = obs
			else:
				if len(obs.shape) == 2: # (B, s_dim) online 
					if len(batch.act.shape) == 0: # first step, when act is not available
						obs_act = np.zeros([obs.shape[0], self.actor.nn.input_size])
					else:
						obs_act = np.concatenate([obs, batch.act], axis=1)
				else: # (B, L, s_dim) offline self.learn
					obs_act = obs
		if self.actor_input_act:
			logits, hidden = self.actor(obs_act, state=state, info=batch.info)
		else:
			logits, hidden = self.actor(obs, state=state, info=batch.info)
		assert isinstance(logits, tuple)
		dist = Independent(Normal(*logits), 1)
		if self._deterministic_eval and not self.training:
			act = logits[0]
		else:
			act = dist.rsample()
		log_prob = dist.log_prob(act).unsqueeze(-1)
		# apply correction for Tanh squashing when computing logprob from Gaussian
		# You can check out the original SAC paper (arXiv 1801.01290): Eq 21.
		# in appendix C to get some understanding of this equation.
		squashed_action = torch.tanh(act)
		log_prob = log_prob - torch.log((1 - squashed_action.pow(2)) +
										np.finfo(np.float32).eps.item()).sum(-1, keepdim=True)
		return Batch(
			logits=logits,
			act=squashed_action,
			state=hidden,
			dist=dist,
			log_prob=log_prob
		)

class DummyNet(nn.Module):
	"""Return input as output."""
	def forward(self, x):
		return x

class CustomRecurrentActorProb(nn.Module):
	"""Recurrent version of ActorProb.

	edit log:
		1. add rnn_hidden_layer_size and mlp_hidden_sizes for consecutive processing
			original ActorProb only has one hidden layer after lstm. In the new version, 
			we can customize both the size of RNN hidden layer (with rnn_hidden_layer_size)
			and the size of mlp hidden layer (with mlp_hidden_sizes)
			RNN: rnn_hidden_layer_size * rnn_layer_num
			MLP: mlp_hidden_sizes[0] * mlp_hidden_sizes[1] * ...

	"""
	SIGMA_MIN = -20
	SIGMA_MAX = 2
	
	def __init__(
		self,
		state_shape: Sequence[int],
		action_shape: Sequence[int],
		rnn_layer_num: int = 0,
		rnn_hidden_layer_size: int = 128,
		mlp_hidden_sizes: Sequence[int] = (),
		mlp_softmax: bool = False,
		max_action: float = 1.0,
		device: Union[str, int, torch.device] = "cpu",
		unbounded: bool = False,
		conditioned_sigma: bool = False,
		concat: bool = False,
	) -> None:
		# ! TODO add mlp_softmax
		super().__init__()
		self.device = device
		input_dim = int(np.prod(state_shape))
		action_dim = int(np.prod(action_shape))
		if concat:
			input_dim += action_dim
		if rnn_layer_num:
			self.nn = nn.LSTM(
				input_size=input_dim,
				hidden_size=rnn_hidden_layer_size,
				num_layers=rnn_layer_num,
				batch_first=True,
			)
		else:
			self.nn = DummyNet()
		output_dim = int(np.prod(action_shape))
		# self.mu = nn.Linear(hidden_layer_size, output_dim)
		self.mu = MLP(
			rnn_hidden_layer_size if rnn_layer_num else input_dim,  # type: ignore
			output_dim,
			mlp_hidden_sizes,
			device=self.device
		)
		self._c_sigma = conditioned_sigma
		if conditioned_sigma:
			# self.sigma = nn.Linear(hidden_layer_size, output_dim)
			self.sigma = MLP(
				rnn_hidden_layer_size,  # type: ignore
				output_dim,
				mlp_hidden_sizes,
				device=self.device
			)
		else:
			self.sigma_param = nn.Parameter(torch.zeros(output_dim, 1))
		self._max = max_action
		self._unbounded = unbounded

	def forward(
		self,
		obs: Union[np.ndarray, torch.Tensor],
		state: Optional[Dict[str, torch.Tensor]] = None,
		info: Dict[str, Any] = {},
	) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
		"""Almost the same as :class:`~tianshou.utils.net.common.Recurrent`."""
		obs = torch.as_tensor(
			obs,
			device=self.device,
			dtype=torch.float32,
		)
		# obs [bsz, len, dim] (training) or [bsz, dim] (evaluation)
		# In short, the tensor's shape in training phase is longer than which
		# in evaluation phase.
		if len(obs.shape) == 2:
			obs = obs.unsqueeze(-2)
		self.nn.flatten_parameters()
		if state is None:
			obs, (hidden, cell) = self.nn(obs)
		else:
			# we store the stack data in [bsz, len, ...] format
			# but pytorch rnn needs [len, bsz, ...]
			obs, (hidden, cell) = self.nn(
				obs, (
					state["hidden"].transpose(0, 1).contiguous(),
					state["cell"].transpose(0, 1).contiguous()
				)
			)
		logits = obs[:, -1]
		mu = self.mu(logits)
		if not self._unbounded:
			mu = self._max * torch.tanh(mu)
		if self._c_sigma:
			sigma = torch.clamp(self.sigma(logits), min=self.SIGMA_MIN, max=self.SIGMA_MAX).exp()
		else:
			shape = [1] * len(mu.shape)
			shape[1] = -1
			sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
		# please ensure the first dim is batch size: [bsz, len, ...]
		return (mu, sigma), {
			"hidden": hidden.transpose(0, 1).detach(),
			"cell": cell.transpose(0, 1).detach()
		}

# Runner

class DefaultRLRunner:
	""" Runner for RL algorithms
	Work flow: 
		entry.py
			initialize cfg.runner, pass cfg to runner
		runner.py(XXXRunner)
			initialize envs, policy, buffer, etc.
			call DefaultRLRunner.__init__(cfg) for common initialization
			call XXXRunner.__init__(cfg) for specific initialization
	"""
	def start(self, cfg):
		print("RLRunner start ...")
		self.cfg = cfg
		self.env = cfg.env
		# init
		wandb.init(project=cfg.task_name, tags=cfg.tags, config=utils.config_format(cfg),dir=cfg.output_dir)
		utils.seed_everything(cfg.seed) # TODO add env seed
		self.train_envs = tianshou.env.DummyVectorEnv([partial(utils.make_env, cfg.env) for _ in range(cfg.env.train_num)])
		self.test_envs = tianshou.env.DummyVectorEnv([partial(utils.make_env, cfg.env) for _ in range(cfg.env.test_num)])
		self.env = utils.make_env(cfg.env) # to get obs_space and act_space
		print("RLRunner end!")

class SACRunner(DefaultRLRunner):
	def start(self, cfg):
		print("SACRunner init start ...")
		super().start(cfg)
		env = self.env
		train_envs = self.train_envs
		test_envs = self.test_envs
		actor = cfg.actor(state_shape=env.observation_space.shape, action_shape=env.action_space.shape, max_action=env.action_space.high[0]).to(cfg.device)
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
		train_collector.collect(n_step=cfg.start_timesteps, random=True)
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
		print("SACRunner init end!")
