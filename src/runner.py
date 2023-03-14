import pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
from typing import Callable, Any, Dict, List, Optional, Type, Union, Tuple
from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as
import numpy as np
import torch
import hydra
from pprint import pprint
import torch
import wandb
import numpy as np
import gymnasium as gym
from tianshou.policy.base import _nstep_return
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


# policy
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
		self.actor_use_oracle_obs = kwargs.pop("actor_use_oracle_obs")
		self.global_cfg = kwargs.pop("global_cfg")
		if self.global_cfg.historical_act:
			if self.global_cfg.historical_act.type == "cat_mlp":
				# assert actor.__class__ is not CustomRecurrentActorProb, "cat_mlp should not be used with recurrent actor"
				assert actor.rnn_layer_num == 0, "cat_mlp should not be used with recurrent actor"
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
		if self.global_cfg.historical_act:
			if self.global_cfg.historical_act.type == "cat_mlp":
				if self.critic_use_oracle_obs:
					current_q = critic(batch.info["obs_nodelay"], batch.act).flatten()
				else:
					current_q = critic(batch.obs, batch.act).flatten()
			elif self.global_cfg.historical_act.type == "stack_rnn":
				if self.critic_use_oracle_obs:
					bsz_len_shape = batch.act.shape[:2]
					flatten_num = np.prod(bsz_len_shape)
					current_q = critic(
						batch.info["obs_nodelay"].reshape(flatten_num, -1), 
						batch.act.reshape(flatten_num, -1)
					).flatten()
				else:
					current_q = critic(
						batch.obs.reshape(flatten_num, -1),
						batch.act.reshape(flatten_num, -1)
					).flatten() # ! TODO can be merged with the above
			else:
				raise NotImplementedError
		else:
			if self.critic_use_oracle_obs:
				current_q = critic(batch.info["obs_nodelay"], batch.act).flatten()
			else:
				current_q = critic(batch.obs, batch.act).flatten()
		target_q = batch.returns.flatten()
		td = current_q - target_q
		if self.global_cfg.historical_act and self.global_cfg.historical_act.type == "stack_rnn":
			critic_loss = (td.pow(2) * weight)
			critic_loss = critic_loss.reshape(*bsz_len_shape, -1)
			critic_loss = critic_loss[:, self.global_cfg.historical_act.burnin_num:]
			critic_loss = critic_loss.mean()
		else:
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

		# actor ###ACTOR_FORWARD
		# if not self.actor_rnn:
		# 	if not self.actor_input_act:
		# 		if not self.historical_act:
		# 			obs_result = self(batch)
		# 		else:
		# 			batch.obs_cat_act = np.concatenate([batch.obs, batch.info["historical_act"]], axis=1)
		# 			obs_result = self(batch, input="obs_cat_act")
		# 	else:
		# 		batch.obs_act = np.concatenate([batch.obs, batch.info["act_prev"]], axis=1)
		# 		obs_result = self(batch, input="obs_act")
		# else:
		# 	# ! TODO check order, whether -1 is correct or 0
		# 	# ! TODO check step of obs, ...
		# 	if not self.actor_input_act: 
		# 		obs_result = self(batch)
		# 	else:
		# 		batch.obs_act = np.concatenate([batch.obs, batch.info["stacked_act_prev"]], axis=2) # B,L,... use (S_t,a_t-1) # ! TODO check order, whether -1 is correct or 0
		# 		obs_result = self(batch, input="obs_act") # actor use delayed obs
		# act = obs_result.act
		# get act
		if self.global_cfg.historical_act:
			if self.global_cfg.historical_act.type == "cat_mlp":
				batch.obs_cat_act = np.concatenate([self.get_actor_obs(batch,"cur"), batch.info["historical_act"]], axis=1) \
					if self.global_cfg.historical_act.num > 0 else self.get_actor_obs(batch,"cur") # TODO put a place holder here
				obs_result = self(batch, input="obs_cat_act")
			elif self.global_cfg.historical_act.type == "stack_rnn":
				assert batch.is_preprocessed == True, "batch.is_preprocessed == True"
				assert hasattr(batch, "obs_stack_act"), "hasattr(batch, 'obs_stack_act')"
				obs_result = self(batch, input="obs_stack_act")
			else:
				raise NotImplementedError
		else:
			obs_result = self(batch)
		act = obs_result.act

		# cal q
		if self.global_cfg.historical_act:
			if self.global_cfg.historical_act.type == "cat_mlp":
				if self.critic_use_oracle_obs:
					current_q1a = self.critic1(batch.info["obs_nodelay"], act).flatten()
					current_q2a = self.critic2(batch.info["obs_nodelay"], act).flatten()
				else:
					current_q1a = self.critic1(batch.obs, act).flatten()
					current_q2a = self.critic2(batch.obs, act).flatten()
			elif self.global_cfg.historical_act.type == "stack_rnn":
				if self.critic_use_oracle_obs: # ! TODO can be merged with above 
					bsz_len_shape = batch.act.shape[:2]
					flatten_num = np.prod(bsz_len_shape)
					current_q1a = self.critic1(
						batch.info["obs_nodelay"].reshape(flatten_num, -1),
						act.reshape(flatten_num, -1)
					).flatten()
					current_q2a = self.critic2(
						batch.info["obs_nodelay"].reshape(flatten_num, -1),
						act.reshape(flatten_num, -1)
					).flatten()
				else:
					current_q1a = self.critic1(batch.obs, act).flatten()
					current_q2a = self.critic2(batch.obs, act).flatten()
			else:
				raise NotImplementedError
		else:
			if self.critic_use_oracle_obs:
				current_q1a = self.critic1(batch.info["obs_nodelay"], act).flatten()
				current_q2a = self.critic2(batch.info["obs_nodelay"], act).flatten()
			else:
				current_q1a = self.critic1(batch.obs, act).flatten()
				current_q2a = self.critic2(batch.obs, act).flatten()

		# if not self.actor_rnn:
		# 	if self.critic_use_oracle_obs:
		# 		current_q1a = self.critic1(batch.info["obs_nodelay"], act).flatten()
		# 		current_q2a = self.critic2(batch.info["obs_nodelay"], act).flatten()
		# 	else:
		# 		current_q1a = self.critic1(batch.obs, act).flatten()
		# 		current_q2a = self.critic2(batch.obs, act).flatten()
		# else:
		# 	if self.critic_use_oracle_obs:
		# 		current_q1a = self.critic1(batch.info["obs_nodelay"][:,-1], act).flatten()
		# 		current_q2a = self.critic2(batch.info["obs_nodelay"][:,-1], act).flatten()
		# 	else:
		# 		current_q1a = self.critic1(batch.obs[:,-1], act).flatten()
		# 		current_q2a = self.critic2(batch.obs[:,-1], act).flatten()
		if self.global_cfg.historical_act and self.global_cfg.historical_act.type == "stack_rnn":
			actor_loss = self._alpha * obs_result.log_prob.flatten() - \
				torch.min(current_q1a, current_q2a)
			actor_loss = actor_loss.reshape(*bsz_len_shape, -1)
			actor_loss = actor_loss[:, self.global_cfg.historical_act.burnin_num:]
			actor_loss = actor_loss.mean()
		else:
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
		###ACTOR_FORWARD
		# if not self.actor_rnn:
		# 	if not self.actor_input_act: 
		# 		if self.historical_act:
		# 			historical_act = buffer[buffer.next(indices)].info["historical_act"]
		# 			batch.obs_cat_act = np.concatenate([batch.obs_next, historical_act], axis=1)
		# 			obs_next_result = self(batch, input="obs_cat_act")
		# 		else:
		# 			obs_next_result = self(batch, input="obs_next") # actor use delayed obs
		# 	else:
		# 		obs_next_result = self(batch, input="obs_next")
		# else:
		# 	if not self.actor_input_act:
		# 		obs_next_result = self(batch)
		# 		obs_next_result = self(batch, input="obs_next") # actor use delayed obs
		# 	else:
		# 		act = buffer.get(indices, "act")
		# 		batch.obs_act = np.concatenate([batch.obs_next,act], axis=2) # B,L,... use (S_t+1,a_t)# ! TODO check order, whether -1 is correct or 0
		# 		obs_next_result = self(batch, input="obs_act") # actor use delayed obs
		if self.global_cfg.historical_act:
			if self.global_cfg.historical_act.type == "cat_mlp":
				next_historical_act = buffer[buffer.next(indices)].info["historical_act"]
				batch.obs_cat_act = np.concatenate([self.get_actor_obs(batch,"next"), next_historical_act], axis=-1) \
					if self.global_cfg.historical_act.num > 0 else self.get_actor_obs(batch,"next")
				# ! TODO check buffer[buffer.next(indices)].obs == buffer[indices].obs_next
				batch.is_preprocessed = True
				obs_next_result = self(batch, input="obs_cat_act")
			elif self.global_cfg.historical_act.type == "stack_rnn":
				historical_obs = buffer.get(indices, "obs_next", stack_num=self.global_cfg.historical_act.num)
				if len(historical_obs.shape) == 2: historical_obs = historical_obs[:,None,:]
				# historical_act = buffer.get(buffer.prev(indices), "act", stack_num=self.global_cfg.historical_act.num)
				# buffer[buffer.next(indices)].obs == buffer[indices].obs_next
				# buffer.get(indices, "act", stack_num=self.global_cfg.historical_act.num)
				# buffer.get(buffer.next(indices), "info", stack_num=self.global_cfg.historical_act.num)["prev_act"]
				# buffer[indices].act == buffer[buffer.next(indices)].info["prev_act"]
				# buffer[indices].info["prev_act"] == buffer[buffer.prev(indices)].act
				# buffer.get(indices, "act", stack_num=self.global_cfg.historical_act.num) == buffer.get(buffer.next(indices), "info", stack_num=self.global_cfg.historical_act.num)["prev_act"]
				historical_act = buffer.get(indices, "act", stack_num=self.global_cfg.historical_act.num)
				if len(historical_act.shape) == 2: historical_act = historical_act[:,None,:]
				# ! TODO check buffer[buffer.next(indices)].obs == buffer[indices].obs_next
				batch.obs_stack_act = np.concatenate([historical_obs, historical_act], axis=-1)
				batch.is_preprocessed = True
				obs_next_result = self(batch, input="obs_stack_act")
			else:
				raise NotImplementedError
		else: # normal mode
			obs_next_result = self(batch, input="obs_next")

		act_ = obs_next_result.act
		if self.global_cfg.historical_act:
			if self.global_cfg.historical_act.type == "cat_mlp":
				target_q = torch.min(
					self.critic1_old(batch.info["obs_next_nodelay"], act_) \
					if self.critic_use_oracle_obs else \
					self.critic1_old(batch.obs_next, act_),
					self.critic2_old(batch.info["obs_next_nodelay"], act_) \
					if self.critic_use_oracle_obs else \
					self.critic2_old(batch.obs_next, act_)
				) - self._alpha * obs_next_result.log_prob
			elif self.global_cfg.historical_act.type == "stack_rnn":
				info_stacked = buffer.get(indices, "info", stack_num=self.global_cfg.historical_act.num)
				obs_next_stacked = buffer.get(indices, "obs_next", stack_num=self.global_cfg.historical_act.num)
				# flatten and reshape bak s=(256,80,19) -> (256*80,19), a=(256,80,4) -> (256*80,4)
				bsz_len_shape = act_.shape[:2]
				flatten_num = np.prod(bsz_len_shape)
				target_q = torch.min( # ! TODO can be merged with above # (256, 80, 19)
					self.critic1_old(info_stacked["obs_next_nodelay"].reshape(flatten_num,-1), act_.reshape(flatten_num,-1)) \
					if self.critic_use_oracle_obs else \
					self.critic1_old(obs_next_stacked.reshape(flatten_num,-1), act_.reshape(flatten_num,-1)),
					self.critic2_old(info_stacked["obs_next_nodelay"].reshape(flatten_num,-1), act_.reshape(flatten_num,-1)) \
					if self.critic_use_oracle_obs else \
					self.critic2_old(obs_next_stacked.reshape(flatten_num,-1), act_.reshape(flatten_num,-1))
				) - self._alpha * obs_next_result.log_prob.reshape(flatten_num,-1)
				target_q = target_q.reshape(*bsz_len_shape, -1)
			else:
				raise NotImplementedError
		else:
			target_q = torch.min(
				self.critic1_old(batch.info["obs_next_nodelay"], act_) \
				if self.critic_use_oracle_obs else \
				self.critic1_old(batch.obs_next, act_),
				self.critic2_old(batch.info["obs_next_nodelay"], act_) \
				if self.critic_use_oracle_obs else \
				self.critic2_old(batch.obs_next, act_)
			) - self._alpha * obs_next_result.log_prob

		return target_q

	def process_fn(
		self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
	) -> Batch:
		# add basic keys
		if self.critic_use_oracle_obs or self.actor_use_oracle_obs:
			prev_batch = buffer[buffer.prev(indices)]
			batch.info["obs_nodelay"] = prev_batch.info["obs_next_nodelay"]
		if self.global_cfg.historical_act:
			if self.global_cfg.historical_act.type == "cat_mlp":
				pass # ! check whether we need more keys
			elif self.global_cfg.historical_act.type == "stack_rnn":
				# get [{obs_t-N, act_t-N-1}, {obs_t-N+1, act_t-N}, ..., {obs_t-1, act_t-2}, {obs_t, act_t-1}]
				# historical_obs = buffer.get(indices, "obs", stack_num=self.global_cfg.historical_act.num)
				# if len(historical_obs.shape) == 2: historical_obs = historical_obs[:,None,:]
				# historical_act = buffer.get(buffer.prev(indices), "act", stack_num=self.global_cfg.historical_act.num)
				# if len(historical_act.shape) == 2: historical_act = historical_act[:,None,:]
				batch.obs_next = buffer.get(buffer.next(indices), "obs", stack_num=self.global_cfg.historical_act.num) # TODO for kv
				batch.obs = buffer.get(indices, "obs", stack_num=self.global_cfg.historical_act.num)
				batch.act = buffer.get(indices, "act", stack_num=self.global_cfg.historical_act.num)
				batch.obs_stack_act = np.concatenate([self.get_actor_obs(batch,"cur"), batch.act], axis=-1)
				batch.info = buffer.get(indices, "info", stack_num=self.global_cfg.historical_act.num)
				batch.info["obs_nodelay"] = buffer.get(buffer.prev(indices), "info", stack_num=self.global_cfg.historical_act.num)["obs_next_nodelay"]
				batch.is_another_episode = None
				indices_stacked = []
				latest_step = indices
				for i in range(self.global_cfg.historical_act.num):
					indices_stacked.insert(0, latest_step)
					latest_step = buffer.prev(latest_step)
				indices_stacked = np.stack(indices_stacked, axis=-1)
				indices_stacked_prev = buffer.prev(indices_stacked)
				start = (indices_stacked == indices_stacked_prev)
				# index                [0, 0, 0, 0, 1, 2, 3, 4]
				# index_prev		   [0, 0, 0, 0, 0, 1, 2, 3]
				# cur==prev            [1, 1, 1, 1, 0, 0, 0, 0]
				# is_another_episode   [1, 1, 1, 0, 0, 0, 0, 0]
				for b in range(start.shape[0]): # set the last True to False
					for i in range(start.shape[1], 0, -1):
						if start[b,i-1]:
							start[b,i-1] = False
							break
				batch.is_another_episode = start
		# if not self.actor_rnn:
		# 	if self.actor_input_act:
		# 		batch.info["act_prev"] = buffer[buffer.prev(indices)].act
		# 	else:
		# 		batch.info["historical_act_next"] = buffer[buffer.next(indices)].info["historical_act"]
		# else:
		# 	if self.actor_input_act:
		# 		batch.info["stacked_act"] = buffer.get(indices, "act")
		# 		batch.info["stacked_act_prev"] = buffer.get(buffer.prev(indices), "act")
		# add return key (need basic keys)
		batch = self.compute_nstep_return(
			batch, buffer, indices, self._target_q, self._gamma, self._n_step,
			self._rew_norm
		)
		# end flag
		batch.is_preprocessed = True
		return batch

	@staticmethod
	def compute_nstep_return(
		batch: Batch,
		buffer: ReplayBuffer,
		indice: np.ndarray,
		target_q_fn: Callable[[ReplayBuffer, np.ndarray], torch.Tensor],
		gamma: float = 0.99,
		n_step: int = 1,
		rew_norm: bool = False,
	) -> Batch:
		r"""Compute n-step return for Q-learning targets.

		.. math::
			G_t = \sum_{i = t}^{t + n - 1} \gamma^{i - t}(1 - d_i)r_i +
			\gamma^n (1 - d_{t + n}) Q_{\mathrm{target}}(s_{t + n})

		where :math:`\gamma` is the discount factor, :math:`\gamma \in [0, 1]`,
		:math:`d_t` is the done flag of step :math:`t`.

		:param Batch batch: a data batch, which is equal to buffer[indice].
		:param ReplayBuffer buffer: the data buffer.
		:param function target_q_fn: a function which compute target Q value
			of "obs_next" given data buffer and wanted indices.
		:param float gamma: the discount factor, should be in [0, 1]. Default to 0.99.
		:param int n_step: the number of estimation step, should be an int greater
			than 0. Default to 1.
		:param bool rew_norm: normalize the reward to Normal(0, 1), Default to False.

		:return: a Batch. The result will be stored in batch.returns as a
			torch.Tensor with the same shape as target_q_fn's return tensor.
		"""
		assert not rew_norm, \
			"Reward normalization in computing n-step returns is unsupported now."
		rew = buffer.rew
		bsz = len(indice)
		indices = [indice]
		for _ in range(n_step - 1):
			indices.append(buffer.next(indices[-1]))
		indices = np.stack(indices)
		# terminal indicates buffer indexes nstep after 'indice',
		# and are truncated at the end of each episode
		terminal = indices[-1]
		with torch.no_grad():
			target_q_torch = target_q_fn(buffer, terminal)  # (bsz, ?)
		target_q = to_numpy(target_q_torch.reshape(bsz, -1))
		target_q = target_q * BasePolicy.value_mask(buffer, terminal).reshape(-1, 1)
		end_flag = buffer.done.copy()
		end_flag[buffer.unfinished_index()] = True
		target_q = _nstep_return(rew, end_flag, target_q, indices, gamma, n_step)
		batch.returns = to_torch_as(target_q, target_q_torch)
		if hasattr(batch, "weight"):  # prio buffer update
			batch.weight = to_torch_as(batch.weight, target_q_torch)
		return batch

	def forward(  # type: ignore
		self,
		batch: Batch,
		state: Optional[Union[dict, Batch, np.ndarray]] = None,
		input: str = "obs",
		**kwargs: Any,
	) -> Batch:
		###ACTOR_FORWARD note that env.step would call this function automatically
		obs = batch[input]
		if self.global_cfg.historical_act:
			if self.global_cfg.historical_act.type == "cat_mlp":
				if hasattr(batch, "is_preprocessed") and batch.is_preprocessed: # offline learn
					assert input == "obs_cat_act", "input should be obs_cat_act for cat_mlp historical model"
					obs = batch[input] # check whether CAT OBS is used
				else: # online input
					if len(batch.act.shape) == 0: # first step (zero cat)
						obs = np.zeros([obs.shape[0], self.actor.nn.input_size])
					else: # normal step
						# every element of obs and obs_next should be the same
						assert (batch["obs"] == batch["obs_next"]).all() # TODO DEBUG ONLY remove later
						obs = np.concatenate([self.get_actor_obs(batch,"next"), batch.info["historical_act"]], axis=-1) \
							if self.global_cfg.historical_act.num > 0 else self.get_actor_obs(batch,"next")
				logits, hidden = self.actor(obs, state=state, info=batch.info)
				pass
			elif self.global_cfg.historical_act.type == "stack_rnn":
				if hasattr(batch, "is_preprocessed") and batch.is_preprocessed: # offline learn
					assert input == "obs_stack_act", "input should be obs_stack_act for stack_rnn historical model"
					obs = batch[input]
					logits, hidden = self.actor(obs, state=state, info=batch.info)
				else: # online input - cat(act, obs))
					if len(batch.act.shape) == 0: # first step
						obs = np.zeros([obs.shape[0], self.actor.nn.input_size])
					else:
						obs = np.concatenate([obs, batch.info["prev_act"]], axis=-1)
					logits, hidden = self.actor(obs, state=state, info=batch.info)
					logits = tuple([logit.squeeze(0) for logit in logits])
			else:
				raise NotImplementedError(f"historical_act.type {self.global_cfg.historical_act.type} not implemented")
		else: # normal mode
			obs = batch[input]
			logits, hidden = self.actor(obs, state=state, info=batch.info)

		# if not self.actor_input_act: # ! TODO check
		# 	if self.historical_act:
		# 		if len(batch.act.shape) == 0: # online - first step, when act is not available
		# 			obs = np.zeros([obs.shape[0], self.actor.nn.input_size])
		# 		else:
		# 			if input == "obs":
		# 				obs = np.concatenate([obs, batch.info["historical_act"]], axis=-1)
		# 			elif input == "obs_next":
		# 				raise NotImplementedError(f"input {input} not implemented")
		# 			elif input == "obs_cat_act":
		# 				obs = batch.obs_cat_act
		# 			else:
		# 				raise NotImplementedError(f"input {input} not implemented")
		# 	logits, hidden = self.actor(obs, state=state, info=batch.info)
		# else:
		# 	if not self.actor_rnn:
		# 		assert len(obs.shape) == 2, f"obs.shape {obs.shape} != 2"
		# 		if obs.shape[1] == self.actor.nn.input_size: # (B, s+act_dim) offline
		# 			assert input == "obs_act", f"input {input} != obs_act"
		# 			obs_act = obs
		# 		else:
		# 			if len(batch.act.shape) == 0: # online - first step, when act is not available
		# 				obs_act = np.zeros([obs.shape[0], self.actor.nn.input_size])
		# 			else: # online - normal
		# 				obs_act = np.concatenate([obs, batch.act], axis=1)
		# 	else:
		# 		if len(obs.shape) == 2: # (1, s_dim) online
		# 			if len(batch.act.shape) == 0: # online - first step, when act is not available
		# 				obs_act = np.zeros([obs.shape[0], 1, self.actor.nn.input_size])
		# 			else: # online - normal
		# 				obs_act = np.concatenate([obs, batch.info["prev_act"]], axis=1)
		# 		elif len(obs.shape) == 3: # (B, L, s+a_dim) offline self.learn
		# 			assert input == "obs_act", f"input {input} != obs_act"
		# 			assert obs.shape[2] == self.actor.nn.input_size, f"obs.shape[2] {obs.shape[2]} != self.actor.nn.input_size {self.actor.nn.input_size}"
		# 			obs_act = obs
		# 		else:
		# 			raise ValueError(f"obs.shape {obs.shape} is not supported")
		# 	logits, hidden = self.actor(obs_act, state=state, info=batch.info)
			
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

	def get_critic_obs(self, batch: Batch, cur_or_next):
		"""Get obs for critic.

		Args:
			batch (Batch): batch data
			cur_or_next (str): "cur" or "next"

		Returns:
			np.ndarray: obs for critic
		"""
		if cur_or_next == "cur":
			if self.actor_use_oracle_obs: obs = batch.info["obs_nodelay"]
			else: obs = batch.obs
		elif cur_or_next == "next":
			if self.actor_use_oracle_obs: obs = batch.info["obs_next_nodelay"]
			else: obs = batch.obs_next
		return obs
	
	def get_actor_obs(self, batch: Batch, cur_or_next):
		"""Get obs for actor.

		Args:
			batch (Batch): batch data
			cur_or_next (str): "cur" or "next"

		Returns:
			np.ndarray: obs for actor
		"""
		if cur_or_next == "cur":
			if self.actor_use_oracle_obs: obs = batch.info["obs_nodelay"]
			else: obs = batch.obs
		elif cur_or_next == "next":
			if self.actor_use_oracle_obs: obs = batch.info["obs_next_nodelay"]
			else: obs = batch.obs_next
		return obs

# net

class Critic(nn.Module):
    """Simple critic network. Will create an actor operated in continuous \
    action space with structure of preprocess_net ---> 1(q value).

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.
    :param linear_layer: use this module as linear layer. Default to nn.Linear.
    :param bool flatten_input: whether to flatten input data for the last layer.
        Default to True.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        hidden_sizes: Sequence[int] = (),
        device: Union[str, int, torch.device] = "cpu",
        preprocess_net_output_dim: Optional[int] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
        flatten_input: bool = True,
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = 1
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
        self.last = MLP(
            input_dim,  # type: ignore
            1,
            hidden_sizes,
            device=self.device,
            linear_layer=linear_layer,
            flatten_input=flatten_input,
        )

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        act: Optional[Union[np.ndarray, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> torch.Tensor:
        """Mapping: (s, a) -> logits -> Q(s, a)."""
        obs = torch.as_tensor(
            obs,
            device=self.device,
            dtype=torch.float32,
        ).flatten(1)
        if act is not None:
            act = torch.as_tensor(
                act,
                device=self.device,
                dtype=torch.float32,
            ).flatten(1)
            obs = torch.cat([obs, act], dim=1)
        logits, hidden = self.preprocess(obs)
        logits = self.last(logits)
        return logits

class CustomRecurrentCritic(Critic):
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
		historical_act: str = None,
		global_cfg: object = None,
	) -> None:
		# ! TODO add mlp_softmax ! remove extra cfg
		# ! TODO add rnn dummy mechinism introduction
		super().__init__()
		self.global_cfg = global_cfg # ! TODO customize network structure
		self.device = device
		self.rnn_layer_num = rnn_layer_num
		input_dim = int(np.prod(state_shape))
		action_dim = int(np.prod(action_shape))
		if self.global_cfg.historical_act: # e.g. {type: "cat-8"}
			if self.global_cfg.historical_act.type == "cat_mlp":
				input_dim += action_dim * self.global_cfg.historical_act.num
				assert rnn_layer_num == 0, "rnn_layer_num must be 0 when using historical_act"
				assert concat == False, "concat must be False when using historical_act"
			elif self.global_cfg.historical_act.type == "stack_rnn":
				input_dim += action_dim
				assert rnn_layer_num > 0, "rnn_layer_num must be > 0 when using historical_act"
				assert concat == False, "concat must be False when using historical_act"
			else:
				raise ValueError(f"historical_act.type {self.global_cfg.historical_act.type} is not supported")
		else:
			if concat:
				input_dim += action_dim
		if rnn_layer_num:
			self.nn = nn.GRU(
				input_size=input_dim,
				hidden_size=rnn_hidden_layer_size,
				num_layers=rnn_layer_num,
				batch_first=True,
			)
		else:
			self.nn = DummyNet(input_dim=input_dim, input_size=input_dim)
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
				rnn_hidden_layer_size if rnn_layer_num else input_dim,  # type: ignore
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
        act: Optional[Union[np.ndarray, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
	) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
		"""Almost the same as :class:`~tianshou.utils.net.common.Recurrent`."""
		obs = torch.as_tensor(
			obs,
			device=self.device,
			dtype=torch.float32,
		)
		### forward rnn
		# obs [bsz, len, dim] (training) or [bsz, dim] (evaluation)
		# In short, the tensor's shape in training phase is longer than which
		# in evaluation phase. 
		if self.rnn_layer_num: # use rnn
			if len(obs.shape) == 2:
				obs = obs.unsqueeze(-2) # TODO seems not good
			self.nn.flatten_parameters()
			if state is None:
				obs, hidden = self.nn(obs)
			else:
				# we store the stack data in [bsz, len, ...] format
				# but pytorch rnn needs [len, bsz, ...]
				obs, hidden = self.nn(
					obs, 
					state["hidden"].transpose(0, 1).contiguous()
				)
			logits = obs
		else: # skip rnn
			logits = obs
		### forward mlp
		mu = self.flatten_foward(self.mu, logits)
		if not self._unbounded:
			mu = self._max * torch.tanh(mu)
		if self._c_sigma:
			sigma = self.flatten_foward(self.sigma, logits)
			sigma = torch.clamp(sigma, min=self.SIGMA_MIN, max=self.SIGMA_MAX).exp()
		else:
			shape = [1] * len(mu.shape)
			shape[1] = -1
			sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
		# please ensure the first dim is batch size: [bsz, len, ...]
		# hidden and cell are [num_layers, bsz, hidden_size]
		return (mu, sigma), {
			"hidden": hidden.transpose(0, 1).detach(),
			# "cell": cell.transpose(0, 1).detach()
		} if self.rnn_layer_num else None
	
	def flatten_foward(self, model, input):
		""" flattent the input except the last dim, then forward, then convert back
			input: [N_0, N_1, ..., N_{n-1}, dim]
			output: [N_0 * N_1 * ... * N_{n-1}, dim]
		"""
		shape = input.shape
		input = input.reshape(-1, shape[-1])
		output = model(input)
		output = output.reshape(*shape[:-1], -1)
		return output

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
		historical_act: str = None,
		global_cfg: object = None,
	) -> None:
		# ! TODO add mlp_softmax ! remove extra cfg
		# ! TODO add rnn dummy mechinism introduction
		super().__init__()
		self.global_cfg = global_cfg # ! TODO customize network structure
		self.device = device
		self.rnn_layer_num = rnn_layer_num
		input_dim = int(np.prod(state_shape))
		action_dim = int(np.prod(action_shape))
		if self.global_cfg.historical_act: # e.g. {type: "cat-8"}
			if self.global_cfg.historical_act.type == "cat_mlp":
				input_dim += action_dim * self.global_cfg.historical_act.num
				assert rnn_layer_num == 0, "rnn_layer_num must be 0 when using historical_act"
				assert concat == False, "concat must be False when using historical_act"
			elif self.global_cfg.historical_act.type == "stack_rnn":
				input_dim += action_dim
				assert rnn_layer_num > 0, "rnn_layer_num must be > 0 when using historical_act"
				assert concat == False, "concat must be False when using historical_act"
			else:
				raise ValueError(f"historical_act.type {self.global_cfg.historical_act.type} is not supported")
		else:
			if concat:
				input_dim += action_dim
		if rnn_layer_num:
			self.nn = nn.GRU(
				input_size=input_dim,
				hidden_size=rnn_hidden_layer_size,
				num_layers=rnn_layer_num,
				batch_first=True,
			)
		else:
			self.nn = DummyNet(input_dim=input_dim, input_size=input_dim)
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
				rnn_hidden_layer_size if rnn_layer_num else input_dim,  # type: ignore
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
		### forward rnn
		# obs [bsz, len, dim] (training) or [bsz, dim] (evaluation)
		# In short, the tensor's shape in training phase is longer than which
		# in evaluation phase. 
		if self.rnn_layer_num: # use rnn
			if len(obs.shape) == 2:
				obs = obs.unsqueeze(-2) # TODO seems not good
			self.nn.flatten_parameters()
			if state is None:
				obs, hidden = self.nn(obs)
			else:
				# we store the stack data in [bsz, len, ...] format
				# but pytorch rnn needs [len, bsz, ...]
				obs, hidden = self.nn(
					obs, 
					state["hidden"].transpose(0, 1).contiguous()
				)
			logits = obs
		else: # skip rnn
			logits = obs
		### forward mlp
		mu = self.flatten_foward(self.mu, logits)
		if not self._unbounded:
			mu = self._max * torch.tanh(mu)
		if self._c_sigma:
			sigma = self.flatten_foward(self.sigma, logits)
			sigma = torch.clamp(sigma, min=self.SIGMA_MIN, max=self.SIGMA_MAX).exp()
		else:
			shape = [1] * len(mu.shape)
			shape[1] = -1
			sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
		# please ensure the first dim is batch size: [bsz, len, ...]
		# hidden and cell are [num_layers, bsz, hidden_size]
		return (mu, sigma), {
			"hidden": hidden.transpose(0, 1).detach(),
			# "cell": cell.transpose(0, 1).detach()
		} if self.rnn_layer_num else None
	
	def flatten_foward(self, model, input):
		""" flattent the input except the last dim, then forward, then convert back
			input: [N_0, N_1, ..., N_{n-1}, dim]
			output: [N_0 * N_1 * ... * N_{n-1}, dim]
		"""
		shape = input.shape
		input = input.reshape(-1, shape[-1])
		output = model(input)
		output = output.reshape(*shape[:-1], -1)
		return output



# utils

class DummyNet(nn.Module):
	"""Return input as output."""
	def __init__(self, **kwargs):
		super().__init__()
		# set all kwargs as self.xxx
		for k, v in kwargs.items():
			setattr(self, k, v)
		
	def forward(self, x):
		return x

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
