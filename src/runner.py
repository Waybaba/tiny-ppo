import pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
from typing import Callable, Any, Dict, List, Optional, Type, Union, Tuple
from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as
import numpy as np
from time import time
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
				assert hasattr(self.actor, "act_num"), "actor must have act_num attribute"
				act_num = self.actor.act_num
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
		self.global_cfg = kwargs.pop("global_cfg")
		self.init_wandb_summary()
		if self.global_cfg.actor_input.history_merge_method == "cat_mlp":
			assert actor.net.rnn_layer_num == 0, "cat_mlp should not be used with recurrent actor"
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
		current_q, critic_state = critic(batch.critic_input_cur_offline)
		target_q = torch.tensor(batch.returns).to(current_q.device)
		td = current_q.flatten() - target_q.flatten()
		if self.global_cfg.actor_input.history_merge_method in ["none", "cat_mlp"]:
			critic_loss = (td.pow(2) * weight).mean()
		else:
			raise NotImplementedError
			critic_loss = (td.pow(2) * weight)
			# critic_loss = critic_loss.reshape(*bsz_len_shape, -1)
			critic_loss = critic_loss[:, self.global_cfg.historical_act.burnin_num:] # ! burn in num
			critic_loss = critic_loss.mean()
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
		batch.weight = (td1 + td2) / 2.0  # prio-buffer # TODO check complibility with burn in mask

		# actor ###ACTOR_FORWARD
		# obs_result = self(batch, input="actor_input_cur")
		# act = obs_result.act
		
		### CRITIC_FORWARD
		current_q1a, state_ = self.critic1(batch.critic_input_cur_online)
		current_q2a, state_ = self.critic2(batch.critic_input_cur_online)
		current_q1a = current_q1a.flatten()
		current_q2a = current_q2a.flatten()

		if self.global_cfg.actor_input.history_merge_method == "stack_rnn":
			raise NotImplementedError("check the code below to use the method in critic")
			actor_loss = self._alpha * obs_result.log_prob.flatten() - \
				torch.min(current_q1a, current_q2a)
			actor_loss = actor_loss.reshape(*bsz_len_shape, -1)
			actor_loss = actor_loss[:, self.global_cfg.historical_act.burnin_num:]
			actor_loss = actor_loss.mean()
		elif self.global_cfg.actor_input.history_merge_method in ["cat_mlp", "none"]:
			actor_loss = (
				self._alpha * batch.log_prob_cur.flatten() -
				torch.min(current_q1a, current_q2a)
			).mean()
		else:
			raise NotImplementedError
		self.actor_optim.zero_grad()
		actor_loss.backward()
		self.actor_optim.step()

		if self._is_auto_alpha:
			log_prob = batch.log_prob_cur.detach() + self._target_entropy
			# please take a look at issue #258 if you'd like to change this line
			alpha_loss = -(self._log_alpha * log_prob).mean()
			self._alpha_optim.zero_grad()
			alpha_loss.backward()
			self._alpha_optim.step()
			self._alpha = self._log_alpha.detach().exp()

		self.sync_weight()

		result = {
			"learn/loss_actor": actor_loss.item(),
			"learn/loss_critic1": critic1_loss.item(),
			"learn/loss_critic2": critic2_loss.item(),
		}
		if self._is_auto_alpha:
			result["learn/target_entropy"] = self._target_entropy
			result["learn/loss_alpha"] = alpha_loss.item()
			result["learn/_log_alpha"] = self._log_alpha.item()
			result["learn/alpha"] = self._alpha.item()  # type: ignore
		### log - learn
		if not hasattr(self, "learn_step"): self.learn_step = 1
		if not hasattr(self, "start_time"): self.start_time = time()
		self.learn_step += 1
		if self.learn_step % self.global_cfg.log_interval == 0:
			minutes = (time() - self.start_time) / 60
			to_logs = {
				"learn/loss_actor": actor_loss.item(),
				"learn/loss_critic1": critic1_loss.item(),
				"learn/loss_critic2": critic2_loss.item(),
			}
			if self._is_auto_alpha:
				to_logs["learn/target_entropy"] = self._target_entropy
				to_logs["learn/loss_alpha"] = alpha_loss.item()
				to_logs["learn/_log_alpha"] = self._log_alpha.item()
				to_logs["learn/alpha"] = self._alpha.item()
			wandb.log(to_logs, step=self.learn_step)
			# wandb.log({"time_related/learn_step": self.train_env_infer_step}, step=int(minutes))
		return result
	
	def process_fn(
		self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
	) -> Batch:
		"""
		ps. ensure that all process is after the process_fn. The reason is that the we 
		use to_torch to move all the data to the device. So the following operations 
		should be consistent
		"""
		batch.is_preprocessed = True # for distinguishing whether the batch is from env
		# init
		prev_batch = buffer[buffer.prev(indices)]
		batch.info["obs_nodelay"] = prev_batch.info["obs_next_nodelay"] # for first step, the obs is not delayed
		batch.to_torch(device=self.actor.device) # move all to self.device
		### actor input
		if self.global_cfg.actor_input.history_merge_method == "none":
			batch.actor_input_cur = self.get_obs_base(batch, "actor", "cur")
			batch.actor_input_next = self.get_obs_base(batch, "actor", "next")
		elif self.global_cfg.actor_input.history_merge_method == "cat_mlp":
			assert self.global_cfg.actor_input.history_num >= 0
			# ! TODO when met start, it should be the same with the first step instead of zero in the env
			if self.global_cfg.actor_input.history_num > 0:
				# set start as zero
				indices_buf = []
				last_indices = indices
				for _ in range(self.global_cfg.actor_input.history_num):
					last_indices = buffer.prev(last_indices)
					indices_buf.insert(0, last_indices)
				indices_buf = np.stack(indices_buf, axis=-1)
				act_prev_2 = buffer[indices_buf].act # ! assert == 
				start_indices = indices_buf == buffer.prev(indices_buf)
				act_prev = buffer.get(buffer.prev(indices), "act", stack_num=self.global_cfg.actor_input.history_num)
				if self.global_cfg.actor_input.history_num == 1: act_prev = np.expand_dims(act_prev, axis=-2)
				act_prev[start_indices] = 0
				act_prev = torch.from_numpy(act_prev).to(self.actor.device)
				act_prev = act_prev.reshape(act_prev.shape[0], -1) # flatten (batch, history_num * act_dim)
				# DEBUG
				# assert act_prev[0] == batch.info["historical_act"][0]
				batch.actor_input_cur = torch.cat([
					self.get_obs_base(batch, "actor", "cur"),
					act_prev], dim=-1)
				indices_buf = []
				last_indices = indices
				for _ in range(self.global_cfg.actor_input.history_num):
					indices_buf.insert(0, last_indices)
					last_indices = buffer.prev(last_indices)
				indices_buf = np.stack(indices_buf, axis=-1)
				start_indices = indices_buf == buffer.prev(indices_buf)
				act_cur = buffer.get(indices, "act", stack_num=self.global_cfg.actor_input.history_num)
				if self.global_cfg.actor_input.history_num == 1: act_cur = np.expand_dims(act_cur, axis=-2)
				act_cur[start_indices] = 0
				act_cur = torch.from_numpy(act_cur).to(self.actor.device)
				act_cur = act_cur.reshape(act_cur.shape[0], -1) # flatten (batch, history_num * act_dim)
				batch.actor_input_next = torch.cat([
					self.get_obs_base(batch, "actor", "next"),
					act_cur], dim=-1)
			else:
				batch.actor_input_cur = self.get_obs_base(batch, "actor", "cur")
				batch.actor_input_next = self.get_obs_base(batch, "actor", "next")
		elif self.global_cfg.actor_input.history_merge_method == "stack_rnn":
			raise NotImplementedError
		else:
			raise ValueError("unknown history_merge_method: {}".format(self.global_cfg.actor_input.history_merge_method))
		# critic input
		if self.global_cfg.critic_input.history_merge_method == "none":
			actor_result_cur = self.forward(batch, input="actor_input_cur")
			actor_result_next = self.forward(batch, input="actor_input_next")
			batch.critic_input_cur_offline = torch.cat([
				self.get_obs_base(batch, "critic", "cur"),
				batch.act], dim=-1)
			batch.critic_input_cur_online = torch.cat([
				self.get_obs_base(batch, "critic", "cur"),
				actor_result_cur.act], dim=-1)
			batch.critic_input_next_online = torch.cat([
				self.get_obs_base(batch, "critic", "next"),
				actor_result_next.act], dim=-1).detach()
			batch.log_prob_cur = actor_result_cur.log_prob
			batch.log_prob_next = actor_result_next.log_prob # TODO remove log_prob
		elif self.global_cfg.critic_input.history_merge_method == "cat_mlp":
			raise NotImplementedError
		elif self.global_cfg.critic_input.history_merge_method == "stack_rnn":
			raise NotImplementedError
		else:
			raise ValueError("unknown history_merge_method: {}".format(self.global_cfg.critic_input.history_merge_method))
		
		# batch.returns = self.compute_return_custom(batch)
		if "from_target_q" not in batch:
			batch = self.compute_nstep_return( # ! TODO
				batch, buffer, indices, self._target_q, self._gamma, self._n_step,
				self._rew_norm
			)
		# end flag
		batch.is_preprocessed = True
		return batch

	def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
		batch = buffer[indices]  # batch.obs: s_{t+n}
		batch.from_target_q = True
		batch = self.process_fn(batch, buffer, indices)
		target_q = torch.min(
			self.critic1_old(batch.critic_input_next_online)[0],
			self.critic2_old(batch.critic_input_next_online)[0],
		) - self._alpha * batch.log_prob_next
		return target_q

	def process_online_batch(
		self, batch: Batch
		):
		obs = batch.obs
		# if first step when act is none
		if self.global_cfg.actor_input.history_merge_method == "cat_mlp":
			if len(batch.act.shape) == 0: # first step (zero cat) # TODO check stack rnn and cat mlp
				obs = np.zeros([obs.shape[0], self.actor.net.input_dim])
			else: # normal step
				if (len(batch.info["historical_act"].shape) == 1 and batch.info["historical_act"].shape[0] == 1):
					# ps. historical_act == None when no error
					assert batch.info["historical_act"][0] == False
					obs = np.concatenate([
						self.get_obs_base(batch, "actor", "next"),
					], axis=-1)
				elif (len(batch.info["historical_act"].shape) == 2 and batch.info["historical_act"].shape[0] == 1):
					obs = np.concatenate([
						self.get_obs_base(batch, "actor", "next"),
						batch.info["historical_act"]
					], axis=-1)
				else: raise ValueError("historical_act shape not implemented")
		elif self.global_cfg.actor_input.history_merge_method == "stack_rnn":
			
			raise NotImplementedError(f"stack_rnn not implemented")
		elif self.global_cfg.actor_input.history_merge_method == "none":
			if len(batch.act.shape) == 0: # first step (zero cat)
				obs = np.zeros([obs.shape[0], self.actor.net.input_dim])
			else: # normal step
				# obs = batch.info["obs_next_nodelay"] if self.global_cfg.actor_input.obs_type == "oracle" \
				# 	else batch.obs_next
				obs = self.get_obs_base(batch, "actor", "next")
		else:
			raise ValueError(f"history_merge_method {self.global_cfg.actor_input.history_merge_method} not implemented")
		batch.online_input = obs
		return batch

	def compute_return_custom(self, batch):
		""" custom defined calculation of returns (only one step)
		equations:
			1. returns = rewards + gamma * (1 - done) * next_value
		"""
		with torch.no_grad():
			# get next value
			value_next = torch.min(
				self.critic1_old(batch.critic_input_next_online)[0],
				self.critic2_old(batch.critic_input_next_online)[0],
			) - self._alpha * batch.log_prob_next
			# compute returns
			returns = batch.rew + self._gamma * (1 - batch.done.int()) * value_next.flatten()
			return returns

	def forward(  # type: ignore
		self,
		batch: Batch,
		state: Optional[Union[dict, Batch, np.ndarray]] = None,
		input: str = "obs",
		**kwargs: Any,
	) -> Batch:
		###ACTOR_FORWARD note that env.step would call this function automatically
		if not hasattr(batch, "is_preprocessed") or not batch.is_preprocessed: # online learn
			### process the input from env (online learn).
			input = "online_input"
			batch = self.process_online_batch(batch)
		actor_input = batch[input]
		logits, hidden = self.actor(actor_input, state=state, info=batch.info)
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
		
		### log - train_env_infer
		if not hasattr(self, "train_env_infer_step"): self.train_env_infer_step = 0
		if not hasattr(self, "start_time"): self.start_time = time()
		if input == "online_input" and self.training: 
			self.train_env_infer_step += 1
			if (self.train_env_infer_step % self.global_cfg.log_interval) == 0:
				minutes = (time() - self.start_time) / 60
				to_logs = {
					"train_env_infer/expectedT_1mStep_min": minutes / self.train_env_infer_step * 1e6,
					"train_env_infer/expectedT_1mStep_hr": minutes / self.train_env_infer_step * 1e6 / 60,
					"train_env_infer/time_minutes": minutes,
				}
				wandb.log(to_logs, step=self.train_env_infer_step)
				# wandb.log({"time_related/train_env_infer_step": self.train_env_infer_step}, step=int(time() - self.start_time))
		return Batch(
			logits=logits,
			act=squashed_action,
			state=hidden,
			dist=dist,
			log_prob=log_prob
		)
	
	def init_wandb_summary(self):
		wandb.define_metric("train_env_infer/expectedT_1mStep_min", summary="last")
		wandb.define_metric("train_env_infer/expectedT_1mStep_hr", summary="last")
		wandb.define_metric("key/reward", summary="last")

	def get_obs_base(self, batch, a_or_c, stage):
		""" return the obs base for actor and critic
		the return is depends on self.global_cfg.actor_input.obs_type and \
			self.global_cfg.critic_input.obs_type
		:param batch: batch
		:param a_or_c: "actor" or "critic"
		:param stage: "cur" or "next"
		ps. only called in process stage
		"""
		assert stage in ["cur", "next"]
		assert a_or_c in ["actor", "critic"]
		assert self.global_cfg.actor_input.obs_type in ["normal", "oracle"]
		if a_or_c == "actor":
			if self.global_cfg.actor_input.obs_type == "normal":
				if stage == "cur": return batch.obs
				elif stage == "next": return batch.obs_next
			elif self.global_cfg.actor_input.obs_type == "oracle":
				if stage == "cur": return batch.info["obs_nodelay"]
				elif stage == "next": return batch.info["obs_next_nodelay"]
		elif a_or_c == "critic":
			if self.global_cfg.critic_input.obs_type == "normal":
				if stage == "cur": return batch.obs
				elif stage == "next": return batch.obs_next
			elif self.global_cfg.critic_input.obs_type == "oracle":
				if stage == "cur": return batch.info["obs_nodelay"]
				elif stage == "next": return batch.info["obs_next_nodelay"]
		
# net

class RNN_MLP_Net(nn.Module):
	""" RNNS with MLPs as the core network
	ps. assume input is one dim
	ps. head_num = 1 for critic
	"""
	def __init__(
		self,
		input_dim: int,
		output_dim: int,
		rnn_layer_num: int,
		rnn_hidden_layer_size: int,
		mlp_hidden_sizes: Sequence[int],
		mlp_softmax: bool,  # TODO add
		device: str,
		head_num: int
	):
		super().__init__()
		self.device = device
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.rnn_layer_num = rnn_layer_num
		self.rnn_hidden_layer_size = rnn_hidden_layer_size
		self.mlp_hidden_sizes = mlp_hidden_sizes
		# build rnn
		if rnn_layer_num:
			self.nn = nn.GRU(
				input_size=input_dim,
				hidden_size=rnn_hidden_layer_size,
				num_layers=rnn_layer_num,
				batch_first=True,
			)
		else:
			self.nn = DummyNet(input_dim=input_dim, input_size=input_dim)
		# build mlp
		self.mlps = []
		for i in range(head_num):
			self.mlps.append(
				MLP(
					rnn_hidden_layer_size if rnn_layer_num else input_dim,  # type: ignore
					output_dim,
					mlp_hidden_sizes,
					device=self.device
				)
			)
		self.mlp = nn.ModuleList(self.mlps)
	
	def forward(
		self,
		obs: Union[np.ndarray, torch.Tensor],
		state: Optional[Dict[str, torch.Tensor]] = None,
		info: Dict[str, Any] = {},
		):
		"""
		input
		"""
		obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
		### forward rnn
		# obs [bsz, len, dim] (training) or [bsz, dim] (evaluation)
		# In short, the tensor's shape in training phase is longer than which
		# in evaluation phase. 
		if self.rnn_layer_num: 
			assert len(obs.shape) == 3 or len(obs.shape) == 2
			if len(obs.shape) == 2: obs = obs.unsqueeze(-2) # make seq_len dim
			bsz, len, dim = obs.shape
			self.nn = None
			self.nn.flatten_parameters()
			if state is None: after_rnn, hidden = self.nn(obs)
			else: after_rnn, hidden = self.nn(obs, state["hidden"].transpose(0, 1).contiguous())
		else: # skip rnn
			after_rnn = obs
		### forward mlp # ! TODO actor max min clip
		outputs = []
		for mlp in self.mlps:
			outputs.append(self.flatten_foward(mlp, after_rnn))
		return outputs, {
			"hidden": hidden.transpose(0, 1).detach(),
		} if self.rnn_layer_num else None

	def flatten_foward(self, net, input):
		"""Flatten input for mlp forward, then reshape output to original shape.
		input: 
			mlp: a mlp module
			after_rnn: tensor [N1, N2, ..., Nk, D_in]
		output:
			tensor [N1, N2, ..., Nk, D_out]
		"""
		# flatten
		shape = input.shape
		input = input.reshape(-1, shape[-1])
		# forward
		output = net(input)
		# reshape
		shape = list(shape)
		shape[-1] = output.shape[-1]
		output = output.reshape(*shape)
		return output

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
		state: Optional[Dict[str, torch.Tensor]] = None,
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

class CustomRecurrentCritic(nn.Module):
	def __init__(
		self,
		state_shape: Sequence[int],
		action_shape: Sequence[int],
		**kwargs,
	) -> None:
		super().__init__()
		self.hps = kwargs
		assert len(state_shape) == 1 and len(action_shape) == 1, "now, only support 1d state and action"
		if self.hps["global_cfg"].critic_input.history_merge_method == "cat_mlp":
			input_dim = state_shape[0] + action_shape[0] * self.hps["global_cfg"].actor_input.history_num
			output_dim = 1
		elif self.hps["global_cfg"].critic_input.history_merge_method == "stack_rnn":
			input_dim = state_shape[0] + action_shape[0]
			output_dim = 1
		elif self.hps["global_cfg"].critic_input.history_merge_method == "none":
			input_dim = state_shape[0] + action_shape[0]
			output_dim = 1
		else:
			raise NotImplementedError
		self.net = self.hps["net"](input_dim, output_dim, device=self.hps["device"], head_num=1)

	def forward(
		self,
		critic_input: Union[np.ndarray, torch.Tensor],
		info: Dict[str, Any] = {},
	) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
		assert type(info) == dict, "info should be a dict, check whether missing 'info' as act"
		output, state_ = self.net(critic_input)
		value = output[0]
		return value, state_

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
		**kwargs,
	) -> None:
		super().__init__()
		self.hps = kwargs
		assert len(state_shape) == 1 and len(action_shape) == 1
		self.device = self.hps["device"]
		self.state_shape, self.action_shape = state_shape, action_shape
		self.act_num = action_shape[0]
		if self.hps["global_cfg"].actor_input.history_merge_method == "cat_mlp":
			input_dim = state_shape[0] + action_shape[0] * self.hps["global_cfg"].actor_input.history_num
			output_dim = int(np.prod(action_shape))
		elif self.hps["global_cfg"].actor_input.history_merge_method == "stack_rnn":
			input_dim = state_shape[0] + action_shape[0]
			output_dim = int(np.prod(action_shape))
		elif self.hps["global_cfg"].actor_input.history_merge_method == "none":
			input_dim = int(np.prod(state_shape))
			output_dim = int(np.prod(action_shape))
		else:
			raise NotImplementedError
		self.net = self.hps["net"](input_dim, output_dim, device=self.hps["device"], head_num=2)

	def forward(
		self,
		actor_input: Union[np.ndarray, torch.Tensor],
		state: Optional[Dict[str, torch.Tensor]] = None,
		info: Dict[str, Any] = {},
	) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
		"""Almost the same as :class:`~tianshou.utils.net.common.Recurrent`."""
		### forward
		output, state_ = self.net(actor_input, state)
		assert len(output) == 2, "output should be a tuple of (mu, sigma), as there are two heads for actor network"
		mu, sigma = output
		if not self.hps["unbounded"]:
			mu = self.hps["max_action"] * torch.tanh(mu)
		if self.hps["conditioned_sigma"]:
			sigma = torch.clamp(sigma, min=self.SIGMA_MIN, max=self.SIGMA_MAX).exp()
		else:
			raise NotImplementedError
		return (mu, sigma), state_
	

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
		seed = int(time()) if cfg.seed is None else cfg.seed
		utils.seed_everything(seed) # TODO add env seed
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
		critic1 = cfg.critic1(env.observation_space.shape, action_shape=env.action_space.shape).to(cfg.device)
		critic1_optim = cfg.critic1_optim(critic1.parameters())
		critic2 = cfg.critic2(env.observation_space.shape, action_shape=env.action_space.shape).to(cfg.device)
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
