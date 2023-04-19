import pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
from tianshou.data import Batch, ReplayBuffer
import numpy as np
from time import time
import torch
import torch
import wandb
import numpy as np
from tianshou.policy import SACPolicy
from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union, Callable
import numpy as np
import torch
from torch import nn
from tianshou.utils.net.common import MLP
import tianshou
from torch.utils.tensorboard import SummaryWriter
import utils
from functools import partial
import torch.nn.functional as F


import warnings
warnings.filterwarnings('ignore')
from tianshou.policy import DDPGPolicy
from tianshou.exploration import BaseNoise
from torch.distributions import Independent, Normal
from copy import deepcopy
from rich.progress import Progress
from rich.progress import track
from rich.console import Console


def kl_divergence(mu1, logvar1, mu2, logvar2):
	"""
	mu1, logvar1: mean and log variance of the first Gaussian distribution
	mu2, logvar2: mean and log variance of the second Gaussian distribution
	input:
		mu1, mu2: (B, K)
		logvar1, logvar2: (B, K)
	output:
		kl: (B, )
	"""
	kl = 0.5 * (logvar2 - logvar1 + (torch.exp(logvar1) + (mu1 - mu2).pow(2)) / torch.exp(logvar2) - 1)
	return kl.sum(dim=-1)

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
		next_batch.c_in_curur = next_batch.info["obs_cur"]
		# obs_next_result = self(batch, input="obs_cur")
		obs_next_result = self(next_batch, input="obs_cur")
		act_ = obs_next_result.act
		target_q = torch.min(
			# self.critic1_old(batch.obs_next, act_),
			# self.critic2_old(batch.obs_next, act_),
			self.critic1_old(next_batch.c_in_curur, act_),
			self.critic2_old(next_batch.c_in_curur, act_),
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
		self.state_space = kwargs.pop("state_space")
		self.init_wandb_summary()
		if self.global_cfg.actor_input.history_merge_method == "cat_mlp":
			assert actor.net.rnn_layer_num == 0, "cat_mlp should not be used with recurrent actor"
		super().__init__(
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
		
		assert not (self.global_cfg.actor_input.obs_pred.turn_on and self.global_cfg.actor_input.obs_encode.turn_on), "obs_pred and obs_encode cannot be used at the same time"
		
		if self.global_cfg.actor_input.obs_pred.turn_on:
			self.pred_net = self.global_cfg.actor_input.obs_pred.net(
				state_shape=self.state_space.shape,
				action_shape=kwargs["action_space"].shape,
				global_cfg=self.global_cfg,
			)
			self._pred_optim = self.global_cfg.actor_input.obs_pred.optim(
				self.pred_net.parameters(),
			)
			if self.global_cfg.actor_input.obs_pred.auto_kl_target:
				self.kl_weight_log = torch.tensor([np.log(
					self.global_cfg.actor_input.obs_pred.norm_kl_loss_weight
				)], device=self.actor.device, requires_grad=True)
				self._auto_kl_optim = self.global_cfg.actor_input.obs_pred.auto_kl_optim([self.kl_weight_log])
			else:
				self.kl_weight_log = torch.tensor([np.log(
					self.global_cfg.actor_input.obs_pred.norm_kl_loss_weight
				)], device=self.actor.device)
		
		if self.global_cfg.actor_input.obs_encode.turn_on:
			self.encode_net = self.global_cfg.actor_input.obs_encode.net(
				state_shape=self.state_space.shape,
				action_shape=kwargs["action_space"].shape,
				global_cfg=self.global_cfg,
			)
			self._encode_optim = self.global_cfg.actor_input.obs_encode.optim(
				self.encode_net.parameters(),
			)
			if self.global_cfg.actor_input.obs_encode.auto_kl_target:
				self.kl_weight_log = torch.tensor([np.log(
					self.global_cfg.actor_input.obs_encode.norm_kl_loss_weight
				)], device=self.actor.device, requires_grad=True)
				self._auto_kl_optim = self.global_cfg.actor_input.obs_encode.auto_kl_optim([self.kl_weight_log])
			else:
				self.kl_weight_log = torch.tensor([np.log(
					self.global_cfg.actor_input.obs_encode.norm_kl_loss_weight
				)], device=self.actor.device)
	
	def _mse_optimizer(self,
		batch: Batch, critic: torch.nn.Module, optimizer: torch.optim.Optimizer
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""A simple wrapper script for updating critic network."""
		weight = getattr(batch, "weight", 1.0)
		current_q, critic_state = critic(batch.critic_input_cur_offline)
		target_q = torch.tensor(batch.returns).to(current_q.device)
		td = current_q.flatten() - target_q.flatten()
		critic_loss = (
			(td.pow(2) * weight) * batch.valid_mask.flatten()
		).mean()
		critic_loss = (td.pow(2) * weight)
		critic_loss = critic_loss.mean()
		optimizer.zero_grad()
		critic_loss.backward()
		optimizer.step()
		return td, critic_loss
	
	def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
		to_logs = {}
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

		### actor loss
		actor_loss = ((
			self._alpha * batch.log_prob_cur.flatten() 
			- torch.min(current_q1a, current_q2a)
		) * batch.valid_mask.flatten()
		).mean()

		### pred/encode loss
		if self.global_cfg.actor_input.obs_pred.turn_on:
			pred_loss = (batch.pred_output_cur - batch.info["obs_nodelay"]) ** 2
			pred_loss = pred_loss * batch.valid_mask.unsqueeze(-1)
			pred_loss = pred_loss.mean()
			combined_loss = actor_loss + pred_loss * self.global_cfg.actor_input.obs_pred.pred_loss_weight
			to_logs["learn/obs_pred/loss_pred"] = pred_loss.item()
			to_logs["learn/obs_pred/abs_error_pred"] = pred_loss.item() ** 0.5
			if self.global_cfg.actor_input.obs_pred.net_type == "vae":
				kl_loss = kl_divergence(
					batch.pred_info_cur_mu,
					batch.pred_info_cur_logvar,
					torch.zeros_like(batch.pred_info_cur_mu),
					torch.zeros_like(batch.pred_info_cur_logvar),
				)
				kl_loss = kl_loss * batch.valid_mask.unsqueeze(-1)
				kl_loss = kl_loss.mean()
				combined_loss = combined_loss + kl_loss * self.global_cfg.actor_input.obs_pred.norm_kl_loss_weight
				to_logs["learn/obs_pred/loss_kl"] = kl_loss.item()
				if self.global_cfg.actor_input.obs_pred.auto_kl_target:
					kl_weight_loss = - (kl_loss.detach() - self.global_cfg.actor_input.obs_pred.auto_kl_target) * torch.exp(self.kl_weight_log)
					self._auto_kl_optim.zero_grad()
					kl_weight_loss.backward()
					self._auto_kl_optim.step()
					to_logs["learn/obs_pred/kl_weight_log"] = self.kl_weight_log.detach().cpu().numpy()
					to_logs["learn/obs_pred/kl_weight"] = torch.exp(self.kl_weight_log).detach().cpu().numpy()
			self.actor_optim.zero_grad()
			self._pred_optim.zero_grad()
			combined_loss.backward()
			self.actor_optim.step()
			self._pred_optim.step()
		elif self.global_cfg.actor_input.obs_encode.turn_on:
			kl_loss = kl_divergence(batch.encode_oracle_info_cur_mu, batch.encode_oracle_info_cur_logvar, batch.encode_normal_info_cur_mu, batch.encode_normal_info_cur_logvar)
			kl_loss = kl_loss * batch.valid_mask.unsqueeze(-1)
			kl_loss = kl_loss.mean()
			combined_loss = actor_loss + kl_loss * torch.exp(self.kl_weight_log).detach()
			to_logs["learn/obs_encode/loss_kl"] = kl_loss.item()
			if self.global_cfg.actor_input.obs_encode.pred_loss_weight:
				batch.pred_obs_output_cur, _ = self.encode_net.decode(batch.encode_obs_output_cur)
				pred_loss = (batch.pred_obs_output_cur - batch.info["obs_nodelay"]) ** 2
				pred_loss = pred_loss * batch.valid_mask.unsqueeze(-1)
				pred_loss = pred_loss.mean()
				to_logs["learn/obs_encode/loss_pred"] = pred_loss.item()
				to_logs["learn/obs_encode/abs_error_pred"] = pred_loss.item() ** 0.5
				combined_loss = actor_loss + pred_loss * self.global_cfg.actor_input.obs_pred.pred_loss_weight
			self.actor_optim.zero_grad()
			self._encode_optim.zero_grad()
			combined_loss.backward()
			self.actor_optim.step()
			self._encode_optim.step()
			if self.global_cfg.actor_input.obs_encode.auto_kl_target:
				kl_weight_loss = - (kl_loss.detach() - self.global_cfg.actor_input.obs_encode.auto_kl_target) * torch.exp(self.kl_weight_log)
				self._auto_kl_optim.zero_grad()
				kl_weight_loss.backward()
				self._auto_kl_optim.step()
				to_logs["learn/obs_encode/kl_weight_log"] = self.kl_weight_log.detach().cpu().numpy()
				to_logs["learn/obs_encode/kl_weight"] = torch.exp(self.kl_weight_log).detach().cpu().numpy()
		else:
			self.actor_optim.zero_grad()
			actor_loss.backward()
			self.actor_optim.step()
			
			
		### alpha loss
		if self._is_auto_alpha:
			log_prob = batch.log_prob_cur.detach() + self._target_entropy
			# please take a look at issue #258 if you'd like to change this line
			alpha_loss = (
				-(self._log_alpha * log_prob.flatten()) * batch.valid_mask.flatten()
				).mean()
			self._alpha_optim.zero_grad()
			alpha_loss.backward()
			self._alpha_optim.step()
			self._alpha = self._log_alpha.detach().exp()
		self.sync_weight()
		to_logs["learn/loss_actor"] = actor_loss.item()
		to_logs["learn/loss_critic1"] = critic1_loss.item()
		to_logs["learn/loss_critic2"] = critic2_loss.item()
		if self._is_auto_alpha:
			to_logs["learn/target_entropy"] = self._target_entropy
			to_logs["learn/loss_alpha"] = alpha_loss.item()
			to_logs["learn/_log_alpha"] = self._log_alpha.item()
			to_logs["learn/alpha"] = self._alpha.item()  # type: ignore
		### log - learn
		if not hasattr(self, "learn_step"): self.learn_step = 1
		if not hasattr(self, "start_time"): self.start_time = time()
		self.learn_step += 1
		if self.learn_step % self.global_cfg.log_interval == 0:
			if hasattr(batch, "valid_mask"):
				to_logs["learn/mask_valid_ratio"] = batch.valid_mask.float().mean().item()
			wandb.log(to_logs, commit=self.global_cfg.log_instant_commit)
		return to_logs
	
	def process_fn(
		self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
	) -> Batch:
		"""
		ps. ensure that all process is after the process_fn. The reason is that the we 
		use to_torch to move all the data to the device. So the following operations 
		should be consistent
		"""
		# init
		bsz = len(indices)
		batch.info["obs_nodelay"] = buffer[buffer.prev(indices)].info["obs_next_nodelay"] # (B, T, *)
		batch.valid_mask = buffer.next(indices) != indices
		batch.to_torch(device=self.actor.device) # move all to self.device
		batch.is_preprocessed = True
		### actor input
		if self.global_cfg.actor_input.history_merge_method == "none":
			batch.a_in_cur = self.get_obs_base(batch, "actor", "cur")
			batch.a_in_next = self.get_obs_base(batch, "actor", "next")
		elif self.global_cfg.actor_input.history_merge_method == "cat_mlp":
			assert self.global_cfg.actor_input.history_num >= 0
			if self.global_cfg.actor_input.history_num > 0:
				idx_stack = utils.idx_stack(indices, buffer, self.global_cfg.actor_input.history_num, direction=self.global_cfg.actor_input.trace_direction) # (B, T)
				# del indices
				idx_end = idx_stack[:,-1] # (B, )
				batch_end = buffer[idx_end] # (B, *)
				batch_end.info["obs_nodelay"] = buffer[buffer.prev(idx_end)].info["obs_next_nodelay"] # (B, T, *)
				batch_end.actor_input_cur = torch.cat([
					torch.tensor(self.get_obs_base(batch_end, "actor", "cur"),device=self.actor.device), # (B, T, *)
					self.get_historical_act(idx_end, self.global_cfg.actor_input.history_num, buffer, "cat", self.actor.device) \
					if not self.global_cfg.actor_input.noise_act_debug else \
					torch.normal(size=stacked_batch_prev["act"].reshape(batch_end.obs.shape[0],-1).shape, mean=0., std=1.,device=self.actor.device),
				], dim=-1) # (B, T, obs_dim + act_dim * history_num)
				batch_end.actor_input_next = torch.cat([
					torch.tensor(self.get_obs_base(batch_end, "actor", "next"),device=self.actor.device), # (B, T, *)
					self.get_historical_act(buffer.next(idx_end), self.global_cfg.actor_input.history_num, buffer, "cat", self.actor.device) \
					if not self.global_cfg.actor_input.noise_act_debug else \
					torch.normal(size=stacked_batch_cur["act"].reshape(len(batch_end),-1).shape, mean=0., std=1.,device=self.actor.device),
				], dim=-1) # (B, T, obs_dim + act_dim * history_num)
				# make mask
				if self.global_cfg.actor_input.trace_direction == "next":
					# all that reach end of episode should be invalid
					batch_end.valid_mask = torch.tensor(idx_end != buffer.next(idx_end), device=self.actor.device).int() # (B, T)
				elif self.global_cfg.actor_input.trace_direction == "prev":
					# all are valid while before the first action should be 0 filled
					batch_end.valid_mask = torch.ones(idx_end.shape, device=self.actor.device).int() # (B, T)
				else: raise ValueError("trace_direction should be next or prev")
				# obs_pred & obs_encode
				if self.global_cfg.actor_input.obs_pred.turn_on:
					batch_end.pred_input_cur = batch_end.actor_input_cur
					batch_end.pred_input_next = batch_end.actor_input_next
					batch_end.pred_output_cur, pred_info_cur = self.pred_net(batch_end.pred_input_cur)
					batch_end.pred_output_next, pred_info_next = self.pred_net(batch_end.pred_input_next)
					if self.global_cfg.actor_input.obs_pred.input_type == "obs":
						batch_end.actor_input_cur = batch_end.pred_output_cur
						batch_end.actor_input_next = batch_end.pred_output_next
					elif self.global_cfg.actor_input.obs_pred.input_type == "feat":
						batch_end.actor_input_cur = pred_info_cur["feats"]
						batch_end.actor_input_next = pred_info_next["feats"]
					else:
						raise NotImplementedError
					# detach
					if self.global_cfg.actor_input.obs_pred.middle_detach: 
						batch_end.actor_input_cur = batch_end.actor_input_cur.detach()
						batch_end.actor_input_next = batch_end.actor_input_next.detach()
					if self.global_cfg.actor_input.obs_pred.net_type == "vae":
						batch_end.pred_info_cur_mu = pred_info_cur["mu"]
						batch_end.pred_info_cur_logvar = pred_info_cur["logvar"]
				if self.global_cfg.actor_input.obs_encode.turn_on:
					batch_end.encode_obs_input_cur = batch_end.actor_input_cur
					batch_end.encode_obs_input_next = batch_end.actor_input_next
					batch_end.encode_obs_output_cur, encode_obs_info_cur = self.encode_net.normal_encode(batch_end.encode_obs_input_cur)
					batch_end.encode_obs_output_next, encode_obs_info_next = self.encode_net.normal_encode(batch_end.encode_obs_input_next)
					batch_end.encode_oracle_obs_output_cur, encode_oracle_obs_info_cur = self.encode_net.oracle_encode(batch_end.info["obs_nodelay"])
					batch_end.encode_oracle_obs_output_next, encode_oracle_obs_info_next = self.encode_net.oracle_encode(batch_end.info["obs_next_nodelay"])
					batch_end.encode_normal_info_cur_mu = encode_obs_info_cur["mu"]
					batch_end.encode_normal_info_cur_logvar = encode_obs_info_cur["logvar"]
					batch_end.encode_oracle_info_cur_mu = encode_oracle_obs_info_cur["mu"]
					batch_end.encode_oracle_info_cur_logvar = encode_oracle_obs_info_cur["logvar"]
					if self.global_cfg.actor_input.obs_encode.train_eval_async == True:
						batch_end.actor_input_cur = batch_end.encode_oracle_obs_output_cur
						batch_end.actor_input_next = batch_end.encode_oracle_obs_output_next
					elif self.global_cfg.actor_input.obs_encode.train_eval_async == False:
						batch_end.actor_input_cur = batch_end.encode_obs_output_cur
						batch_end.actor_input_next = batch_end.encode_obs_output_next
					else:
						raise ValueError("batch_end error")
					if self.global_cfg.actor_input.obs_encode.pred_loss_weight:
						batch_end.pred_obs_output_cur, _ = self.encode_net.decode(batch_end.encode_obs_output_cur)
					if self.global_cfg.actor_input.obs_encode.before_policy_detach:
						batch_end.actor_input_cur = batch_end.actor_input_cur.detach()
						batch_end.actor_input_next = batch_end.actor_input_next.detach()
				# end
				if "from_target_q" in batch: batch_end.from_target_q = batch.from_target_q
				if "is_preprocessed" in batch: batch_end.is_preprocessed = batch.is_preprocessed
				indices = idx_end
				batch = batch_end
				batch.to_torch(device=self.actor.device)
			else:
				batch.a_in_cur = self.get_obs_base(batch, "actor", "cur")
				batch.a_in_next = self.get_obs_base(batch, "actor", "next")
		elif self.global_cfg.actor_input.history_merge_method == "stack_rnn":
			assert self.global_cfg.actor_input.history_num > 1, "stack_rnn requires history_num > 1, ususally, it would be 20,40,... since we process long history when running online."
			assert self.global_cfg.actor_input.history_num > self.global_cfg.actor_input.burnin_num, "stack_rnn requires history_num > burnin_num, ususally, it could be a little larger than burnin_num"
			idx_stack = utils.idx_stack(indices, buffer, self.global_cfg.actor_input.history_num, direction=self.global_cfg.actor_input.trace_direction) # (B, T)
			del indices
			idx_end = idx_stack[:,-1] # (B, )
			batch_end = buffer[idx_end] # (B, *)
			batch_stack = buffer[idx_stack] # (B, T, *)
			batch_end.info["obs_nodelay"] = buffer[buffer.prev(idx_end)].info["obs_next_nodelay"] # (B, *)
			batch_stack.info["obs_nodelay"] = buffer[buffer.prev(idx_stack)].info["obs_next_nodelay"] # (B, T, *)
			batch_stack.actor_input_cur = torch.cat([
				torch.tensor(self.get_obs_base(buffer[idx_stack], "actor", "cur"),device=self.actor.device), # (B, T, obs_dim) # (B, T, act_dim)
				self.get_historical_act(idx_end, self.global_cfg.actor_input.history_num, buffer, "stack", self.actor.device) \
				if not self.global_cfg.actor_input.noise_act_debug else \
				torch.normal(size=stacked_batch_prev["act"].reshape(batch_end.obs.shape[0],-1).shape, mean=0., std=1.,device=self.actor.device), # TODO
			], dim=-1) # (B, T, obs_dim+act_dim)
			batch_stack.actor_input_next = torch.cat([
				torch.tensor(self.get_obs_base(buffer[idx_stack], "actor", "next"),device=self.actor.device), # (B, T, obs_dim) # (B, T, act_dim)
				self.get_historical_act(buffer.next(idx_end), self.global_cfg.actor_input.history_num, buffer, "stack", self.actor.device) \
				if not self.global_cfg.actor_input.noise_act_debug else \
				torch.normal(size=stacked_batch_cur["act"].reshape(len(batch_end),-1).shape, mean=0., std=1.,device=self.actor.device), # TODO
			], dim=-1) # (B, T, obs_dim+act_dim)
			# make mask
			# if self.global_cfg.actor_input.trace_direction == "next":
			# 	# end step is invalid
			# 	batch_stack.valid_mask = torch.tensor(idx_stack != buffer.next(idx_stack), device=self.actor.device).int() # (B, T)
			# elif self.global_cfg.actor_input.trace_direction == "prev":
			# 	# start step is invalid
			# 	batch_stack.valid_mask = torch.tensor(idx_stack != buffer.prev(idx_stack), device=self.actor.device).int() # (B, T)
			# else: raise ValueError("trace_direction should be next or prev")

			# if the start of the idx reach start or end of the idx reach end, then the whole episode is invalid
			# idx_stack: B, T
			batch_stack.valid_mask = np.ones_like(idx_stack) # (B, T)
			if self.global_cfg.actor_input.seq_mask == True:
				reach_start = idx_stack[:,0] == buffer.prev(idx_stack[:,0]) # (B, )
				reach_end = idx_stack[:,-1] == buffer.next(idx_stack[:,-1]) # (B, )
				batch_stack.valid_mask[reach_start==1,:] = 0
				batch_stack.valid_mask[reach_end==1,:] = 0
			elif self.global_cfg.actor_input.seq_mask == False:
				reach_start = idx_stack == buffer.prev(idx_stack) # (B, )
				reach_end = idx_stack == buffer.next(idx_stack) # (B, )
				batch_stack.valid_mask[reach_start==1] = 0
				batch_stack.valid_mask[reach_end==1] = 0
			else: raise ValueError("seq_mask should be True or False")
			burn_in_num = int(self.global_cfg.actor_input.burnin_num * self.global_cfg.actor_input.history_num) \
			if type(self.global_cfg.actor_input.burnin_num) == float \
			else self.global_cfg.actor_input.burnin_num
			batch_stack.valid_mask[:,:burn_in_num] = 0
			# obs_pred & obs_encode
			if self.global_cfg.actor_input.obs_pred.turn_on:
				batch_stack.pred_input_cur = batch_stack.actor_input_cur
				batch_stack.pred_input_next = batch_stack.actor_input_next # TODO the following
				batch_stack.pred_output_cur, pred_info_cur = self.pred_net(batch_stack.pred_input_cur, state=None)
				batch_stack.pred_output_next, pred_info_next = self.pred_net(batch_stack.pred_input_next, state=None)
				if self.global_cfg.actor_input.obs_pred.input_type == "obs":
					batch_stack.actor_input_cur = batch_stack.pred_output_cur # (B*T, *)
					batch_stack.actor_input_next = batch_stack.pred_output_next # (B*T, *)
				elif self.global_cfg.actor_input.obs_pred.input_type == "feat":
					batch_stack.actor_input_cur = pred_info_cur["feats"] # (B*T, *)
					batch_stack.actor_input_next = pred_info_next["feats"]
				else:
					raise NotImplementedError
				# detach
				if self.global_cfg.actor_input.obs_pred.middle_detach: 
					batch_stack.actor_input_cur = batch_stack.actor_input_cur.detach()
					batch_stack.actor_input_next = batch_stack.actor_input_next.detach()
				if self.global_cfg.actor_input.obs_pred.net_type == "vae":
					raise NotImplementedError("vae for rnn is not implemented yet")
					batch_stack.pred_info_cur_mu = pred_info_cur["mu"]
					batch_stack.pred_info_cur_logvar = pred_info_cur["logvar"]
			if self.global_cfg.actor_input.obs_encode.turn_on:
				raise NotImplementedError("obs_encode is not implemented yet")
				batch_end.encode_obs_input_cur = batch_end.actor_input_cur
				batch_end.encode_obs_input_next = batch_end.actor_input_next
				batch_end.encode_obs_output_cur, encode_obs_info_cur = self.encode_net.normal_encode(batch_end.encode_obs_input_cur)
				batch_end.encode_obs_output_next, encode_obs_info_next = self.encode_net.normal_encode(batch_end.encode_obs_input_next)
				batch_end.encode_oracle_obs_output_cur, encode_oracle_obs_info_cur = self.encode_net.oracle_encode(batch_end.info["obs_nodelay"])
				batch_end.encode_oracle_obs_output_next, encode_oracle_obs_info_next = self.encode_net.oracle_encode(batch_end.info["obs_next_nodelay"])
				batch_end.encode_normal_info_cur_mu = encode_obs_info_cur["mu"]
				batch_end.encode_normal_info_cur_logvar = encode_obs_info_cur["logvar"]
				batch_end.encode_oracle_info_cur_mu = encode_oracle_obs_info_cur["mu"]
				batch_end.encode_oracle_info_cur_logvar = encode_oracle_obs_info_cur["logvar"]
				if self.global_cfg.actor_input.obs_encode.train_eval_async == True:
					batch_end.actor_input_cur = batch_end.encode_oracle_obs_output_cur
					batch_end.actor_input_next = batch_end.encode_oracle_obs_output_next
				elif self.global_cfg.actor_input.obs_encode.train_eval_async == False:
					batch_end.actor_input_cur = batch_end.encode_obs_output_cur
					batch_end.actor_input_next = batch_end.encode_obs_output_next
				else:
					raise ValueError("batch_end error")
				if self.global_cfg.actor_input.obs_encode.pred_loss_weight:
					batch_end.pred_obs_output_cur, _ = self.encode_net.decode(batch_end.encode_obs_output_cur)
				if self.global_cfg.actor_input.obs_encode.before_policy_detach:
					batch_end.actor_input_cur = batch_end.actor_input_cur.detach()
					batch_end.actor_input_next = batch_end.actor_input_next.detach()
			# end
			if "from_target_q" in batch: batch_stack.from_target_q = batch.from_target_q
			if "is_preprocessed" in batch: batch_stack.is_preprocessed = batch.is_preprocessed
			# indices = idx_stack
			batch = batch_stack
			batch.to_torch(device=self.actor.device)
		else: raise ValueError("unknown history_merge_method: {}".format(self.global_cfg.actor_input.history_merge_method))
		# critic input 
		if self.global_cfg.critic_input.history_merge_method == "none":
			actor_result_cur = self.forward(batch, input="actor_input_cur")
			actor_result_next = self.forward(batch, input="actor_input_next")
			batch.critic_input_cur_offline = torch.cat([
				self.get_obs_base(batch, "critic", "cur"),
				batch.act], dim=-1) # (B, T, obs_dim + act_dim) or (B, obs_dim + act_dim)
			batch.critic_input_cur_online = torch.cat([
				self.get_obs_base(batch, "critic", "cur"),
				actor_result_cur.act.reshape(*self.get_obs_base(batch, "critic", "cur").shape[:-1], -1)
				], dim=-1) # (B, T, obs_dim + act_dim) or (B, obs_dim + act_dim)
			batch.critic_input_next_online = torch.cat([
				self.get_obs_base(batch, "critic", "next"),
				actor_result_next.act.reshape(*self.get_obs_base(batch, "critic", "next").shape[:-1], -1)
				], dim=-1).detach()
			batch.log_prob_cur = actor_result_cur.log_prob.reshape(*batch.obs.shape[:-1], -1)
			batch.log_prob_next = actor_result_next.log_prob.reshape(*batch.obs.shape[:-1], -1)
		elif self.global_cfg.critic_input.history_merge_method == "cat_mlp":
			raise NotImplementedError
		elif self.global_cfg.critic_input.history_merge_method == "stack_rnn":
			raise NotImplementedError
		else: raise ValueError("unknown history_merge_method: {}".format(self.global_cfg.critic_input.history_merge_method))
		# return cal
		if self.global_cfg.actor_input.custom_return_cal == True:
			batch.returns = self.compute_return_custom(batch)
		elif self.global_cfg.actor_input.custom_return_cal == False:
			if not hasattr(batch, "from_target_q") or not batch.from_target_q:
				batch = self.compute_nstep_return(
					batch, buffer, indices, self._target_q, self._gamma, self._n_step,
					self._rew_norm
				)
		else: raise ValueError("batch_end error")
		return batch

	def _target_q(self, buffer, indices) -> torch.Tensor:
		batch = buffer[indices]  # batch.obs: s_{t+n}
		batch.from_target_q = True
		batch = self.process_fn(batch, buffer, indices)
		target_q = torch.min(
			self.critic1_old(batch.critic_input_next_online)[0], # (B, 1)
			self.critic2_old(batch.critic_input_next_online)[0],
		) - self._alpha * batch.log_prob_next # (B, a_dim, 1)
		return target_q

	def process_online_batch(self, batch, state):
		obs = batch.obs
		process_online_batch_info = {}
		# if first step when act is none
		assert batch.obs.shape[0] == 1, "for online batch, batch size must be 1"
		if self.global_cfg.actor_input.history_merge_method == "cat_mlp":
			if len(batch.act.shape) == 0: # first step (zero cat)
				if self.global_cfg.actor_input.obs_pred.turn_on:
					new_dim = self.pred_net.input_dim
				elif self.global_cfg.actor_input.obs_encode.turn_on:
					new_dim = self.encode_net.normal_encode_dim
				else:
					new_dim = self.actor.net.input_dim
				obs = np.zeros([obs.shape[0], new_dim])
			else: # normal step
				if (len(batch.info["historical_act"].shape) == 1 and batch.info["historical_act"].shape[0] == 1): # when historical_num == 0, would return False as historical_act
					# ps. historical_act == None when no error
					assert batch.info["historical_act"][0] == False
					obs = np.concatenate([
						self.get_obs_base(batch, "actor", "cur"),
					], axis=-1)
				elif (len(batch.info["historical_act"].shape) == 2 and batch.info["historical_act"].shape[0] == 1):
					obs = np.concatenate([
						self.get_obs_base(batch, "actor", "cur"),
						batch.info["historical_act"] \
						if not self.global_cfg.actor_input.noise_act_debug else \
						np.random.normal(size=batch.info["historical_act"].shape, loc=0, scale=1),
					], axis=-1)
				else: raise ValueError("historical_act shape not implemented")
			if self.global_cfg.actor_input.obs_pred.turn_on:
				pred_output, pred_info = self.pred_net(obs)
				if self.global_cfg.actor_input.obs_pred.input_type == "obs":
					obs = pred_output.cpu()
				elif self.global_cfg.actor_input.obs_pred.input_type == "feat":
					obs = pred_info["feats"].cpu()
				else:
					raise ValueError("unknown input_type: {}".format(self.global_cfg.actor_input.obs_pred.input_type))
				process_online_batch_info["pred_output"] = pred_output
			elif self.global_cfg.actor_input.obs_encode.turn_on:
				encode_output, encode_info = self.encode_net.normal_encode(obs)
				obs = encode_output.cpu()
		elif self.global_cfg.actor_input.history_merge_method == "stack_rnn":
			if len(batch.act.shape) == 0: # first step (zero cat)
				if self.global_cfg.actor_input.obs_pred.turn_on:
					new_dim = self.pred_net.input_dim
				elif self.global_cfg.actor_input.obs_encode.turn_on:
					new_dim = self.encode_net.normal_encode_dim
				else:
					new_dim = self.actor.net.input_dim
				obs = np.zeros([obs.shape[0], new_dim])
			else: # normal step
				if (len(batch.info["historical_act"].shape) == 1 and batch.info["historical_act"].shape[0] == 1): # when historical_num == 0, would return False as historical_act
					assert batch.info["historical_act"][0] == False
					obs = np.concatenate([
						self.get_obs_base(batch, "actor", "next"), # (B, obs_dim) # TODO change network init
					], axis=-1)
				elif (len(batch.info["historical_act"].shape) == 2 and batch.info["historical_act"].shape[0] == 1):
					latest_act = batch.info["historical_act"][:, -self.actor.act_num:]
					obs = np.concatenate([
						self.get_obs_base(batch, "actor", "next"), # (B, obs_dim)
						latest_act # (B, act_num)
					], axis=-1)
				else: raise ValueError("historical_act shape not implemented")
			if self.global_cfg.actor_input.obs_pred.turn_on:
				pred_output, pred_info = self.pred_net(obs, None if state is None else {"hidden": state["hidden_obs_pred_rnn"]}) # ! TODO check should
				process_online_batch_info["hidden_obs_pred_rnn"] = pred_info["state"]["hidden"]
				if self.global_cfg.actor_input.obs_pred.input_type == "obs":
					obs = pred_output.cpu()
				elif self.global_cfg.actor_input.obs_pred.input_type == "feat":
					obs = pred_info["feats"].cpu()
				else:
					raise ValueError("unknown input_type: {}".format(self.global_cfg.actor_input.obs_pred.input_type))
				process_online_batch_info["pred_output"] = pred_output
			elif self.global_cfg.actor_input.obs_encode.turn_on:
				raise NotImplementedError("stack_rnn + obs_encode not implemented")
				encode_output, res_state = self.encode_net.normal_encode(obs)
				obs = encode_output.cpu()
				process_online_batch_info["hidden_obs_encode_rnn"] = res_state["hidden"]
		elif self.global_cfg.actor_input.history_merge_method == "none":
			if len(batch.act.shape) == 0: # first step (zero cat)
				obs = np.zeros([obs.shape[0], self.actor.net.input_dim])
			else: # normal step
				obs = self.get_obs_base(batch, "actor", "next")
		else:
			raise ValueError(f"history_merge_method {self.global_cfg.actor_input.history_merge_method} not implemented")
		batch.online_input = obs
		return batch, process_online_batch_info

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
			returns = batch.rew + self._gamma * (1 - batch.done.int()) * value_next.reshape(*batch.done.shape[:-1],-1)
			return returns

	def forward(  # type: ignore
		self,
		batch: Batch,
		state: Optional[Union[dict, Batch, np.ndarray]] = None,
		input: str = "obs",
		**kwargs: Any,
	) -> Batch:
		###ACTOR_FORWARD note that env.step would call this function automatically
		state_res = {}
		if not hasattr(batch, "is_preprocessed") or not batch.is_preprocessed: # online learn
			### process the input from env (online learn).
			input = "online_input"
			batch, process_online_batch_info = self.process_online_batch(batch, state)
		actor_input = batch[input]
		logits, actor_state_res = self.actor(
			actor_input, 
			state=state if isinstance(state, dict) and "hidden" in state else None,
			info=batch.info)
		if not (actor_state_res is None): state_res["hidden"] = actor_state_res["hidden"]
		assert isinstance(logits, tuple)
		dist = Independent(Normal(*logits), 1)
		if self._deterministic_eval and not self.training: act = logits[0]
		else: act = dist.rsample()
		log_prob = dist.log_prob(act).unsqueeze(-1)
		squashed_action = torch.tanh(act)
		log_prob = log_prob - torch.log((1 - squashed_action.pow(2)) +
										np.finfo(np.float32).eps.item()).sum(-1, keepdim=True)
		# process hidden
		if self.global_cfg.actor_input.obs_pred.turn_on and self.global_cfg.actor_input.history_merge_method == "stack_rnn" and input == "online_input":
			state_res["hidden_obs_pred_rnn"] = process_online_batch_info["hidden_obs_pred_rnn"]
		if self.global_cfg.actor_input.obs_encode.turn_on and self.global_cfg.actor_input.history_merge_method == "stack_rnn" and input == "online_input":
			raise NotImplementedError("stack_rnn + obs_encode not implemented")
			state_res["hidden_encode_pred_rnn"]  = process_online_batch_info["hidden_encode_pred_rnn"]
		### log - train_env_infer
		if input == "online_input" and self.training: 
			if not hasattr(self, "start_time"): self.start_time = time()
			if not hasattr(self, "train_env_infer_step"): self.train_env_infer_step = 0
			self.train_env_infer_step += 1
			if (self.train_env_infer_step % self.global_cfg.log_interval) == 0:
				minutes = (time() - self.start_time) / 60
				hours = minutes / 60
				to_logs = {
					"train_env_infer/left_hr": hours *  ((1e6 - self.train_env_infer_step) / self.train_env_infer_step),
					"train_env_infer/past_hr": hours,
					"train_env_infer/1mStep_hr": hours * (1e6 / self.train_env_infer_step),
					"train_env_infer/step": self.train_env_infer_step,
				}
				if self.global_cfg.actor_input.obs_pred.turn_on and self.global_cfg.actor_input.obs_pred.input_type == "obs":
					with torch.no_grad():
						pred_loss = (batch.online_input - torch.tensor(batch.info["obs_next_nodelay"],device=batch.online_input.device)).pow(2).mean().cpu().item()
						to_logs["train_env_infer/pred_loss"] = pred_loss
						to_logs["train_env_infer/abs_error_pred"] = pred_loss ** 0.5
				wandb.log(to_logs, commit=self.global_cfg.log_instant_commit)
		return Batch(
			logits=logits,
			act=squashed_action,
			state=state_res,
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

	def get_historical_act(self, indices, step, buffer, type=None, device=None):
		""" get historical act
		input [t_0, t_1, ...]
		output [
			[t_0-step, t_0-step+1, ... t_0-1],
			[t_1-step, t_1-step+1, ... t_1-1],
			...
		]
		ps. note that cur step is not included
		ps. the neg step is set to 0.
		:param indices: indices of the batch (B,)
		:param step: the step of the batch. int
		:param buffer: the buffer. 
		:return: historical act (B, step)
		"""
		assert type in ["cat", "stack"], "type must be cat or stack"
		# [t_0-step, t_0-step+1, ... t_0-1, t_0]
		idx_stack_plus1 = utils.idx_stack(indices, buffer, step+1, direction="prev")
		# [t_0-step,   t_0-step+1, ..., t_0-1]
		idx_stack_next = idx_stack_plus1[:, :-1] # (B, step)
		# [t_0-step+1, t_0-step+2, ...,   t_0]
		idx_stack = idx_stack_plus1[:, 1:] # (B, step)
		invalid = (idx_stack_next == idx_stack) # (B, step)
		historical_act = buffer[idx_stack].act # (B, step, act_dim)
		historical_act[invalid] = 0.
		if type == "cat":
			historical_act = historical_act.reshape(historical_act.shape[0], -1) # (B, step*act_dim)
		historical_act = torch.tensor(historical_act, device=device)
		return historical_act





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
		head_num: int,
		dropout: float = None,
	):
		super().__init__()
		self.device = device
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.rnn_layer_num = rnn_layer_num
		self.rnn_hidden_layer_size = rnn_hidden_layer_size
		self.mlp_hidden_sizes = mlp_hidden_sizes
		self.dropout = dropout
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
		assert len(mlp_hidden_sizes) > 0, "mlp_hidden_sizes must be > 0"
		before_head_mlp_hidden_sizes = mlp_hidden_sizes[:-1]
		self.mlp_before_head = []
		self.mlp_before_head.append(MLP(
			rnn_hidden_layer_size if rnn_layer_num else input_dim,
			mlp_hidden_sizes[-1],
			before_head_mlp_hidden_sizes,
			device=self.device,
			activation=nn.ReLU
		))
		if self.dropout:
			self.mlp_before_head.append(nn.Dropout(self.dropout))
		self.heads = []
		for _ in range(head_num):
			head = MLP(
				mlp_hidden_sizes[-1],
				output_dim,
				hidden_sizes=(),
				device=self.device,
				activation=nn.ReLU
			)
			self.heads.append(head.to(self.device))
		self.mlp_before_head = nn.Sequential(*self.mlp_before_head)
	
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
			to_unsqueeze_from_1 = False
			to_unsqueeze = False
			if len(obs.shape) == 1: 
				obs = obs.unsqueeze(0)
				to_unsqueeze_from_1 = True
			assert len(obs.shape) == 3 or len(obs.shape) == 2, "obs.shape: {}".format(obs.shape)
			
			if len(obs.shape) == 2: 
				to_unsqueeze = True
				obs = obs.unsqueeze(-2) # make seq_len dim
			B, L, D = obs.shape
			self.nn.flatten_parameters()
			if state is None or state["hidden"] is None:
				# first step of online or offline
				hidden = torch.zeros(self.rnn_layer_num, B, self.rnn_hidden_layer_size, device=self.device)
				after_rnn, hidden = self.nn(obs, hidden)
			else: 
				# normal step of online
				after_rnn, hidden = self.nn(obs, state["hidden"].transpose(0, 1).contiguous())
			if to_unsqueeze: 
				after_rnn = after_rnn.squeeze(-2)
			if to_unsqueeze_from_1:
				after_rnn = after_rnn.squeeze(0)
		else: # skip rnn
			after_rnn = obs
		### forward mlp
		before_head = self.flatten_foward(self.mlp_before_head, after_rnn)
		### forward head
		outputs = []
		for head in self.heads:
			outputs.append(self.flatten_foward(head, before_head))
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
			if self.hps["global_cfg"].actor_input.obs_pred.turn_on:
				if self.hps["global_cfg"].actor_input.obs_pred.input_type == "feat":
					input_dim = self.hps["global_cfg"].actor_input.obs_pred.feat_dim
				elif self.hps["global_cfg"].actor_input.obs_pred.input_type == "obs":
					input_dim = state_shape[0]
				else:
					raise ValueError("invalid input_type")
			elif self.hps["global_cfg"].actor_input.obs_encode.turn_on:
				input_dim = self.hps["global_cfg"].actor_input.obs_encode.feat_dim
			else:
				input_dim = state_shape[0] + action_shape[0] * self.hps["global_cfg"].actor_input.history_num
			output_dim = int(np.prod(action_shape))
		elif self.hps["global_cfg"].actor_input.history_merge_method == "stack_rnn":
			if self.hps["global_cfg"].actor_input.obs_pred.turn_on:
				if self.hps["global_cfg"].actor_input.obs_pred.input_type == "feat":
					input_dim = self.hps["global_cfg"].actor_input.obs_pred.feat_dim
				elif self.hps["global_cfg"].actor_input.obs_pred.input_type == "obs":
					input_dim = state_shape[0]
				else:
					raise ValueError("invalid input_type")
			elif self.hps["global_cfg"].actor_input.obs_encode.turn_on:
				input_dim = self.hps["global_cfg"].actor_input.obs_encode.feat_dim
			else:
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
		state: Optional[Dict[str, torch.Tensor]],
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
		if self.hps["pure_random"]:
			# mu = mu * 1e-10 + torch.normal(mean=0., std=1., size=mu.shape, device=self.device)
			mu = (torch.rand_like(mu, device=self.device) * 2) - 1
			# sigma = torch.ones_like(sigma, device=self.device) * 1e-10
			sigma = sigma * 1e-10
		return (mu, sigma), state_
	
class ObsPredNet(nn.Module):
	"""
	input delayed state and action, output the non-delayed state
	"""
	def __init__(
		self,
		state_shape: Sequence[int],
		action_shape: Sequence[int],
		global_cfg: Dict[str, Any],
		**kwargs,
	) -> None:
		super().__init__()
		self.global_cfg = global_cfg
		self.hps = kwargs
		# assert head_num == 1, the rnn layer of decoder is 0
		assert self.hps["net_type"] in ["vae", "mlp", "rnn"], "invalid net_type {}".format(self.hps["net_type"])
		if self.hps["net_type"] in ["vae", "mlp"]:
			self.input_dim = state_shape[0] + action_shape[0] * global_cfg.actor_input.history_num
		elif self.hps["net_type"] == "rnn":
			self.input_dim = state_shape[0] + action_shape[0]
		self.output_dim = state_shape[0]
		self.feat_dim = self.hps["feat_dim"]
		self.encoder_input_dim = self.input_dim
		self.encoder_output_dim = self.feat_dim
		self.decoder_input_dim = self.feat_dim
		self.decoder_output_dim = self.output_dim
		if self.hps["net_type"]=="vae":
			self.encoder_net = self.hps["encoder_net"](self.encoder_input_dim, self.encoder_output_dim, device=self.hps["device"], head_num=2)
		elif self.hps["net_type"]=="mlp":
			self.encoder_net = self.hps["encoder_net"](self.encoder_input_dim, self.encoder_output_dim, device=self.hps["device"], head_num=1)
		elif self.hps["net_type"]=="rnn":
			self.encoder_net = self.hps["encoder_net"](self.encoder_input_dim, self.encoder_output_dim, device=self.hps["device"], head_num=1)
		self.decoder_net = self.hps["decoder_net"](self.decoder_input_dim, self.decoder_output_dim, device=self.hps["device"], head_num=1)
		self.encoder_net.to(self.hps["device"])
		self.decoder_net.to(self.hps["device"])
		
	def forward(
		self,
		input,
		state = None,
		info = {},
	) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
		assert type(info) == dict, "info should be a dict, check whether missing 'info' as act"
		info = {}
		encoder_outputs, state_ = self.encoder_net(input)
		if self.hps["net_type"] == "vae":
			mu, logvar = encoder_outputs
			feats = self.vae_sampling(mu, logvar)
			info["mu"] = mu
			info["logvar"] = logvar
		elif self.hps["net_type"] == "mlp":
			feats = encoder_outputs[0]
		elif self.hps["net_type"] == "rnn":
			feats = encoder_outputs[0]
		output, _ = self.decoder_net(feats, state)
		output = output[0]
		info["state"] = state_
		info["feats"] = feats
		return output, info

	def vae_sampling(self, mu, log_var):
		std = torch.exp(0.5*log_var)
		eps = torch.randn_like(std)
		return eps.mul(std).add_(mu)

class ObsEncodeNet(nn.Module):
	"""
	input delayed state and action, output the non-delayed state
	"""
	def __init__(
		self,
		state_shape: Sequence[int],
		action_shape: Sequence[int],
		global_cfg: Dict[str, Any],
		**kwargs,
	) -> None:
		super().__init__()
		self.global_cfg = global_cfg
		self.hps = kwargs
		self.normal_encode_dim = state_shape[0] + action_shape[0] * global_cfg.actor_input.history_num
		self.oracle_encode_dim = state_shape[0]
		self.feat_dim = self.hps["feat_dim"]
		self.normal_encoder_net = self.hps["encoder_net"](self.normal_encode_dim, self.feat_dim, device=self.hps["device"], head_num=2)
		self.oracle_encoder_net = self.hps["encoder_net"](self.oracle_encode_dim, self.feat_dim, device=self.hps["device"], head_num=2)
		self.decoder_net = self.hps["decoder_net"](self.feat_dim, self.oracle_encode_dim, device=self.hps["device"], head_num=1)
		self.normal_encoder_net.to(self.hps["device"])
		self.oracle_encoder_net.to(self.hps["device"])
		self.decoder_net.to(self.hps["device"])
		
	def forward(
		self,
		input: Union[np.ndarray, torch.Tensor],
		info: Dict[str, Any] = {},
	) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
		raise ValueError("should call normal_encode or oracle_encode")

	def normal_encode(
			self,
			input: Union[np.ndarray, torch.Tensor],
			info: Dict[str, Any] = {},
		):
		return self.net_forward(self.normal_encoder_net, input, info)
	
	def oracle_encode(
			self,
			input: Union[np.ndarray, torch.Tensor],
			info: Dict[str, Any] = {},
		):
		return self.net_forward(self.oracle_encoder_net, input, info)
	
	def net_forward(self, net, input, info):
		info = {}
		encoder_outputs, state_ = net(input)
		mu, logvar = encoder_outputs
		feats = self.vae_sampling(mu, logvar)
		info["mu"] = mu
		info["logvar"] = logvar
		info["state"] = state_
		info["feats"] = feats
		return feats, info

	def vae_sampling(self, mu, log_var):
		std = torch.exp(0.5*log_var)
		eps = torch.randn_like(std)
		return eps.mul(std).add_(mu)

	def decode(
			self,
			input: Union[np.ndarray, torch.Tensor],
			info: Dict[str, Any] = {},
		):
		"""
		ps. there is only one type of decoder since it is always from latent dim to the oracle obs
		"""
		info = {}
		encoder_outputs, state_ = self.decoder_net(input)
		res = encoder_outputs[0]
		return res, info

class TD3Policy(DDPGPolicy):
	"""Implementation of TD3, arXiv:1802.09477.

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
	:param float exploration_noise: the exploration noise, add to the action.
		Default to ``GaussianNoise(sigma=0.1)``
	:param float policy_noise: the noise used in updating policy network.
		Default to 0.2.
	:param int update_actor_freq: the update frequency of actor network.
		Default to 2.
	:param float noise_clip: the clipping range used in updating policy network.
		Default to 0.5.
	:param bool reward_normalization: normalize the reward to Normal(0, 1).
		Default to False.
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
		tau: float = 0.005,
		gamma: float = 0.99,
		exploration_noise=None,
		policy_noise: float = 0.2,
		update_actor_freq: int = 2,
		noise_clip: float = 0.5,
		reward_normalization: bool = False,
		estimation_step: int = 1,
		**kwargs: Any,
	) -> None:
		super().__init__(
			actor, actor_optim, None, None, tau, gamma, exploration_noise,
			reward_normalization, estimation_step, **kwargs
		)
		self.critic1, self.critic1_old = critic1, deepcopy(critic1)
		self.critic1_old.eval()
		self.critic1_optim = critic1_optim
		self.critic2, self.critic2_old = critic2, deepcopy(critic2)
		self.critic2_old.eval()
		self.critic2_optim = critic2_optim
		self._policy_noise = policy_noise
		self._freq = update_actor_freq
		self._noise_clip = noise_clip
		self._cnt = 0
		self._last = 0

	def train(self, mode: bool = True) -> "TD3Policy":
		self.training = mode
		self.actor.train(mode)
		self.critic1.train(mode)
		self.critic2.train(mode)
		return self

	def sync_weight(self) -> None:
		self.soft_update(self.critic1_old, self.critic1, self.tau)
		self.soft_update(self.critic2_old, self.critic2, self.tau)
		self.soft_update(self.actor_old, self.actor, self.tau)

	def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
		batch = buffer[indices]  # batch.obs: s_{t+n}
		act_ = self(batch, model="actor_old", input="obs_next").act
		noise = torch.randn(size=act_.shape, device=act_.device) * self._policy_noise
		if self._noise_clip > 0.0:
			noise = noise.clamp(-self._noise_clip, self._noise_clip)
		act_ += noise
		target_q = torch.min(
			self.critic1_old(batch.obs_next, act_),
			self.critic2_old(batch.obs_next, act_),
		)
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
		if self._cnt % self._freq == 0:
			actor_loss = -self.critic1(batch.obs, self(batch, eps=0.0).act).mean()
			self.actor_optim.zero_grad()
			actor_loss.backward()
			self._last = actor_loss.item()
			self.actor_optim.step()
			self.sync_weight()
		self._cnt += 1
		return {
			"loss/actor": self._last,
			"loss/critic1": critic1_loss.item(),
			"loss/critic2": critic2_loss.item(),
		}

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
	
class TianshouTD3Wrapper(TD3Policy):
	def __init__(self, *args, **kwargs):
		self.global_cfg = kwargs.pop("global_cfg")
		self.state_space = kwargs.pop("state_space")
		super().__init__(*args, **kwargs)

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
		self.cfg = cfg
		self.env = cfg.env
		# init
		seed = int(time()) if cfg.seed is None else cfg.seed
		utils.seed_everything(seed) # TODO add env seed
		self.train_envs = tianshou.env.DummyVectorEnv([partial(utils.make_env, cfg.env) for _ in range(cfg.env.train_num)])
		self.test_envs = tianshou.env.DummyVectorEnv([partial(utils.make_env, cfg.env) for _ in range(cfg.env.test_num)])
		self.env = utils.make_env(cfg.env) # to get obs_space and act_space
		self.console = Console()
		self.log = self.console.log
		self.log("RLRunner end!")


class SACRunner(DefaultRLRunner):
	def start(self, cfg):
		self.log("SACRunner init start ...")
		# TODO add cfg check here e.g. global_cfg == rnn, rnn_layer > 0
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
			state_space=env.observation_space,
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
				"eval/reward": epoch_stat["test_reward"],
			}
			to_log.update(epoch_stat)
			to_log.update(info)
			wandb.log(to_log)
		wandb.finish()
		self.log("SACRunner init end!")


class EnvCollector:
	"""
	Use policy to collect data from env.
	This collector will continue from the last state of the env.
	"""

	def __init__(self, env):
		self.env = env
		self.env_loop = self.create_env_loop()

	def collect(self, act_func, n_step=None, n_episode=None, env_max_step=5000, reset=False, progress_bar=None, rich_progress=None):
		"""
		Return 
			res: a list of Batch(obs, act, rew, done, obs_next, info).
			# info: {
			# 	"rwds": list of total rewards of each episode,
			# }
		Policy can be function or string "random". 
			function: 
				input: batch, state output: a, state
				state is {"hidden": xxx, "hidden_pre": xxx}
		n_step and n_episode should be provided one and only one.
		Will continue from the last state of the env if reset=False.
		"""
		assert isinstance(act_func, (str, Callable)), "act_func should be a function or string 'random'"
		assert (n_step is None) ^ (n_episode is None), "n_step and n_episode should be provided one and only one"
		if progress_bar is not None: assert rich_progress is not None, "rich process must be provided to display process bar"

		if reset == True: self.to_reset = True
		self.act_func = act_func
		self.env_max_step = env_max_step
		res_list = []
		res_info = {
			"rwd_sum_list": [],
			"ep_len_list": []
		}

		step_cnt = 0
		episode_cnt = 0
		finish_flag = False
		if progress_bar is not None:
			progress = rich_progress
			task = progress.add_task(progress_bar, total=n_episode if n_episode is not None else n_step)
		while not finish_flag:
			batch, env_loop_info = next(self.env_loop)
			res_list.append(batch)

			if (batch.terminated or batch.truncated).any(): 
				episode_cnt += 1
				if progress_bar is not None: progress.update(task, advance=1)
				res_info["rwd_sum_list"].append(env_loop_info["rwd_sum"])
				res_info["ep_len_list"].append(env_loop_info["ep_len"])

			if n_step is not None:
				step_cnt += 1
				if progress_bar is not None: progress.update(task, advance=1)
				finish_flag = step_cnt >= n_step
			elif n_episode is not None:
				finish_flag = episode_cnt >= n_episode
		if progress_bar is not None: progress.remove_task(task)
		return res_list, res_info

	def create_env_loop(self):
		"""
		Infinite loop, yield a Batch(obs, act, rew, done, obs_next, info).
		Will restart from 0 and return Batch(s_0, ...) if self.to_reset = True.
		"""
		while True:
			env_step_cur = 0
			rwd_sum_cur = 0.
			s, info = self.env.reset(), {"is_first_step": True}
			last_state = None

			while True:
				a, last_state = self._select_action(Batch(obs=s, info=info), last_state)
				# if a is tensor, turn to numpy array
				if isinstance(a, torch.Tensor):
					a = a.detach()
					if a.device != torch.device("cpu"):
						a = a.cpu()
					a = a.numpy()
				s_, r, terminated, info = self.env.step(a)
				rwd_sum_cur += r

				truncated = env_step_cur == self.env_max_step
				batch = Batch(obs=s, act=a, rew=r, terminated=terminated, truncated=truncated, obs_next=s_, info=info)
				
				yield batch, {
					"rwd_sum": rwd_sum_cur,
					"ep_len": env_step_cur,
				}

				if self.to_reset:
					self.to_reset = False
					break

				if terminated or truncated:
					break

				env_step_cur += 1
				s = s_

	def _select_action(self, s, state):
		if self.act_func == "random":
			return self.env.action_space.sample(), None
		else:
			return self.act_func(s, state)

	def reset(self):
		self.to_reset = True

class WaybabaRecorder:
	"""
	store all digit values during training and render in different ways.
	self.data = {
		"name": {
			"value": [],
			"show_in_progress_bar": True,
			"upload_to_wandb": False,
			"wandb_logged": False,
		},
		...
	}
	"""
	def __init__(self):
		self.data = {}
	
	def __call__(self, k, v, wandb_=None, progress_bar=None):
		"""
		would update upload_to_wandb and show_in_progress_bar if provided.
		"""
		if k not in self.data: self.data[k] = self._create_new()
		self.data[k]["values"].append(v)
		if progress_bar in [True, False]: self.data[k]["show_in_progress_bar"] = progress_bar
		if wandb_ in [True, False]:  self.data[k]["upload_to_wandb"] = wandb_
		self.data[k]["wandb_logged"] = False

	def upload_to_wandb(self, *args, **kwargs):
		to_upload = {}
		for k, v in self.data.items():
			if v["upload_to_wandb"] and not v["wandb_logged"] and len(v["values"]) > 0:
				to_upload[k] = v["values"][-1]
				self.data[k]["wandb_logged"] = True
		if len(to_upload) > 0:
			wandb.log(to_upload, *args, **kwargs)

	def to_progress_bar_description(self):
		return self.__str__()

	def _create_new(self):
		return {
			"values": [],
			"show_in_progress_bar": True,
			"upload_to_wandb": True,
			"wandb_logged": False,
		}
	
	def __str__(self):
		info_dict = {
			k: v["values"][-1] for k, v in \
			sorted(self.data.items(), key=lambda item: item[0]) \
			if v["show_in_progress_bar"] and len(v["values"]) > 0
		}
		for k, v in info_dict.items():
			if type(v) == int:
				info_dict[k] = str(v)
			elif type(v) == float:
				info_dict[k] = '{:.2f}'.format(v)
			else:
				info_dict[k] = '{:.2f}'.format(v)

		# Find the maximum length of keys and values
		max_key_length = max(len(k) for k in info_dict.keys())
		max_value_length = max(len(v) for v in info_dict.values())

		# Align keys to the left and values to the right
		aligned_info = []
		for k, v in info_dict.items():
			left_aligned_key = k.ljust(max_key_length)
			right_aligned_value = v.rjust(max_value_length)
			aligned_info.append(f"{left_aligned_key} {right_aligned_value}")

		return "\n".join(aligned_info)
	

class OfflineRLRunner(DefaultRLRunner):
	def start(self, cfg):
		super().start(cfg)

		self.log("Init Components ...")
		self.init_components()

		self.log("Initial Exploration ...")
		self._initial_exploration()

		self.log("Training Start ...")
		self.env_step_global = 0
		if cfg.trainer.progress_bar: self.training_task = self.progress.add_task("[green]Training...", total=cfg.trainer.max_epoch*cfg.trainer.step_per_epoch)
		
		while True: # traininng loop
			# env step collect
			self._collect_once()
			
			# update
			if self._should_update(): 
				for _ in range(int(cfg.trainer.step_per_collect/cfg.trainer.update_per_step)):
					batch, indices = self.buf.sample(self.cfg.trainer.batch_size) # sample
					self.update_once(batch, indices)
			
			# evaluate
			if self._should_evaluate():
				self._evaluate()
			
			# log
			if self._should_log():
				self.record.upload_to_wandb(step=self.env_step_global, commit=False)
			
			# upload
			if self._should_upload():
				wandb.log({}, commit=True)
				
			# loop control
			if self._should_end(): break
			if cfg.trainer.progress_bar: self.progress.update(self.training_task, advance=cfg.trainer.step_per_collect, description=f"[green]🚀 Training {self.env_step_global}/{self.cfg.trainer.max_epoch*self.cfg.trainer.step_per_epoch}[/green]\n"+self.record.to_progress_bar_description())
			self.env_step_global += self.cfg.trainer.step_per_collect

		self._end_all()

	def init_components(self):
		cfg = self.cfg
		env = self.env
		self.global_cfg = cfg.global_cfg
		self.actor = cfg.actor(state_shape=env.observation_space.shape, action_shape=env.action_space.shape, max_action=env.action_space.high[0]).to(cfg.device)
		self.actor_optim = cfg.actor_optim(self.actor.parameters())
		self.critic1 = cfg.critic1(env.observation_space.shape, action_shape=env.action_space.shape).to(cfg.device)
		self.critic1_optim = cfg.critic1_optim(self.critic1.parameters())
		self.critic2 = cfg.critic2(env.observation_space.shape, action_shape=env.action_space.shape).to(cfg.device)
		self.critic2_optim = cfg.critic2_optim(self.critic2.parameters())
		self.actor_old = deepcopy(self.actor)
		self.critic1_old = deepcopy(self.critic1)
		self.critic2_old = deepcopy(self.critic2)
		self.critic1.train()
		self.critic2.train()
		self.critic1_old.train()
		self.critic2_old.train()
		self.actor_old.eval()
		self.buf = cfg.buffer
		self.train_collector = EnvCollector(env)
		self.test_collector = EnvCollector(env)
		self.exploration_noise = cfg.policy.initial_exploration_noise
		self.record = WaybabaRecorder()
		if self.cfg.trainer.progress_bar:
			self.progress = Progress()
			self.progress.start()
		self._noise = self.cfg.policy.exploration_noise
		self._noise_clip = self.cfg.policy.noise_clip

	def _initial_exploration(self):
		"""exploration before training and add to self.buf"""
		initial_batches, info_ = self.train_collector.collect(
			act_func="random", n_step=self.cfg.start_timesteps, reset=True, 
			progress_bar="Initial Exploration ..." if self.cfg.trainer.progress_bar else None,
			rich_progress=self.progress if self.cfg.trainer.progress_bar else None
		)
		self.train_collector.reset()
		for batch in initial_batches: self.buf.add(batch)

	def _collect_once(self):
		"""collect data and add to self.buf"""

		batches, info_ = self.train_collector.collect(
			act_func=partial(self.select_act_for_env, mode="train"), 
			n_step=self.cfg.trainer.step_per_collect, reset=False
		)
		for batch in batches: self.buf.add(batch)

		# store history
		if not hasattr(self, "rwd_sum_history"): self.rwd_sum_history = []
		if not hasattr(self, "ep_len_history"): self.ep_len_history = []
		self.rwd_sum_history += info_["rwd_sum_list"]
		self.ep_len_history += info_["ep_len_list"]
		res_info = {
			"batches": batches,
			**info_
		}
		self._on_collect_end(**res_info)

	def _on_collect_end(self, **kwargs):
		"""called after a step of data collection"""
		self.on_collect_end(**kwargs)
	
	def update_once(self, batch, indices):
		raise NotImplementedError

	def _evaluate(self):
		"""Evaluate the performance of an agent in an environment.
		Args:
			env: Environment to evaluate on.
			act_func: Action selection function. It should take a single argument
				(observation) and return a single action.
		Returns:
			Episode reward.
		"""
		if not hasattr(self, "epoch_cnt"): self.epoch_cnt = 0
		eval_batches, info_ = self.test_collector.collect(
			act_func=partial(self.select_act_for_env, mode="eval"), 
			n_episode=self.cfg.trainer.episode_per_test, reset=True,
			progress_bar="Evaluating ..." if self.cfg.trainer.progress_bar else None,
			rich_progress=self.progress if self.cfg.trainer.progress_bar else None,
		)
		eval_rwds = [0. for _ in range(self.cfg.trainer.episode_per_test)]
		eval_lens = [0 for _ in range(self.cfg.trainer.episode_per_test)]
		cur_ep = 0
		for i, batch in enumerate(eval_batches): 
			eval_rwds[cur_ep] += batch.rew
			eval_lens[cur_ep] += 1
			if batch.terminated or batch.truncated:
				cur_ep += 1
		self.epoch_cnt += 1
		res_info = {
			"rwd_mean": np.mean(eval_rwds),
			"len_mean": np.mean(eval_lens)
		}
		self._on_evaluate_end(**res_info)
		return res_info
	
	def _on_evaluate_end(self, **kwargs):
		to_print = self.record.__str__().replace("\n", "  ")
		to_print = "[Epoch {: 5d}/{}] ### ".format(self.epoch_cnt-1, self.cfg.trainer.max_epoch) + to_print
		if not self.cfg.trainer.hide_eval_info_print:
			print(to_print)
		self.on_evaluate_end(**kwargs)
	
	def on_evaluate_end(self, **kwargs):
		pass

	def _end_all(self):
		if self.cfg.trainer.progress_bar: self.progress.stop()
		wandb.finish()

	def select_act_for_env(self, obs, info=None, state=None, mode=None):
		raise NotImplementedError

	def _should_update(self):
		# TODO since we collect x steps, so we always update
		# if not hasattr(self, "should_update_record"): self.should_update_record = {}
		# cur_update_tick = self.env_step_global // self.cfg.trainer.
		# if cur_update_tick not in self.should_update_record:
		# 	self.should_update_record[cur_update_tick] = True
		# 	return True
		# return False
		return True
	
	def _should_evaluate(self):
		if not hasattr(self, "should_evaluate_record"): self.should_evaluate_record = {}
		cur_evaluate_tick = self.env_step_global // self.cfg.trainer.step_per_epoch
		if cur_evaluate_tick not in self.should_evaluate_record:
			self.should_evaluate_record[cur_evaluate_tick] = True
			return True
		return False
	
	def _should_log(self):
		if not hasattr(self, "should_log_record"): self.should_log_record = {}
		cur_log_tick = self.env_step_global // self.cfg.trainer.log_interval
		if cur_log_tick not in self.should_log_record:
			self.should_log_record[cur_log_tick] = True
			return True
		return False

	def _should_upload(self):
		if not self.cfg.trainer.log_upload_interval: return True # missing or zero, always upload
		if not hasattr(self, "should_upload_record"): self.should_upload_record = {}
		cur_upload_tick = self.env_step_global // self.cfg.trainer.log_upload_interval
		if cur_upload_tick not in self.should_upload_record:
			self.should_upload_record[cur_upload_tick] = True
			return True
		return False

	def _should_end(self):
		return self.env_step_global >= self.cfg.trainer.max_epoch * self.cfg.trainer.step_per_epoch


class TD3Runner(OfflineRLRunner):

	def select_act_for_env(self, batch, state, mode=None):
		a_in = batch.obs
		process_online_batch_info = {}
		# if first step when act is none
		assert len(batch.obs.shape) == 1, "for online batch, batch size must be 1"
		if self.global_cfg.actor_input.history_merge_method == "none":
			if self.global_cfg.actor_input.obs_type == "normal":
				a_in = batch.obs
			elif self.global_cfg.actor_input.obs_type == "oracle":
				a_in = batch.info["obs_next_nodelay"]
		elif self.global_cfg.actor_input.history_merge_method == "cat_mlp":
			if "is_first_step" in batch.info: # first step (zero cat)
				if self.global_cfg.actor_input.obs_pred.turn_on:
					new_dim = self.pred_net.input_dim
				elif self.global_cfg.actor_input.obs_encode.turn_on:
					new_dim = self.encode_net.normal_encode_dim
				else:
					new_dim = self.actor.net.input_dim
				a_in = np.zeros([new_dim])
			else: # normal step
				if self.global_cfg.actor_input.obs_type == "normal": a_in = batch.obs
				elif self.global_cfg.actor_input.obs_type == "oracle": a_in = batch.info["obs_next_nodelay"]
				if self.global_cfg.actor_input.history_num > 0:
					a_in = np.concatenate([
						a_in,
						batch.info["historical_act"] \
						if not self.global_cfg.actor_input.noise_act_debug else \
						np.random.normal(size=batch.info["historical_act"].shape, loc=0, scale=1),
					], axis=-1)
			if self.global_cfg.actor_input.obs_pred.turn_on:
				pred_output, pred_info = self.pred_net(a_in)
				if self.global_cfg.actor_input.obs_pred.input_type == "obs":
					a_in = pred_output.cpu()
				elif self.global_cfg.actor_input.obs_pred.input_type == "feat":
					a_in = pred_info["feats"].cpu()
				else:
					raise ValueError("unknown input_type: {}".format(self.global_cfg.actor_input.obs_pred.input_type))
				process_online_batch_info["pred_output"] = pred_output
			elif self.global_cfg.actor_input.obs_encode.turn_on:
				encode_output, encode_info = self.encode_net.normal_encode(a_in)
				a_in = encode_output.cpu()
		elif self.global_cfg.actor_input.history_merge_method == "stack_rnn":
			if "is_first_step" in batch.info:
				if self.global_cfg.actor_input.obs_pred.turn_on:
					raise NotImplementedError
					new_dim = self.pred_net.input_dim
				elif self.global_cfg.actor_input.obs_encode.turn_on:
					raise NotImplementedError
					new_dim = self.encode_net.normal_encode_dim
				else:
					new_dim = self.actor.net.input_dim
				a_in = np.zeros([new_dim])
			else: # normal step
				assert "historical_act" in batch.info, "must have historical act"
				assert self.global_cfg.actor_input.history_num > 0, "stack rnn must len > 0"
				if self.global_cfg.actor_input.obs_type == "normal": a_in = batch.obs
				elif self.global_cfg.actor_input.obs_type == "oracle": a_in = batch.info["obs_next_nodelay"]
				latest_act = batch.info["historical_act"][-self.actor.act_num:]
				a_in = np.concatenate([a_in, latest_act], axis=-1)
			if self.global_cfg.actor_input.obs_pred.turn_on:
				pred_output, pred_info = self.pred_net(a_in, None if state is None else {"hidden": state["hidden_obs_pred_rnn"]}) # ! TODO check should
				process_online_batch_info["hidden_obs_pred_rnn"] = pred_info["state"]["hidden"]
				if self.global_cfg.actor_input.obs_pred.input_type == "obs":
					a_in = pred_output.cpu()
				elif self.global_cfg.actor_input.obs_pred.input_type == "feat":
					a_in = pred_info["feats"].cpu()
				else:
					raise ValueError("unknown input_type: {}".format(self.global_cfg.actor_input.obs_pred.input_type))
				process_online_batch_info["pred_output"] = pred_output
			elif self.global_cfg.actor_input.obs_encode.turn_on:
				raise NotImplementedError("stack_rnn + obs_encode not implemented")
				encode_output, res_state = self.encode_net.normal_encode(a_in)
				a_in = encode_output.cpu()
				process_online_batch_info["hidden_obs_encode_rnn"] = res_state["hidden"]
		else:
			raise ValueError(f"history_merge_method {self.global_cfg.actor_input.history_merge_method} not implemented")
		

		if not isinstance(a_in, torch.Tensor):
			a_in = torch.tensor(a_in, dtype=torch.float32).to(self.cfg.device)
		
		# train eval diff
		a_ori, res_state = self.actor(a_in, state)
		a_ori = a_ori[0]
		if mode == "train":
			noise = torch.tensor(self._noise(a_ori.shape), device=self.cfg.device)
			if self.cfg.policy.noise_clip > 0.0:
				noise = noise.clamp(-self.cfg.policy.noise_clip, self.cfg.policy.noise_clip)
			res = a_ori + noise
		elif mode == "eval":
			res = self.actor(a_in, state)[0][0]
		else:
			raise NotImplementedError
		return res, res_state

	def on_collect_end(self, **kwargs):
		"""called after a step of data collection"""
		if "rwd_sum_list" in kwargs and kwargs["rwd_sum_list"]:
			for i in range(len(kwargs["rwd_sum_list"])): self.record("collect/rwd_sum", kwargs["rwd_sum_list"][i])
		if "ep_len_list" in kwargs and kwargs["ep_len_list"]:
			for i in range(len(kwargs["ep_len_list"])): self.record("collect/ep_len", kwargs["ep_len_list"][i])

	def pre_update_process(self, batch, indices):
		batch.to_torch(device=self.cfg.device, dtype=torch.float32)
		# obs_nodelay # TODO ! bug fisrt step should use original
		batch.obs_nodelay = self.buf[self.buf.prev(indices)].info["obs_next_nodelay"]
		batch.obs_nodelay = torch.tensor(batch.obs_nodelay, device=self.cfg.device)
		batch.obs_next_nodelay = batch.info["obs_next_nodelay"]
		# actor input
		if self.global_cfg.actor_input.history_merge_method == "none":
			batch.a_in_cur = self.get_obs_base(batch, "actor", "cur")
			batch.a_in_next = self.get_obs_base(batch, "actor", "next")
			batch.valid_mask = torch.ones([len(batch)], device=self.actor.device).int()
		elif self.global_cfg.actor_input.history_merge_method == "cat_mlp":
			buffer = self.buf
			assert self.global_cfg.actor_input.history_num >= 0
			if self.global_cfg.actor_input.history_num > 0:
				idx_stack = utils.idx_stack(indices, buffer, self.global_cfg.actor_input.history_num, direction=self.global_cfg.actor_input.trace_direction) # (B, T)
				# del indices
				idx_end = idx_stack[:,-1] # (B, )
				batch_end = buffer[idx_end] # (B, *)
				batch_end.info["obs_nodelay"] = buffer[buffer.prev(idx_end)].info["obs_next_nodelay"] # (B, T, *)
				batch_end.a_in_cur = torch.cat([
					torch.tensor(self.get_obs_base(batch_end, "actor", "cur"),device=self.actor.device), # (B, T, *)
					self.get_historical_act(idx_end, self.global_cfg.actor_input.history_num, buffer, "cat", self.actor.device) \
					if not self.global_cfg.actor_input.noise_act_debug else \
					torch.normal(size=stacked_batch_prev["act"].reshape(batch_end.obs.shape[0],-1).shape, mean=0., std=1.,device=self.actor.device),
				], dim=-1) # (B, T, obs_dim + act_dim * history_num)
				batch_end.a_in_next = torch.cat([
					torch.tensor(self.get_obs_base(batch_end, "actor", "next"),device=self.actor.device), # (B, T, *)
					self.get_historical_act(buffer.next(idx_end), self.global_cfg.actor_input.history_num, buffer, "cat", self.actor.device) \
					if not self.global_cfg.actor_input.noise_act_debug else \
					torch.normal(size=stacked_batch_cur["act"].reshape(len(batch_end),-1).shape, mean=0., std=1.,device=self.actor.device),
				], dim=-1) # (B, T, obs_dim + act_dim * history_num)
				# make mask
				if self.global_cfg.actor_input.trace_direction == "next":
					# all that reach end of episode should be invalid
					batch_end.valid_mask = torch.tensor(idx_end != buffer.next(idx_end), device=self.actor.device).int() # (B, T)
				elif self.global_cfg.actor_input.trace_direction == "prev":
					# all are valid while before the first action should be 0 filled
					batch_end.valid_mask = torch.ones(idx_end.shape, device=self.actor.device).int() # (B, T)
				else: raise ValueError("trace_direction should be next or prev")
				# obs_pred & obs_encode
				if self.global_cfg.actor_input.obs_pred.turn_on:
					batch_end.pred_input_cur = batch_end.a_in_cur
					batch_end.pred_input_next = batch_end.a_in_next
					batch_end.pred_output_cur, pred_info_cur = self.pred_net(batch_end.pred_input_cur)
					batch_end.pred_output_next, pred_info_next = self.pred_net(batch_end.pred_input_next)
					if self.global_cfg.actor_input.obs_pred.input_type == "obs":
						batch_end.a_in_cur = batch_end.pred_output_cur
						batch_end.a_in_next = batch_end.pred_output_next
					elif self.global_cfg.actor_input.obs_pred.input_type == "feat":
						batch_end.a_in_cur = pred_info_cur["feats"]
						batch_end.a_in_next = pred_info_next["feats"]
					else:
						raise NotImplementedError
					# detach
					if self.global_cfg.actor_input.obs_pred.middle_detach: 
						batch_end.a_in_cur = batch_end.a_in_cur.detach()
						batch_end.a_in_next = batch_end.a_in_next.detach()
					if self.global_cfg.actor_input.obs_pred.net_type == "vae":
						batch_end.pred_info_cur_mu = pred_info_cur["mu"]
						batch_end.pred_info_cur_logvar = pred_info_cur["logvar"]
				if self.global_cfg.actor_input.obs_encode.turn_on:
					batch_end.encode_obs_input_cur = batch_end.a_in_cur
					batch_end.encode_obs_input_next = batch_end.a_in_next
					batch_end.encode_obs_output_cur, encode_obs_info_cur = self.encode_net.normal_encode(batch_end.encode_obs_input_cur)
					batch_end.encode_obs_output_next, encode_obs_info_next = self.encode_net.normal_encode(batch_end.encode_obs_input_next)
					batch_end.encode_oracle_obs_output_cur, encode_oracle_obs_info_cur = self.encode_net.oracle_encode(batch_end.info["obs_nodelay"])
					batch_end.encode_oracle_obs_output_next, encode_oracle_obs_info_next = self.encode_net.oracle_encode(batch_end.info["obs_next_nodelay"])
					batch_end.encode_normal_info_cur_mu = encode_obs_info_cur["mu"]
					batch_end.encode_normal_info_cur_logvar = encode_obs_info_cur["logvar"]
					batch_end.encode_oracle_info_cur_mu = encode_oracle_obs_info_cur["mu"]
					batch_end.encode_oracle_info_cur_logvar = encode_oracle_obs_info_cur["logvar"]
					if self.global_cfg.actor_input.obs_encode.train_eval_async == True:
						batch_end.a_in_cur = batch_end.encode_oracle_obs_output_cur
						batch_end.a_in_next = batch_end.encode_oracle_obs_output_next
					elif self.global_cfg.actor_input.obs_encode.train_eval_async == False:
						batch_end.a_in_cur = batch_end.encode_obs_output_cur
						batch_end.a_in_next = batch_end.encode_obs_output_next
					else:
						raise ValueError("batch_end error")
					if self.global_cfg.actor_input.obs_encode.pred_loss_weight:
						batch_end.pred_obs_output_cur, _ = self.encode_net.decode(batch_end.encode_obs_output_cur)
					if self.global_cfg.actor_input.obs_encode.before_policy_detach:
						batch_end.a_in_cur = batch_end.a_in_cur.detach()
						batch_end.a_in_next = batch_end.a_in_next.detach()
				# end
				indices = idx_end
				batch = batch_end
				batch.obs_nodelay = self.buf[self.buf.prev(indices)].info["obs_next_nodelay"]
				batch.obs_nodelay = torch.tensor(batch.obs_nodelay, device=self.cfg.device)
				batch.obs_next_nodelay = batch.info["obs_next_nodelay"]
				batch.to_torch(device=self.cfg.device, dtype=torch.float32)
			else:
				batch.a_in_cur = self.get_obs_base(batch, "actor", "cur")
				batch.a_in_next = self.get_obs_base(batch, "actor", "next")
				batch.valid_mask = torch.ones([len(batch)], device=self.actor.device).int()
		elif self.global_cfg.actor_input.history_merge_method == "stack_rnn":
			buffer = self.buf
			assert self.global_cfg.actor_input.history_num > 1, "stack_rnn requires history_num > 1, ususally, it would be 20,40,... since we process long history when running online."
			assert self.global_cfg.actor_input.history_num > self.global_cfg.actor_input.burnin_num, "stack_rnn requires history_num > burnin_num, ususally, it could be a little larger than burnin_num"
			idx_stack = utils.idx_stack(indices, buffer, self.global_cfg.actor_input.history_num, direction=self.global_cfg.actor_input.trace_direction) # (B, T)
			del indices
			idx_end = idx_stack[:,-1] # (B, )
			batch_end = buffer[idx_end] # (B, *)
			batch_stack = buffer[idx_stack] # (B, T, *)
			batch_end.info["obs_nodelay"] = buffer[buffer.prev(idx_end)].info["obs_next_nodelay"] # (B, *)
			batch_stack.info["obs_nodelay"] = buffer[buffer.prev(idx_stack)].info["obs_next_nodelay"] # (B, T, *)
			batch_stack.a_in_cur = torch.cat([
				torch.tensor(self.get_obs_base(buffer[idx_stack], "actor", "cur"),device=self.actor.device), # (B, T, obs_dim) # (B, T, act_dim)
				self.get_historical_act(idx_end, self.global_cfg.actor_input.history_num, buffer, "stack", self.actor.device) \
				if not self.global_cfg.actor_input.noise_act_debug else \
				torch.normal(size=stacked_batch_prev["act"].reshape(batch_end.obs.shape[0],-1).shape, mean=0., std=1.,device=self.actor.device), # TODO
			], dim=-1) # (B, T, obs_dim+act_dim)
			batch_stack.a_in_next = torch.cat([
				torch.tensor(self.get_obs_base(buffer[idx_stack], "actor", "next"),device=self.actor.device), # (B, T, obs_dim) # (B, T, act_dim)
				self.get_historical_act(buffer.next(idx_end), self.global_cfg.actor_input.history_num, buffer, "stack", self.actor.device) \
				if not self.global_cfg.actor_input.noise_act_debug else \
				torch.normal(size=stacked_batch_cur["act"].reshape(len(batch_end),-1).shape, mean=0., std=1.,device=self.actor.device), # TODO
			], dim=-1) # (B, T, obs_dim+act_dim)
			# make mask
			# if self.global_cfg.actor_input.trace_direction == "next":
			# 	# end step is invalid
			# 	batch_stack.valid_mask = torch.tensor(idx_stack != buffer.next(idx_stack), device=self.actor.device).int() # (B, T)
			# elif self.global_cfg.actor_input.trace_direction == "prev":
			# 	# start step is invalid
			# 	batch_stack.valid_mask = torch.tensor(idx_stack != buffer.prev(idx_stack), device=self.actor.device).int() # (B, T)
			# else: raise ValueError("trace_direction should be next or prev")

			# if the start of the idx reach start or end of the idx reach end, then the whole episode is invalid
			# idx_stack: B, T
			batch_stack.valid_mask = np.ones_like(idx_stack) # (B, T)
			if self.global_cfg.actor_input.seq_mask == True:
				reach_start = idx_stack[:,0] == buffer.prev(idx_stack[:,0]) # (B, )
				reach_end = idx_stack[:,-1] == buffer.next(idx_stack[:,-1]) # (B, )
				batch_stack.valid_mask[reach_start==1,:] = 0
				batch_stack.valid_mask[reach_end==1,:] = 0
			elif self.global_cfg.actor_input.seq_mask == False:
				reach_start = idx_stack == buffer.prev(idx_stack) # (B, )
				reach_end = idx_stack == buffer.next(idx_stack) # (B, )
				batch_stack.valid_mask[reach_start==1] = 0
				batch_stack.valid_mask[reach_end==1] = 0
			else: raise ValueError("seq_mask should be True or False")
			burn_in_num = int(self.global_cfg.actor_input.burnin_num * self.global_cfg.actor_input.history_num) \
			if type(self.global_cfg.actor_input.burnin_num) == float \
			else self.global_cfg.actor_input.burnin_num
			batch_stack.valid_mask[:,:burn_in_num] = 0
			# obs_pred & obs_encode
			if self.global_cfg.actor_input.obs_pred.turn_on:
				batch_stack.pred_input_cur = batch_stack.a_in_cur
				batch_stack.pred_input_next = batch_stack.a_in_next # TODO the following
				batch_stack.pred_output_cur, pred_info_cur = self.pred_net(batch_stack.pred_input_cur, state=None)
				batch_stack.pred_output_next, pred_info_next = self.pred_net(batch_stack.pred_input_next, state=None)
				if self.global_cfg.actor_input.obs_pred.input_type == "obs":
					batch_stack.a_in_cur = batch_stack.pred_output_cur # (B*T, *)
					batch_stack.a_in_next = batch_stack.pred_output_next # (B*T, *)
				elif self.global_cfg.actor_input.obs_pred.input_type == "feat":
					batch_stack.a_in_cur = pred_info_cur["feats"] # (B*T, *)
					batch_stack.a_in_next = pred_info_next["feats"]
				else:
					raise NotImplementedError
				# detach
				if self.global_cfg.actor_input.obs_pred.middle_detach: 
					batch_stack.a_in_cur = batch_stack.a_in_cur.detach()
					batch_stack.a_in_next = batch_stack.a_in_next.detach()
				if self.global_cfg.actor_input.obs_pred.net_type == "vae":
					raise NotImplementedError("vae for rnn is not implemented yet")
					batch_stack.pred_info_cur_mu = pred_info_cur["mu"]
					batch_stack.pred_info_cur_logvar = pred_info_cur["logvar"]
			if self.global_cfg.actor_input.obs_encode.turn_on:
				raise NotImplementedError("obs_encode is not implemented yet")
				batch_end.encode_obs_input_cur = batch_end.a_in_cur
				batch_end.encode_obs_input_next = batch_end.a_in_next
				batch_end.encode_obs_output_cur, encode_obs_info_cur = self.encode_net.normal_encode(batch_end.encode_obs_input_cur)
				batch_end.encode_obs_output_next, encode_obs_info_next = self.encode_net.normal_encode(batch_end.encode_obs_input_next)
				batch_end.encode_oracle_obs_output_cur, encode_oracle_obs_info_cur = self.encode_net.oracle_encode(batch_end.info["obs_nodelay"])
				batch_end.encode_oracle_obs_output_next, encode_oracle_obs_info_next = self.encode_net.oracle_encode(batch_end.info["obs_next_nodelay"])
				batch_end.encode_normal_info_cur_mu = encode_obs_info_cur["mu"]
				batch_end.encode_normal_info_cur_logvar = encode_obs_info_cur["logvar"]
				batch_end.encode_oracle_info_cur_mu = encode_oracle_obs_info_cur["mu"]
				batch_end.encode_oracle_info_cur_logvar = encode_oracle_obs_info_cur["logvar"]
				if self.global_cfg.actor_input.obs_encode.train_eval_async == True:
					batch_end.a_in_cur = batch_end.encode_oracle_obs_output_cur
					batch_end.a_in_next = batch_end.encode_oracle_obs_output_next
				elif self.global_cfg.actor_input.obs_encode.train_eval_async == False:
					batch_end.a_in_cur = batch_end.encode_obs_output_cur
					batch_end.a_in_next = batch_end.encode_obs_output_next
				else:
					raise ValueError("batch_end error")
				if self.global_cfg.actor_input.obs_encode.pred_loss_weight:
					batch_end.pred_obs_output_cur, _ = self.encode_net.decode(batch_end.encode_obs_output_cur)
				if self.global_cfg.actor_input.obs_encode.before_policy_detach:
					batch_end.a_in_cur = batch_end.a_in_cur.detach()
					batch_end.a_in_next = batch_end.actor_input_next.detach()
			# end
			indices = idx_stack
			batch = batch_stack
			batch.obs_nodelay = self.buf[self.buf.prev(indices)].info["obs_next_nodelay"]
			batch.obs_nodelay = torch.tensor(batch.obs_nodelay, device=self.cfg.device)
			batch.obs_next_nodelay = batch.info["obs_next_nodelay"]
			batch.to_torch(device=self.cfg.device, dtype=torch.float32)
		else: raise ValueError("unknown history_merge_method: {}".format(self.global_cfg.actor_input.history_merge_method))
		# critic input
		if self.cfg.global_cfg.critic_input.obs_type == "normal":
			batch.c_in_cur = self.get_obs_base(batch, "actor", "cur")
			batch.c_in_next = self.get_obs_base(batch, "actor", "next")
		elif self.cfg.global_cfg.critic_input.obs_type == "oracle":
			batch.c_in_cur = batch.obs_nodelay
			batch.c_in_next = batch.obs_next_nodelay
		else: raise NotImplementedError

		# only keep res keys
		keeped_keys = ["a_in_cur", "a_in_next", "c_in_cur", "c_in_next", "done", "terminated", "truncated", "rew", "act", "valid_mask"]
		for k in list(batch.keys()): 
			if k not in keeped_keys: batch.pop(k)
		return batch, indices

	def update_once(self, batch, indices):
		batch, indices = self.pre_update_process(batch, indices)
		if not hasattr(self, "critic_update_cnt"): self.update_cnt = 0
		# update cirtic
		critic_info_ = self.update_critic(batch, indices)
		if self.update_cnt % self.cfg.policy.update_a_per_c == 0:
			# update actor
			actor_info_ = self.update_actor(batch, indices)
			self.record("learn/actor_loss", actor_info_["actor_loss"])
			self.exploration_noise *= self.cfg.policy.noise_decay_rate
		self.soft_update(self.actor_old, self.actor, self.cfg.policy.tau)
		self.soft_update(self.critic1_old, self.critic1, self.cfg.policy.tau)
		self.soft_update(self.critic2_old, self.critic2, self.cfg.policy.tau)
		self.record("learn/critic_loss", critic_info_["critic_loss"])
		self.update_cnt += 1

	def update_critic(self, batch, indices):

		batch.a_next_online = self.actor_old(batch.a_in_next, state=None)[0][0]
		noise = torch.randn(size=batch.a_next_online.shape, device=batch.a_next_online.device) * self.cfg.policy.noise_clip
		if self.cfg.policy.noise_clip > 0.0:
			noise = noise.clamp(-self.cfg.policy.noise_clip, self.cfg.policy.noise_clip)
		batch.a_next_online += noise
		
		target_q = (batch.rew + self.cfg.policy.gamma * (1 - batch.done.int()) * \
			torch.min(
				self.critic1_old(torch.cat([batch.c_in_next, batch.a_next_online],-1))[0],
				self.critic2_old(torch.cat([batch.c_in_next, batch.a_next_online],-1))[0]
			).squeeze(-1)
		).flatten().detach()
		
		critic_loss = F.mse_loss(self.critic1(
			torch.cat([batch.c_in_cur, batch.act],-1)
		)[0].flatten(), target_q, reduce=False) + F.mse_loss(self.critic2(
			torch.cat([batch.c_in_cur, batch.act],-1)
		)[0].flatten(), target_q, reduce=False)
		critic_loss = (critic_loss * batch.valid_mask.flatten()).mean()

		self.critic1_optim.zero_grad()
		self.critic2_optim.zero_grad()
		critic_loss.backward()
		self.critic1_optim.step()
		self.critic2_optim.step()

		return {
			"critic_loss": critic_loss.cpu().item()
		}

	def update_actor(self, batch, indices):
		res_info = {}
		actor_loss, _ = self.critic1(torch.cat([
			batch.c_in_cur, 
			self.actor(batch.a_in_cur, state=None)[0][0]
		],-1))
		actor_loss = actor_loss.flatten()
		actor_loss = (actor_loss * batch.valid_mask.flatten()).mean()
		actor_loss = -actor_loss.mean()
		self.actor_optim.zero_grad()
		actor_loss.backward()
		self.actor_optim.step()
		return {
			"actor_loss": actor_loss.cpu().item()
		}

	def soft_update(self, tgt: nn.Module, src: nn.Module, tau: float) -> None:
		"""Softly update the parameters of target module towards the parameters \
		of source module."""
		for tgt_param, src_param in zip(tgt.parameters(), src.parameters()):
			tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)

	def on_evaluate_end(self, **kwargs):
		"""called after a step of evaluation"""
		self.record("eval/rwd_mean", kwargs["rwd_mean"])
		self.record("eval/len_mean", kwargs["len_mean"])
		self.record("epoch", self.epoch_cnt)



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
				if stage == "cur": 
					try: return batch.info["obs_nodelay"]
					except: return batch.obs_nodelay
				elif stage == "next": return batch.info["obs_next_nodelay"]
		elif a_or_c == "critic":
			if self.global_cfg.critic_input.obs_type == "normal":
				if stage == "cur": return batch.obs
				elif stage == "next": return batch.obs_next
			elif self.global_cfg.critic_input.obs_type == "oracle":
				if stage == "cur": return batch.info["obs_nodelay"]
				elif stage == "next": return batch.info["obs_next_nodelay"]

	def get_historical_act(self, indices, step, buffer, type=None, device=None):
		""" get historical act
		input [t_0, t_1, ...]
		output [
			[t_0-step, t_0-step+1, ... t_0-1],
			[t_1-step, t_1-step+1, ... t_1-1],
			...
		]
		ps. note that cur step is not included
		ps. the neg step is set to 0.
		:param indices: indices of the batch (B,)
		:param step: the step of the batch. int
		:param buffer: the buffer. 
		:return: historical act (B, step)
		"""
		assert type in ["cat", "stack"], "type must be cat or stack"
		# [t_0-step, t_0-step+1, ... t_0-1, t_0]
		idx_stack_plus1 = utils.idx_stack(indices, buffer, step+1, direction="prev")
		# [t_0-step,   t_0-step+1, ..., t_0-1]
		idx_stack_next = idx_stack_plus1[:, :-1] # (B, step)
		# [t_0-step+1, t_0-step+2, ...,   t_0]
		idx_stack = idx_stack_plus1[:, 1:] # (B, step)
		invalid = (idx_stack_next == idx_stack) # (B, step)
		historical_act = buffer[idx_stack].act # (B, step, act_dim)
		historical_act[invalid] = 0.
		if type == "cat":
			historical_act = historical_act.reshape(historical_act.shape[0], -1) # (B, step*act_dim)
		historical_act = torch.tensor(historical_act, device=device)
		return historical_act




				


