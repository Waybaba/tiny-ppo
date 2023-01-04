import pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
from typing import Any, Dict, List, Optional, Type, Union
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
from tianshou.policy import BasePolicy, PGPolicy
from tianshou.utils import RunningMeanStd
import tianshou
from torch.utils.tensorboard import SummaryWriter
import rich
import utils

import warnings
warnings.filterwarnings('ignore')


class PGPolicy(BasePolicy):
    """Implementation of REINFORCE algorithm.

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logitianshou)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param dist_fn: distribution class for computing the action.
    :type dist_fn: Type[torch.distributions.Distribution]
    :param float discount_factor: in [0, 1]. Default to 0.99.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action), "tanh" (for applying tanh
        squashing) for now, or empty string for no bounding. Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjustianshou the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).
    :param bool deterministic_eval: whether to use deterministic action instead of
        stochastic action sampled by the policy. Default to False.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        dist_fn: Type[torch.distributions.Distribution],
        discount_factor: float = 0.99,
        reward_normalization: bool = False,
        action_scaling: bool = True,
        action_bound_method: str = "clip",
        deterministic_eval: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            **kwargs
        )
        self.actor = model
        self.optim = optim
        self.dist_fn = dist_fn
        assert 0.0 <= discount_factor <= 1.0, "discount factor should be in [0, 1]"
        self._gamma = discount_factor
        self._rew_norm = reward_normalization
        self.ret_rms = RunningMeanStd()
        self._eps = 1e-8
        self._deterministic_eval = deterministic_eval

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        r"""Compute the discounted returns for each transition.

        .. math::
            G_t = \sum_{i=t}^T \gamma^{i-t}r_i

        where :math:`T` is the terminal time step, :math:`\gamma` is the
        discount factor, :math:`\gamma \in [0, 1]`.
        """
        v_s_ = np.full(indices.shape, self.ret_rms.mean)
        unnormalized_returns, _ = self.compute_episodic_return(
            batch, buffer, indices, v_s_=v_s_, gamma=self._gamma, gae_lambda=1.0
        )
        if self._rew_norm:
            batch.returns = (unnormalized_returns - self.ret_rms.mean) / \
                np.sqrt(self.ret_rms.var + self._eps)
            self.ret_rms.update(unnormalized_returns)
        else:
            batch.returns = unnormalized_returns
        return batch

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which has 4 keys:

            * ``act`` the action.
            * ``logitianshou`` the network's raw output.
            * ``dist`` the action distribution.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        logitianshou, hidden = self.actor(batch.obs, state=state)
        if isinstance(logitianshou, tuple):
            dist = self.dist_fn(*logitianshou)
        else:
            dist = self.dist_fn(logitianshou)
        if self._deterministic_eval and not self.training:
            if self.action_type == "discrete":
                act = logitianshou.argmax(-1)
            elif self.action_type == "continuous":
                act = logitianshou[0]
        else:
            act = dist.sample()
        return Batch(logitianshou=logitianshou, act=act, state=hidden, dist=dist)

    def learn(  # type: ignore
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        losses = []
        for _ in range(repeat):
            for minibatch in batch.split(batch_size, merge_last=True):
                self.optim.zero_grad()
                result = self(minibatch)
                dist = result.dist
                act = to_torch_as(minibatch.act, result.act)
                ret = to_torch(minibatch.returns, torch.float, result.act.device)
                log_prob = dist.log_prob(act).reshape(len(ret), -1).transpose(0, 1)
                loss = -(log_prob * ret).mean()
                loss.backward()
                self.optim.step()
                losses.append(loss.item())

        return {"loss": losses}

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
@hydra.main(version_base=None, config_path=str(root / "configs"), config_name="vpg.yaml")
def main(cfg):
	# init
	utils.print_config_tree(cfg, resolve=True)
	wandb.init(project=cfg.task_name, tags=cfg.tags,config=utils.config_format(cfg))
	cfg = hydra.utils.instantiate(cfg)
	# env & not & policy
	train_envs = tianshou.env.DummyVectorEnv([lambda: gym.make(cfg.env.name) for _ in range(cfg.env.train_num)])
	test_envs = tianshou.env.DummyVectorEnv([lambda: gym.make(cfg.env.name) for _ in range(cfg.env.test_num)])
	env = gym.make(cfg.env.name)
	net = cfg.net(
		state_shape=env.observation_space.shape or env.observation_space.n, 
		action_shape=env.action_space.shape or env.action_space.n, 
	)
	optimizer = cfg.optimizer(net.parameters()) 
	policy = cfg.policy(
		net, optimizer,
		action_space=env.action_space.shape or env.action_space.n
	)
	# collector
	train_collector = cfg.train_collector(policy, train_envs)
	test_collector = cfg.test_collector(policy, test_envs)
	# train
	logger = tianshou.utils.WandbLogger(config=cfg)
	logger.load(SummaryWriter(cfg.output_dir))
	trainer = cfg.trainer(
		policy, train_collector, test_collector, 
		stop_fn=lambda mean_reward: mean_reward >= 1000,
		logger=logger,
	)
	for epoch, epoch_stat, info in trainer:
		wandb.log(epoch_stat)
		wandb.log(info)

	
if __name__ == "__main__":
	main()
