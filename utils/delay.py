import gymnasium as gym
import numpy as np
from queue import Queue
from copy import deepcopy

class DelayedRoboticEnv(gym.Wrapper):
    def __init__(self, env0: gym.Env, delay_steps=2, global_config=None):
        super().__init__(env0)
        self.env = env0
        self.delay_steps = delay_steps
        self.global_cfg = global_config

        # delayed observations
        self.delay_buf = Queue(maxsize=delay_steps+1)
        self.last_oracle_obs = None


        # history merge
        if self.global_cfg.actor_input.history_merge_method != "none" or \
            self.global_cfg.critic_input.history_merge_method != "none":
            if self.global_cfg.critic_input.history_merge_method != "none": raise NotImplementedError
            self.history_num = self.global_cfg.history_num # short flag
            self.act_buf = [np.zeros(self.env.action_space.shape) for _ in range(self.history_num)]
        else:
            self.history_num = 0

    def reset(self):
        # get obs_next_nodelay
        res = self.env.reset()
        if isinstance(res, tuple): # (obs, {}) # discard info {}
            obs_next_nodelay, info = res
        else:
            obs_next_nodelay, info = res, {}
        
        # empty then fill the delay_buf with zeros
        while not self.delay_buf.empty(): self.delay_buf.get()
        while not self.delay_buf.full(): self.delay_buf.put(np.zeros_like(obs_next_nodelay))
        
        # empty then fill the act_buf with zeros
        self.prev_act = np.zeros(self.env.action_space.shape)
        if self.history_num > 0:
            self.act_buf = [np.zeros(self.env.action_space.shape) for _ in range(self.history_num)]
            info["historical_act"] = np.stack(self.act_buf, axis=0)
        
        # get obs_next_delayed
        obs_next_delayed = self.delay_buf.get()
        self.delay_buf.put(obs_next_nodelay)
        self.last_oracle_obs = obs_next_nodelay
        info["obs_next_nodelay"] = obs_next_nodelay
        info["obs_next_delayed"] = obs_next_delayed
        info["obs_nodelay"] = None
        return obs_next_delayed, info

    def preprocess_fn(self, res, action):
        """
        preprocess the observation before the agent decision
        """
        if len(res) == 4: 
            obs_next_nodelay, reward, done, info = res
        elif len(res) == 5:
            obs_next_nodelay, reward, done, truncated, info = res
        else:
            raise ValueError("Invalid return value from env.step()")
        # operate the delayed observation queue
        assert self.delay_buf.full(), "delay_buf should be filled with zeros in reset()"
        obs_next_delayed = self.delay_buf.get()
        self.delay_buf.put(obs_next_nodelay)
        info["obs_next_nodelay"] = obs_next_nodelay
        info["obs_next_delayed"] = obs_next_delayed
        info["obs_nodelay"] = self.last_oracle_obs
        self.last_oracle_obs = obs_next_nodelay
        # history merge
        if self.history_num > 0:
            # info["historical_act"] = np.concatenate(self.act_buf, axis=0)
            info["historical_act"] = np.stack(self.act_buf, axis=0)
            self.act_buf.append(action)
            self.act_buf.pop(0)
        elif self.history_num == 0:
            info["historical_act"] = False
        return (deepcopy(obs_next_delayed), deepcopy(reward), deepcopy(done or truncated), deepcopy(info))

    def step(self, action):
        """
        make a queue of delayed observations, the size of the queue is delay_steps
        for example, if delay_steps = 2, then the queue is [s_{t-2}, s_{t-1}, s_t]
        for each step, the queue will be updated as [s_{t-1}, s_t, s_{t+1}]
        """
        res = self.env.step(action)
        return self.preprocess_fn(res, action)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

# RLlib version
class RLlibDelayedRoboticEnv(DelayedRoboticEnv):
    def __init__(self, env0: gym.Env, env_config):
        super().__init__(env0, env_config["delay_steps"])



class DelayQueue:
    """ a queue for delayed observations storation
    """
    def __init__(self, size):
        self.size = size
        self.queue = []