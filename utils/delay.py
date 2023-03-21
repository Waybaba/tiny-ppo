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
        self.delayed_obs = np.zeros([delay_steps + 1] + list(self.observation_space.shape), dtype=np.float32)
        self.delay_buf = Queue(maxsize=delay_steps+1)

        # history merge
        if self.global_cfg.actor_input.history_merge_method != "none" or \
            self.global_cfg.critic_input.history_merge_method != "none":
            assert self.global_cfg.actor_input.history_num and \
                self.global_cfg.actor_input.history_num > 0
            self.history_num = self.global_cfg.actor_input.history_num # short flag
            self.act_buf = [np.zeros(self.env.action_space.shape) for _ in range(self.history_num)]
        else:
            self.history_num = 0

    def reset(self):
        res = self.env.reset()
        if isinstance(res, tuple): # (obs, {}) # discard info {}
            # res[-1]["obs_cur"] = res[0]
            return res[0]
        self.prev_act = np.zeros_like(self.env.action_space.shape) # ! TODO conditional set
        if self.global_cfg.historical_act and self.global_cfg.historical_act.num:
            self.act_buf = [np.zeros_like(self.env.action_space.shape) for _ in range(int(self.historical_act.type.split("-")[1]))]
        return res

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
        self.delay_buf.put(obs_next_nodelay)
        while not self.delay_buf.full(): self.delay_buf.put(obs_next_nodelay) # make it full
        obs_next_delayed = self.delay_buf.get()
        info["obs_next_nodelay"] = obs_next_nodelay
        info["obs_next_delayed"] = obs_next_delayed
        # history merge
        if self.history_num > 0:
            info["historical_act"] = np.concatenate(self.act_buf, axis=0)
            self.act_buf.append(action)
            self.act_buf.pop(0)
        return (deepcopy(obs_next_delayed), deepcopy(reward), deepcopy(done or truncated), deepcopy(info))

    def step(self, action):
        """
        make a queue of delayed observations, the size of the queue is delay_steps
        for example, if delay_steps = 2, then the queue is [s_{t-2}, s_{t-1}, s_t]
        for each step, the queue will be updated as [s_{t-1}, s_t, s_{t+1}]
        """
        res = self.env.step(action)
        return self.preprocess_fn(res, action)


    def get_obs(self):
        return self.delayed_obs[0]

    def get_oracle_obs(self):
        return self.delayed_obs[-1]

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