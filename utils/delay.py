import gymnasium as gym
import numpy as np
from queue import Queue
from copy import deepcopy

   


class DelayedRoboticEnv(gym.Wrapper):
    """
    Args:
        fixed_delay: if True, the delay_steps is fixed. Otherwise, 
            the delay_steps is sampled from a uniform distribution
            between [0, max_delay_steps)
            self.delay_buf = [delay=max, delay=max-1, ..., delay=0]
            idx = - (delay + 1) + max
        delay_keep_order_method:
            "none": random sample from the delay_buf
            "expect1": sample from the delay_buf with the step forward of 1 [0,1,2]
    Returns:
        info:
            obs_next_nodelay: the next observation without delay
            obs_next_delayed: the next observation with delay
            historical_act_cur: the historical actions of the next step [a_{t-max-1}, ..., a_{t-2}]
            historical_act_next: the historical actions of the next step [a_{t-max}, ..., a_{t-1}]
            historical_obs_cur: the historical observations of the next step [o_{t-max}, ..., o_{t-1}]
            historical_obs_next: the historical observations of the next step [o_{t-max+1}, ..., o_{t}]

    """
    metadata = {'render.modes': ['human', 'text']}

    def __init__(
            self, 
            base_env: gym.Env, 
            delay_steps, 
            fixed_delay, 
            global_config
        ):
        super().__init__(base_env)
        
        self.env = base_env
        self.delay_steps = delay_steps
        self.fixed_delay = fixed_delay
        self.global_cfg = global_config

        # setup for delayed observations
        self.delay_buf = ListAsQueue(maxsize=delay_steps+1)
        self.last_oracle_obs = None
        self.last_delayed_step = None # for debug

        # setup history merge
        if self.global_cfg.history_num:
            self.history_num = self.global_cfg.history_num
        else:
            self.history_num = 0

    def reset(self):
        # pre: adapt to different envs
        res = self.env.reset()
        if isinstance(res, tuple): obs_next_nodelay, info = res
        else: obs_next_nodelay, info = res, {}
        
        # reset delay_buf - empty then fill the delay_buf with zeros
        while not self.delay_buf.empty(): self.delay_buf.get()
        while not self.delay_buf.full(): self.delay_buf.put(np.zeros_like(obs_next_nodelay))
        
        # reset act_buf,prev_act - empty then fill the act_buf with zeros
        if self.history_num > 0:
            self.act_buf = [np.zeros(self.env.action_space.shape) for _ in range(self.history_num)]
            self.obs_buf = [np.zeros_like(obs_next_nodelay) for _ in range(self.history_num)]
        else:
            pass
        
        # update delay_buf
        self.delay_buf.get()
        self.delay_buf.put(obs_next_nodelay) # [max,max-1, ..., 1, 0]

        # get index
        if not self.fixed_delay:
            if not self.global_cfg.debug.delay_keep_order_method:
                self.last_delayed_step = np.random.randint(0, self.delay_steps+1) if self.delay_steps > 0 else 0
            elif self.global_cfg.debug.delay_keep_order_method == "expect1":
                self.last_delayed_step = self.delay_steps # start from the max delay step
                # self.last_delayed_step = self.delay_steps // 2 # start from the middle delay step
                self.last_delayed_step = np.random.randint(self.last_delayed_step-1, self.last_delayed_step+2)
                self.last_delayed_step = np.clip(self.last_delayed_step, 0, self.delay_steps)
            else:
                raise ValueError("Invalid delay_keep_order_method {}".format(self.global_cfg.debug.delay_keep_order_method))
        else:
            self.last_delayed_step = self.delay_steps

        # get
        obs_next_delayed = self.delay_buf[self.delay_steps - self.last_delayed_step] # 0 -> 0

        # info
        if self.history_num > 0:
            self.obs_buf.append(obs_next_delayed)
            self.obs_buf.pop(0)
            info["historical_act_next"] = np.stack(self.act_buf, axis=0)
            info["historical_act_cur"] = np.stack(self.act_buf, axis=0)
            info["historical_obs_next"] = np.stack(self.obs_buf, axis=0)
            info["historical_obs_cur"] = np.stack(self.obs_buf, axis=0)
        else:
            info["historical_act_next"] = False
            info["historical_act_cur"] = False
            info["historical_obs_next"] = False
            info["historical_obs_cur"] = False


        info["obs_next_nodelay"] = obs_next_nodelay
        info["obs_next_delayed"] = obs_next_delayed
        info["obs_nodelay"] = None
        info["obs_delayed_step_num"] = self.last_delayed_step
        
        # end
        self.last_oracle_obs = obs_next_nodelay

        return obs_next_delayed, info

    def preprocess_fn(self, res, action):
        """
        preprocess the observation before the agent decision
        """
        # pre: adapt to different envs
        if len(res) == 4: 
            obs_next_nodelay, reward, done, info = res
            truncated = False
        elif len(res) == 5:
            obs_next_nodelay, reward, done, truncated, info = res
        else:
            raise ValueError("Invalid return value from env.step()")
        
        # update delay_buf
        obs_next_delayed = self.delay_buf.get()
        self.delay_buf.put(obs_next_nodelay)

        # get index
        if not self.fixed_delay: # replace obs_next_delayed and self.last_delayed_step
            if not self.global_cfg.debug.delay_keep_order_method:
                self.last_delayed_step = np.random.randint(0, self.delay_steps+1) if self.delay_steps > 0 else 0
            elif self.global_cfg.debug.delay_keep_order_method == "expect1":
                self.last_delayed_step = np.random.randint(self.last_delayed_step-1, self.last_delayed_step+2)
                self.last_delayed_step = np.clip(self.last_delayed_step, 0, self.delay_steps)
            else:
                raise ValueError("Invalid delay_keep_order_method {}".format(self.global_cfg.debug.delay_keep_order_method))
        else:
            self.last_delayed_step = self.delay_steps
        
        # get
        obs_next_delayed = self.delay_buf[self.delay_steps - self.last_delayed_step]
        
        
        info["obs_next_nodelay"] = obs_next_nodelay
        info["obs_next_delayed"] = obs_next_delayed
        info["obs_nodelay"] = self.last_oracle_obs
        info["obs_delayed_step_num"] = self.last_delayed_step
        
        # end
        self.last_oracle_obs = obs_next_nodelay

        # act merge
        if self.history_num > 0:
            info["historical_act_cur"] = np.stack(self.act_buf, axis=0)
            info["historical_obs_cur"] = np.stack(self.obs_buf, axis=0)
            self.act_buf.append(action)
            self.obs_buf.append(obs_next_delayed)
            self.act_buf.pop(0)
            self.obs_buf.pop(0)
            info["historical_act_next"] = np.stack(self.act_buf, axis=0)
            info["historical_obs_next"] = np.stack(self.obs_buf, axis=0)
        elif self.history_num == 0:
            info["historical_act_cur"] = False
            info["historical_obs_cur"] = False
            info["historical_act_next"] = False
            info["historical_obs_next"] = False
        
        return (deepcopy(obs_next_delayed), deepcopy(reward), deepcopy(done), deepcopy(truncated), deepcopy(info))

    def step(self, action):
        """
        make a queue of delayed observations, the size of the queue is delay_steps
        for example, if delay_steps = 2, then the queue is [s_{t-2}, s_{t-1}, s_t]
        for each step, the queue will be updated as [s_{t-1}, s_t, s_{t+1}]
        """
        res = self.env.step(action)
        return self.preprocess_fn(res, action)



class StickyActionWrapper(gym.Wrapper):
    """
    Source: https://github.com/openai/random-network-distillation/blob/master/atari_wrappers.py
    """
    def __init__(self, env, p=0.25):
        super().__init__(env)
        self.p = p

    def reset(self):
        self.last_action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        return self.env.reset()

    def step(self, action):
        if self.unwrapped.np_random.uniform() < self.p:
            action = self.last_action
        self.last_action = action
        obs_next_nodelay, reward, done, truncated, info = self.env.step(action)
        return obs_next_nodelay, reward, done, truncated, info


class GaussianNoiseActionWrapper(gym.Wrapper):
    def __init__(self, env, noise_fraction=0.2):
        super().__init__(env)
        self.noise_fraction = noise_fraction

    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        # Calculate the noise scale based on action space range
        action_range = self.action_space.high - self.action_space.low
        noise_scale = self.noise_fraction * action_range * 0.5

        # Add Gaussian noise to the action
        noisy_action = action + np.random.normal(0, noise_scale, size=action.shape)

        # Clip the noisy action to the action range
        clipped_action = np.clip(noisy_action, self.action_space.low, self.action_space.high)

        # Take a step in the environment with the clipped action
        return self.env.step(clipped_action)


# utils

class ListAsQueue:
    """ A queue implemented by list, which support indexing.
    """
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.queue = []
    
    def put(self, item):
        if len(self.queue) >= self.maxsize:
            self.queue.pop(0)
        self.queue.append(item)
    
    def get(self):
        return self.queue.pop(0)

    def empty(self):
        return len(self.queue) == 0
    
    def full(self):
        return len(self.queue) == self.maxsize
    
    def __getitem__(self, idx):
        return self.queue[idx]

    def __len__(self):
        return len(self.queue)
