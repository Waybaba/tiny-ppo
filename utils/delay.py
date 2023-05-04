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

        # delayed observations
        self.delay_buf = ListAsQueue(maxsize=delay_steps+1)
        self.last_oracle_obs = None
        self.last_delayed_step = None # for debug

        # history merge
        if self.global_cfg.history_num:
            self.history_num = self.global_cfg.history_num
            self.act_buf = [np.zeros(self.env.action_space.shape) for _ in range(self.history_num)]
        else:
            self.history_num = 0

    def reset(self):
        # get obs_next_nodelay
        res = self.env.reset()
        if isinstance(res, tuple): obs_next_nodelay, info = res
        else: obs_next_nodelay, info = res, {}
        
        # reset delay_buf - empty then fill the delay_buf with zeros
        while not self.delay_buf.empty(): self.delay_buf.get()
        while not self.delay_buf.full(): self.delay_buf.put(np.zeros_like(obs_next_nodelay))
        
        # reset act_buf,prev_act - empty then fill the act_buf with zeros
        self.prev_act = np.zeros(self.env.action_space.shape)
        if self.history_num > 0:
            self.act_buf = [np.zeros(self.env.action_space.shape) for _ in range(self.history_num)]
            info["historical_act"] = np.stack(self.act_buf, axis=0)
        
        # update delay_buf
        self.delay_buf.get()
        self.delay_buf.put(obs_next_nodelay) # [max,max-1, ..., 1, 0]

        # get index
        if not self.fixed_delay:
            if not self.global_cfg.debug.delay_keep_order_method:
                self.last_delayed_step = np.random.randint(0, self.delay_steps) if self.delay_steps > 0 else 0
            elif self.global_cfg.debug.delay_keep_order_method == "expect1":
                self.last_delayed_step = len(self.delay_buf) - 1 # start from the max delay step
                self.last_delayed_step = np.random.randint(self.last_delayed_step-1, self.last_delayed_step+2)
                self.last_delayed_step = np.clip(self.last_delayed_step, 0, self.delay_steps-1)
            else:
                raise ValueError("Invalid delay_keep_order_method {}".format(self.global_cfg.debug.delay_keep_order_method))
        else:
            self.last_delayed_step = self.delay_steps

        # get
        obs_next_delayed = self.delay_buf[self.delay_steps - self.last_delayed_step] # 0 -> 0

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
        if len(res) == 4: 
            obs_next_nodelay, reward, done, info = res
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

        # history merge
        if self.history_num > 0:
            # info["historical_act"] = np.concatenate(self.act_buf, axis=0)
            info["historical_act"] = np.stack(self.act_buf, axis=0)
            self.act_buf.append(action)
            self.act_buf.pop(0)
        elif self.history_num == 0:
            info["historical_act"] = False
        
        return (deepcopy(obs_next_delayed), deepcopy(reward), deepcopy(done), deepcopy(truncated), deepcopy(info))

    def step(self, action):
        """
        make a queue of delayed observations, the size of the queue is delay_steps
        for example, if delay_steps = 2, then the queue is [s_{t-2}, s_{t-1}, s_t]
        for each step, the queue will be updated as [s_{t-1}, s_t, s_{t+1}]
        """
        res = self.env.step(action)
        return self.preprocess_fn(res, action)



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
