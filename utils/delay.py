import gymnasium as gym
import numpy as np

class DelayedRoboticEnv(gym.Env):
    def __init__(self, env0: gym.Env, delay_steps=2):
        self.env = env0
        self.delay_steps = delay_steps

        self.action_space = self.env.action_space

        self.observation_space = self.env.observation_space
        self.oracle_observation_space = self.env.observation_space

        self.delayed_obs = np.zeros([delay_steps + 1] + list(self.observation_space.shape), dtype=np.float32)
        # self.delayed_obs[-1] is the current obs
        # self.delayed_obs[-n] is the obs delayed for (n - 1) steps

    def reset(self):
        s, info = self.env.reset()
        for stp in range(self.delay_steps + 1):
            self.delayed_obs[stp] = s
        return s, info

    def step(self, action):
        sp, r, done, truncated, info = self.env.step(action)
        for stp in range(self.delay_steps + 1):
            self.delayed_obs[stp] = sp if stp == self.delay_steps else self.delayed_obs[stp + 1]
        return self.delayed_obs[0], r, done, truncated, info

    def get_obs(self):
        return self.delayed_obs[0]

    def get_oracle_obs(self):
        return self.delayed_obs[-1]

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()