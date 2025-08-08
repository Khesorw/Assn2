import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CreateRedBall(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        print('Hello env created red ball')

        # Required: Define dummy action and observation space
        self.action_space = spaces.Discrete(3)         # Example: 3 actions
        self.observation_space = spaces.Discrete(10)   # Example: 10 discrete observations

        self.state = 0
        self.step_count = 0

    def reset(self, seed=None, options=None):
        print('Hello env created reset func')
        self.state = self.observation_space.sample()
        self.step_count = 0
        return self.state, {}

    def step(self, action):
        print('Hello env created step function')
        self.state = self.observation_space.sample()
        reward = 0.0
        terminated = self.step_count >= 100
        truncated = False
        self.step_count += 1
        return self.state, reward, terminated, truncated, {}

    def close(self):
        print('Hello env created close function')
