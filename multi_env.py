#!/usr/bin/env python3

import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from typing import Callable
import gym


class MultiEnv(gym.Env):
    def __init__(self, env: Callable[[int], gym.Env], num_env=8):
        if num_env <= 0:
            return ValueError('Invalid # env = {} <= 0'.format(num_env))
        self.num_env = num_env
        self.envs = [env(i) for i in range(num_env)]
        self.pool = ThreadPool(processes=num_env)

        # NOTE(ycho): For now, replicate `env` rather than
        # using teh broadcasted version.
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def step(self, actions):
        return zip(*self.pool.map(lambda i: self.envs[i].step(actions[i]), range(self.num_env)))

    def reset(self):
        return self.pool.map(lambda i: self.envs[i].reset(), range(self.num_env))

    def sense(self):
        return self.pool.map(lambda i: self.envs[i].sense(), range(self.num_env))
