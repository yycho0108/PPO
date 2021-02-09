#!/usr/bin/env python3

import numpy as np
import time
import gym
from typing import Callable

from subproc import subproc
from multi_env import MultiEnv

from phonebot.sim.pybullet.simulator import PybulletPhonebotEnv, PybulletSimulatorSettings


@subproc
class PybulletPhonebotSubprocEnv(PybulletPhonebotEnv):
    pass


def main():
    # env = PybulletPhonebotSubprocEnv()
    action_size = 8
    num_env = 4

    def get_env(index: int):
        # env = PybulletPhonebotEnv(sim_settings=PybulletSimulatorSettings(
        # render=False, random_orientation=True))
        env = PybulletPhonebotSubprocEnv(
            PybulletSimulatorSettings(render=False, random_orientation=True))
        env.set_seed(index)
        env.reset()
        return env

    env = MultiEnv(get_env, num_env)
    while True:
        print(env.sense())
        res = env.step([np.zeros(action_size) for _ in range(num_env)])
        print(res[0], res[1], res[2], res[3])
        time.sleep(0.1)
        break


if __name__ == '__main__':
    main()
