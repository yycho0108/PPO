#!/usr/bin/env python3

from dataclasses import dataclass
import gym
import gym.envs.classic_control
import time

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

# Hyperparameters
# suggested to be something like 3e-4 by default in survey paper.
learning_rate = 0.0005
gamma = 0.98
lmbda = 0.95
# suggested to be something like 0.25 by default in survey paper.
eps_clip = 0.1
K_epoch = 3
T_horizon = 20

# from phonebot.sim.pybullet.simulator import PybulletPhonebotEnv

from subproc_wrapper import subproc
from subproc_env import MultiEnv


@dataclass
class PPOSettings:
    # learning_rate:float =0.0005
    learning_rate: float = 3e-4
    gamma: float = 0.98
    lmbda: float = 0.95
    # eps_clip:float = 0.1
    eps_clip: float = 0.25
    num_epoch: int = 3
    len_horizon: int = 20


class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.fc1 = nn.Linear(32, 256)
        self.fc_pi = nn.Linear(256, 8)
        self.fc_v = nn.Linear(256, 1)

    def pi(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        # prob = F.softmax(x, dim=-1)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v


class PPOTrainer(object):
    def __init__(self, model: PPO, device: th.device):
        self.device = device
        self.model = model
        self.data = []
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def add_data(self, data):
        self.data.append(data)

    def get_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        # Convert
        s = th.tensor(s_lst, dtype=th.float, device=self.device)
        a = th.tensor(a_lst, device=self.device)
        r = th.tensor(r_lst, dtype=th.float, device=self.device)
        s_prime = th.tensor(
            s_prime_lst, dtype=th.float, device=self.device)
        done_mask = th.tensor(
            done_lst, dtype=th.float, device=self.device)
        prob_a = th.tensor(
            prob_a_lst, dtype=th.float, device=self.device)

        # Clear `self.data`.
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train(self):
        s, a, r, s_prime, done_mask, prob_a = self.get_batch()

        for i in range(K_epoch):
            state_value = self.model.v(s)
            with th.no_grad():
                # compute td target.
                td_target = r + gamma * self.model.v(s_prime) * done_mask
                delta = td_target - state_value

                # reverse delta.
                rvi = th.tensor(
                    range(delta.shape[-1])[::-1], device=self.device)
                delta_rv = delta[..., rvi]

                # compute advantage.
                # I suppose this way is preferable since th does not support negative strides.
                advantage_lst = []
                advantage = 0.0
                for delta_t in delta_rv:
                    advantage = gamma * lmbda * advantage + delta_t[0]
                    advantage_lst.append([advantage])
                advantage_lst.reverse()
                advantage = th.tensor(
                    advantage_lst, dtype=th.float, device=self.device)

            pi = self.model.pi(s)
            pi_a = pi.gather(1, a)
            # a/b == exp(log(a)-log(b))
            ratio = th.exp(th.log(pi_a) - th.log(prob_a))

            surr1 = ratio * advantage
            surr2 = th.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            # policy_loss = -th.min(surr1, surr2)
            # value_loss = F.smooth_l1_loss(state_value, td_target)
            # loss = policy_loss + value_loss
            loss = -th.min(surr1, surr2) + \
                F.smooth_l1_loss(state_value, td_target)

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()


@subproc
class CartPoleSubprocEnv(gym.envs.classic_control.CartPoleEnv):
    pass



def main():
    gym.register(
        id='CartPoleSubproc-v0',
        entry_point='ppo:CartPoleSubprocEnv',
        max_episode_steps=500,
        reward_threshold=475.0,
    )
    def get_env(i):
        # return gym.make('CartPoleSubproc-v0')
        return CartPoleSubprocEnv()
    env = MultiEnv(get_env, 8)

    # env = gym.make('CartPole-v1')
    # env = gym.make('phonebot-pybullet-headless-v0')
    train = True

    # Resolve device.
    if False:
        if th.cuda.is_available():
            print('using cuda')
            device = th.device('cuda')
        else:
            device = th.device('cpu')
    else:
        device = th.device('cpu')
        # device = th.device('cuda')

    model = PPO().to(device)

    if train:
        trainer = PPOTrainer(model, device)

        score = 0.0
        print_interval = 20

        th.save(model.state_dict(), '/tmp/ppo.pth')
        for n_epi in range(10000):
            s = env.reset()
            done = False
            while not done:
                for t in range(T_horizon):
                    prob = model.pi(th.from_numpy(
                        s).float().to(device)).detach().cpu()
                    a = Categorical(prob).sample().item()
                    s_prime, r, done, info = env.step(a)

                    entry = (s, a, r/100.0, s_prime, prob[a].item(), done)
                    trainer.add_data(entry)
                    s = s_prime

                    score += r
                    if done:
                        break

                trainer.train()

            if n_epi % print_interval == 0 and n_epi != 0:
                print("# of episode :{}, avg score : {:.1f}".format(
                    n_epi, score/print_interval))
                score = 0.0
        th.save(model.state_dict(), '/tmp/ppo.pth')
    else:
        state_dict = th.load('/tmp/ppo.pth')
        model.load_state_dict(state_dict)
        model.eval()

        for n_epi in range(5):
            s = env.reset()
            done = False
            i = 0
            while not done:
                with th.no_grad():
                    prob = model.pi(th.from_numpy(s).float().to(device)).cpu()
                a = Categorical(prob).sample().item()
                env.render()
                s, r, done, _ = env.step(a)
                time.sleep(0.01)
                i += 1
            print('Done @ {}'.format(i))

    env.close()


if __name__ == '__main__':
    main()
