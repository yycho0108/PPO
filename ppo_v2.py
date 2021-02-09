#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK


import numpy as np
import gym
import itertools
import time
from functools import partial
from pathlib import Path

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard.writer import SummaryWriter

from dataclasses import dataclass, field, replace, asdict
from simple_parsing import ArgumentParser
import argcomplete

from typing import List, Tuple, Dict

from subproc import subproc
from multi_env import MultiEnv
from ring_buffer import ContiguousRingBuffer
# from phonebot.sim.pybullet.simulator import PybulletPhonebotEnv, PybulletSimulatorSettings


@dataclass
class AcSettings:
    encoder_pi: List[int] = field(default_factory=lambda: [64, 64])
    encoder_v: List[int] = field(default_factory=lambda: [64, 64])
    activation_fn: nn.Module = nn.Tanh
    log_std_init: float = 0.0
    ortho_init: bool = True


@dataclass
class GaeSettings:
    gamma: float = 0.999
    lmbda: float = 0.98


@dataclass
class PpoSettings:
    eps: float = 0.2  # clipping epsilon
    clip_range: float = 0.2

    entropy_weight: float = 0.01
    value_weight: float = 0.5
    max_grad_norm: float = 0.5


@dataclass
class Settings:
    ac: AcSettings = AcSettings()
    gae: GaeSettings = GaeSettings()
    ppo: PpoSettings = PpoSettings()
    num_envs: int = 16
    device: str = 'cuda'
    batch_size: int = 64

    max_steps: int = 5e6
    update_steps: int = 1024
    save_steps: int = 1e5
    num_epochs: int = 4  # ppo-style train epochs

    # exponential decay param for reward distribution estimation
    reward_normalizer_alpha: float = 0.001
    # exponential decay param for state distribution estimation
    state_normalizer_alpha: float = 0.001

    train: bool = True
    model_path: str = '/tmp/model.zip'
    ckpt_path: str = '/tmp/ppo-ckpt'
    subproc: bool = True  # only used if `train`


def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def gae(memory: ContiguousRingBuffer, last_values: np.ndarray,
        dones: np.ndarray, opts: GaeSettings) -> None:
    """
    Post-processing step: compute the returns (sum of discounted rewards)
    and GAE advantage.
    Adapted from Stable-Baselines PPO2.

    Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
    to compute the advantage. To obtain vanilla advantage (A(s) = R - V(S))
    where R is the discounted reward with value bootstrap,
    set ``gae_lambda=1.0`` during initialization.
    """
    last_gae_lam = 0
    n = len(memory)

    advantages = np.zeros(memory['value'].shape, np.float32)
    for step in reversed(range(n)):
        entry = memory[step]

        if step == n - 1:
            next_non_terminal = 1.0 - dones
            next_values = last_values
        else:
            nxt_entry = memory[step + 1]
            next_non_terminal = 1.0 - nxt_entry['done']
            next_values = nxt_entry['value']

        delta = (entry['reward'] + opts.gamma *
                 next_non_terminal * next_values - entry['value'])
        last_gae_lam = (delta + opts.gamma *
                        opts.lmbda * next_non_terminal * last_gae_lam)
        advantages[step] = last_gae_lam
    returns = advantages + memory['value']
    return (advantages, returns)


class AC(nn.Module):
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 opts: AcSettings
                 ):
        super().__init__()
        self.opts = opts

        # encoders
        layers = []
        for prv, nxt in pairwise([state_size] + opts.encoder_pi):
            layers.append(nn.Linear(prv, nxt))
            layers.append(opts.activation_fn())
        self.encoder_pi = nn.Sequential(*layers)

        layers = []
        for prv, nxt in pairwise([state_size] + opts.encoder_v):
            layers.append(nn.Linear(prv, nxt))
            layers.append(opts.activation_fn())
        self.encoder_v = nn.Sequential(*layers)

        # actually useful things
        self.c = nn.Linear(opts.encoder_pi[-1], action_size)
        self.v = nn.Linear(opts.encoder_v[-1], 1)

        # NOTE(ycho):
        self.log_std = nn.Parameter(
            th.full((action_size,), opts.log_std_init),
            requires_grad=True)

        # NOTE(ycho): Orthogonal initialization...
        # apparently quite important???
        if opts.ortho_init:
            gains = {
                self.encoder_pi: np.sqrt(2),
                self.encoder_v: np.sqrt(2),
                self.c: 0.01,
                self.v: 1.0
            }

            def init_weights(m, g: float):
                if not isinstance(m, (nn.Linear, nn.Conv2d)):
                    return
                nn.init.orthogonal_(m.weight, gain=g)
                if m.bias is not None:
                    m.bias.data.fill_(0)

            for m, g in gains.items():
                m.apply(partial(init_weights, g=g))

    def action_dist(self, state: th.Tensor) -> th.distributions.Distribution:
        f_pi = self.encoder_pi(state)
        c = self.c(f_pi)
        dist = Normal(c, self.log_std.exp())
        return dist

    def forward(self, state: th.Tensor,
                deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        # action
        dist = self.action_dist(state)
        a = dist.mean if deterministic else dist.sample()

        # value
        f_v = self.encoder_v(state)
        v = self.v(f_v)

        # log prob
        lp = dist.log_prob(a)
        return (a, v, lp)

    def action(self, state: th.Tensor, deterministic: bool = False):
        dist = self.action_dist(state)
        if deterministic:
            return dist.mean
        return dist.sample()

    def value(self, state: th.Tensor):
        f_v = self.encoder_v(state)
        v = self.v(f_v)
        return v


class PPO:
    def __init__(self, policy: AC, memory: ContiguousRingBuffer,
                 device: th.device, opts: PpoSettings):
        self.policy = policy
        self.device = device
        self.memory = memory
        self.opts = opts

        self.optimizer = th.optim.Adam(policy.parameters(),
                                       lr=2.5e-4,
                                       eps=1e-5)

    def act(self, states: np.ndarray, train: bool = True,
            deterministic: bool = None):
        if deterministic is None:
            deterministic = (not train)
        with th.no_grad():
            states = th.as_tensor(states).to(self.device)
            if train:
                # actions, values, log_probs
                a, v, l = self.policy.forward(states, deterministic)
                return a.cpu().numpy(), v.cpu().numpy(), l.cpu().numpy()
            else:
                a = self.policy.action(states, deterministic)
                return a.cpu().numpy()

    def evaluate(self, states: np.ndarray, actions: np.ndarray):
        states = th.as_tensor(states).to(self.device)
        actions = th.as_tensor(actions).to(self.device)

        v = self.policy.value(states)
        d = self.policy.action_dist(states)
        lp = d.log_prob(actions)
        e = d.entropy()
        return (v, lp, e)

    def compute_loss(self,
                     obs, act,
                     old_lp,
                     new_v, new_lp,
                     ent,
                     adv, ret, info={}):
        # Sanitize args.
        obs = th.as_tensor(obs).to(self.device)
        act = th.as_tensor(act).to(self.device)
        old_lp = th.as_tensor(old_lp).to(self.device)
        adv = th.as_tensor(adv).to(self.device)
        ret = th.as_tensor(ret).to(self.device)

        opts = self.opts
        # Normalize advantage
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ratio between old and new policy, should be one at the first
        # iteration
        ratio = th.exp(new_lp - old_lp)

        # clipped surrogate loss
        policy_loss_1 = adv * ratio
        policy_loss_2 = (
            adv * th.clamp(ratio, 1 - opts.clip_range, 1 + opts.clip_range))
        policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

        # Value loss using the TD(gae_lambda) target
        # NOTE(ycho): flatten to ensure mse_loss not over axes
        # TODO(ycho): investigate why value loss is
        # so much smaller for this impl. compared to
        # the one from stable_baselines3
        value_loss = F.mse_loss(ret.flatten(), new_v.flatten())

        # Entropy loss favor exploration
        entropy_loss = -th.mean(ent)

        info['policy_loss'] = policy_loss
        info['entropy_loss'] = entropy_loss
        info['value_loss'] = value_loss

        loss = (policy_loss + opts.entropy_weight *
                entropy_loss + opts.value_weight * value_loss)
        return loss


class ExponentialMovingGaussian(object):
    """
    Exponentially weighted moving gaussian.
    """

    def __init__(self, alpha=0.01, eps=1e-8):
        self.alpha = alpha
        self.eps = eps
        self.mean = 0.0
        self.var = 0.0
        self.count = 0

    def add(self, value: np.ndarray):
        if self.count == 0:
            self.mean = np.copy(value)
            self.var = np.zeros_like(value)
        else:
            alpha = self.alpha
            diff = value - self.mean
            incr = alpha * diff
            self.mean += incr
            self.var = (1 - alpha) * (self.var + diff * incr)
        self.count += 1

    def normalize(self, value: np.ndarray):
        std = np.sqrt(self.var)
        if self.count == 0 or np.equal(std, 0).any():
            return value
        return (value - self.mean) / (std + self.eps)

    def reset(self):
        self.count = 0

    def params(self):
        return {
            'mean': self.mean,
            'var': self.var,
            'count': self.count
        }

    def load(self, params):
        self.mean = params['mean']
        self.var = params['var']
        self.count = params['count']


class Callback(object):
    def __init__(self, period: int, callback):
        self.period = period
        self.last_call = 0
        self.callback = callback

    def on_step(self, step: int):
        if step < self.last_call + self.period:
            return
        self.callback(step)
        self.last_call = step


class SaveCallback(Callback):
    def __init__(self, period: int, ckpt_path: str, data_fn):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            ckpt_path.mkdir(parents=True, exist_ok=True)
        self.ckpt_path = ckpt_path
        self.data_fn = data_fn
        super().__init__(period, self._save)

    def _save(self, step: int):
        print('saving ... @ {}'.format(step))
        out_file = self.ckpt_path / 'model-{:05d}.zip'.format(step)
        th.save(self.data_fn(), str(out_file))


def train(opts: Settings):
    # === INSTANTIATE ENVIRONMENT ===

    # If `opts.subproc==True`, invoke gym.make() in a subprocess,
    # and treat the resultant instance as a `gym.Env`.
    subproc_gym_make = subproc(gym.make)
    gym_make = subproc_gym_make if opts.subproc else gym.make

    def get_env(index: int):
        env = gym_make('LunarLanderContinuous-v2')
        # env = gym.make('MountainCarContinuous-v0')
        # env = gym.make('LunarLanderContinuous-v2')
        env.seed(index)
        env.reset()
        return env

    env = MultiEnv(get_env, opts.num_envs)
    entry_type = [
        ('state', env.observation_space.dtype, env.observation_space.shape),
        ('action', env.action_space.dtype, env.action_space.shape),
        ('reward', np.float32, (1,)),
        # ('state1', env.observation_space.dtype, env.observation_space.shape),
        ('done', np.bool, (1,)),
        ('value', np.float32, (1,)),
        ('log_prob', np.float32, env.action_space.shape)
    ]

    # === NORMALIZERS FOR INPUTS ===
    reward_normalizer = ExponentialMovingGaussian(
        alpha=opts.reward_normalizer_alpha)
    state_normalizer = ExponentialMovingGaussian(
        alpha=opts.state_normalizer_alpha)

    # === INSTANTIATE MEMORY ===
    memory = ContiguousRingBuffer(
        capacity=opts.update_steps,
        dims=(opts.num_envs,),
        dtype=entry_type
    )

    # === INSTANTIATE POLICY ===
    # FIXME(ycho): Instead of assuming 1D box spaces,
    # explicitly wrap envs with flatten()...
    device = th.device(opts.device)
    policy = AC(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        opts.ac).to(device)

    # === INSTANTIATE AGENT ===
    ppo = PPO(policy, memory, device, opts.ppo)

    # === TRAIN ===
    states = env.reset()
    dones = np.full((opts.num_envs, 1), False, dtype=np.bool)
    returns = np.zeros(opts.num_envs, dtype=np.float32)

    # === LOGGER ===
    writer = SummaryWriter()
    writer.add_graph(policy, th.as_tensor(states).to(device))

    # === CALLBACKS ===
    save_cb = SaveCallback(
        opts.save_steps,
        opts.ckpt_path,
        lambda: {
            'settings': opts,
            'state_dict': policy.state_dict(),
            'reward_normalizer': reward_normalizer.params(),
            'state_normalizer': state_normalizer.params()})

    # === VARIABLES FOR DEBUGGING / LOG TRACKING ===
    reset_count = 0
    start_time = time.time()

    # === START TRAINING ===
    step = 0
    while step < opts.max_steps:
        # Reset any env that has reached termination.
        # FIXME(ycho): assumes isinstance(env, MultiEnv), of course.
        for i in range(opts.num_envs):
            if not dones[i]:
                continue
            states[i][:] = env.envs[i].reset()
            returns[i] = 0.0
            reset_count += 1

        # NOTE(ycho): Workaround for the current limitation of `MultiEnv`.
        # action = [env.action_space.sample() for _ in range(opts.num_envs)]
        # sanitize `states` arg.
        states = np.asarray(states).astype(np.float32)

        # Add states to stats for normalization.
        for s in states:
            state_normalizer.add(s)

        # Normalize states in-place.
        states = state_normalizer.normalize(states)
        states = np.clip(states, -10.0, 10.0)  # clip to +-10 stddev

        with th.no_grad():
            action, value, log_prob = ppo.act(states, True)

        # NOTE(ycho): Clip action within valid domain...
        clipped_action = np.clip(
            action, env.action_space.low, env.action_space.high)

        # Step according to above action.
        out = env.step(clipped_action)

        # Format entry.
        nxt_states, rewards, dones, _ = out

        # Add rewards to stats for normalization.
        # returns[np.asarray(dones).reshape(-1).astype(np.bool)] = 0.0
        returns = returns * opts.gae.gamma + np.reshape(rewards, -1)
        # NOTE(ycho): collect stats on `returns` instead of `rewards`.
        # for r in rewards:
        #    reward_normalizer.add(r)
        for r in returns:
            reward_normalizer.add(r)

        # Train if buffer full ...
        if memory.is_full:
            writer.add_scalar(
                'reward_mean', reward_normalizer.mean, global_step=step)
            writer.add_scalar(
                'reward_var', reward_normalizer.var, global_step=step)
            writer.add_scalar(
                'log_std',
                policy.log_std.detach().cpu().numpy()[0],
                global_step=step)
            writer.add_scalar(
                'fps', step / (time.time() - start_time), global_step=step)

            print('== step {} =='.format(step))
            # Log reward before overwriting with normalized values.
            print('rew = mean {} min {} max {} std {}'.format(
                memory['reward'].mean(),
                memory['reward'].min(),
                memory['reward'].max(),
                memory['reward'].std()))
            # print('rm {} rv {}'.format(reward_normalizer.mean,
            #                           reward_normalizer.var))

            # NOTE(ycho): States have already been normalized,
            # since those states were utilized as input for PPO action.
            # After that, the normalized states were inserted in memory.
            # memory['state'] = state_normalizer.normalize(memory['state'])

            # NOTE(ycho): I think it's fine to delay reward normalization to this point.
            # memory['reward'] = reward_normalizer.normalize(memory['reward'])
            # NOTE(ycho): maybe the proper thing to do is:
            # memory['reward'] = (memory['reward'] - reward_normalizer.mean) / np.sqrt(return_normalizer.var)
            memory['reward'] /= np.sqrt(reward_normalizer.var)
            memory['reward'] = np.clip(memory['reward'], -10.0, 10.0)

            # Create training data slices from memory ...
            dones = np.asarray(dones).reshape(opts.num_envs, 1)
            advs, rets = gae(memory, value, dones, opts.gae)
            # print('std = {}'.format(ppo.policy.log_std.exp()))

            ucount = 0
            info = None
            for _ in range(opts.num_epochs):
                for i in range(0, len(memory), opts.batch_size):
                    # Prepare current minibatch dataset ...
                    exp = memory[i: i + opts.batch_size]
                    act = exp['action']
                    obs = exp['state']
                    old_lp = exp['log_prob']
                    # old_v = exp['value'] # NOTE(ycho): unused
                    adv = advs[i: i + opts.batch_size]
                    ret = rets[i: i + opts.batch_size]

                    # Evaluate what had been done ...
                    # NOTE(ycho): wouldn't new_v == old_v
                    # and new_lp == old_lp for the very first one in the batch??
                    # hmm ....
                    new_v, new_lp, entropy = ppo.evaluate(
                        obs.copy(), act.copy())

                    info_i = {}
                    loss = ppo.compute_loss(
                        obs.copy(),
                        act.copy(),
                        old_lp.copy(),
                        new_v, new_lp, entropy, adv, ret, info_i)

                    # NOTE(ycho): Below, only required for logging
                    if True:
                        with th.no_grad():
                            if info is None:
                                info = info_i
                            else:
                                for k in info.keys():
                                    info[k] += info_i[k]
                        ucount += 1

                    # Optimization step
                    ppo.optimizer.zero_grad()
                    loss.backward()
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(
                        ppo.policy.parameters(), opts.ppo.max_grad_norm)
                    ppo.optimizer.step()

            for k, v in info.items():
                writer.add_scalar(
                    k, v.detach().cpu().numpy() / ucount, global_step=step)

            # Empty the memory !
            memory.reset()

        # Append to memory.
        entry = list(zip(*(states, action, rewards,
                           # nxt_states,
                           dones, value, log_prob)))
        memory.append(entry)

        # Cache `states`, update steps and continue.
        states = nxt_states
        step += opts.num_envs

        save_cb.on_step(step)

    writer.close()

    # Save ...
    th.save({
        'settings': opts,
        'state_dict': policy.state_dict(),
        'reward_normalizer': reward_normalizer.params(),
        'state_normalizer': state_normalizer.params()
    }, opts.model_path)


def test(opts: Settings):
    # == CREATE ENV ===
    env = gym.make('LunarLanderContinuous-v2')

    # === NORMALIZE INPUTS ===
    state_normalizer = ExponentialMovingGaussian()

    model = th.load(opts.model_path)
    # print(model['settings'])

    #print('opts', opts)
    #opts = replace(opts, **asdict(model['settings']))
    #print('->opts', opts)

    # === INSTANTIATE POLICY ===
    # FIXME(ycho): Instead of assuming 1D box spaces,
    # explicitly wrap envs with flatten()...
    device = th.device(opts.device)
    policy = AC(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        opts.ac).to(device)

    # === LOAD ===
    policy.load_state_dict(model['state_dict'])
    policy.eval()

    state_normalizer.load(model['state_normalizer'])

    # === INSTANTIATE AGENT ===
    ppo = PPO(policy, None, device, opts.ppo)

    state = env.reset()

    done = False
    episode_reward = 0.0
    while True:
        if done:
            print('episode reward={}'.format(episode_reward))
            episode_reward = 0.0
            state = env.reset()
        env.render()
        state = np.asarray(state).astype(np.float32)
        state = state_normalizer.normalize(state)
        action = ppo.act(state, False, True)

        clipped_action = np.clip(
            action, env.action_space.low, env.action_space.high)
        out = env.step(action)
        state, reward, done, _ = out
        episode_reward += reward
        time.sleep(0.01)


def main():
    parser = ArgumentParser()
    parser.add_arguments(Settings, dest='opts')
    argcomplete.autocomplete(parser)
    opts = parser.parse_args().opts
    if opts.train:
        train(opts)
    else:
        test(opts)


if __name__ == '__main__':
    main()
