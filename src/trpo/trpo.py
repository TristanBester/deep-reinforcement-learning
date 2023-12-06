import warnings

import gymnasium as gym
import numpy as np
import scipy
import torch
import torch.nn.functional as F
from buffer import Trajectory
from gymnasium.spaces import Box, Discrete
from networks import Actor, Critic
from scipy.sparse.linalg import cg
from torch import nn
from torch.distributions import Categorical

warnings.filterwarnings("ignore")


def rollout(env, actor, n_episodes=10):
    # buffer = RolloutBuffer()
    trajectories = []

    for _ in range(n_episodes):
        trajectory = Trajectory()
        obs, _ = env.reset()
        done = False

        while not done:
            action = actor.get_action(obs)
            obs_next, done, reward, _, _ = env.step(action)

            trajectory.store(obs, action, obs_next, reward)
            obs = obs_next
        trajectories.append(trajectory)
    return trajectories


# def discount_cumsum(self, x: np.array, discount: float) -> np.array:
#     """Compute discounted cumulative sums of vectors."""
#     return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def discount_cumsum(vector, discount):
    wts = discount ** torch.arange(
        len(vector), dtype=torch.float64
    )  # pre-compute all discounts
    x = wts * vector  # weighted vector
    cum_sum = torch.cumsum(x, dim=0)  # forward cumsum
    re_cum_sum = x - cum_sum + cum_sum[-1:]  # reversed cumsum
    return re_cum_sum / wts


def compute_returns(trajectories, gamma):
    returns = []

    for t in trajectories:
        returns.append(discount_cumsum(t.rewards, gamma))
    return torch.cat(returns)


def compute_advantages(trajectories, critic, gamma, lambda_):
    advantages = []

    for t in trajectories:
        values = critic.get_value(t.obs)
        values_next = critic.get_value(t.obs_next)
        values_next[-1] = 0.0

        delta = t.rewards + gamma * values_next - values
        advantage = discount_cumsum(delta, gamma * lambda_)
        advantages.append(advantage)
    return torch.cat(advantages).flatten()


def train_trpo(
    env,
    hidden_layers,
    hidden_size,
    n_epochs: int = 100,
    n_steps_per_epoch: int = 4,
    gamma=0.99,
    lambda_=0.95,
    seed: int = 0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    actor = Actor(obs_dim, n_actions, is_discrete=True)
    critic = Critic(obs_dim)

    trajectories = rollout(env, actor, n_episodes=10)

    advantages = compute_advantages(trajectories, critic, gamma, lambda_)
    returns = compute_returns(trajectories, gamma)
    print(advantages)
    print(returns)


if __name__ == "__main__":
    environment = gym.make("CartPole-v1")
    train_trpo(environment, 2, 32)
