from collections import namedtuple

import gymnasium as gym
import torch
import torch.nn.functional as F
from scipy.sparse.linalg import cg
from torch import nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam

env = gym.make("CartPole-v1")
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

max_d_kl = 0.01


def train(env, n_epochs=100, n_rollouts=10):
    """Train the agent on the environment for n_epochs."""
    #

    Rollout = namedtuple("Rollout", ["obs", "actions", "rewards", "next_obs"])

    for _ in range(n_epochs):
        rollouts = []

        for _ in range(n_rollouts):
            obs, _ = env.reset()
            done = False
            samples = []

            while not done:
                with torch.no_grad():
                    action = get_action(obs)
                    next_obs, reward, done, _, _ = env.step(action)

                    samples.append((obs, action, reward, next_obs))
                    obs = next_obs

            obs, actions, rewards, next_obs = zip(*samples)
            obs = torch.stack([torch.from_numpy(o) for o in obs], dim=0).float()
            next_obs = torch.stack(
                [torch.from_numpy(o) for o in next_obs], dim=0
            ).float()
            actions = torch.as_tensor(actions).unsqueeze(1)
            rewards = torch.as_tensor(rewards).unsqueeze(1)
            rollouts.append(Rollout(obs, actions, rewards, next_obs))

        update_agent(rollout)


actor = nn.Sequential(
    nn.Linear(obs_size, 32), nn.ReLU(), nn.Linear(32, n_actions), nn.Softmax()
)
critic = nn.Sequential(
    nn.Linear(obs_size, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
)
critic_optimiser = Adam(params=critic.parameters(), lr=1e-3)


def get_action(obs):
    """Get action"""
    obs = torch.tensor(obs).float().unsqueeze(0)
    action_dist = Categorical(probs=actor(obs))
    return action_dist.sample().item()


def update_critic(advantages):
    """Update network"""
    loss = 0.5 * (advantages**2).mean()
    critic_optimiser.zero_grad()
    loss.backward()
    critic_optimiser.step()


def update_agent(rollouts):
    obs = torch.cat([r.obs for r in rollouts], dim=0)
    actions = torch.cat([r.actions for r in rollouts], dim=0)

    advantages = [estimate_advantages(o, n_o[-1], r) for o, _, n_o in rollouts]
    advantages = torch.cat(advantages, dim=0).flatten()

    # Normalise advantages to improve convergence
    advantages = (advantages - advantages.mean()) / advantages.std()

    update_critic(advantages)

    action_dist = actor(obs)
    action_dist = torch.distributions.utils.clamp_probs(action_dist)
    probabilities = action_dist[range(action_dist.shape[0]), actions]

    L = surrogate_loss(probabilities, probabilities.detach(), advantages)
    KL = kl_div(distribution, distribution)

    parameters = list(actor.parameters())

    g = flat_grad(L, parameters, retain_graph=True)
    d_kl = flat_grad(KL, parameters, create_graph=True)

    def HVP(v):
        return flat_grad(d_kl @ v, parameters, retain_graph=True)

    search_dir = conjugate_gradient(HVP, g)
    max_length = torch.sqrt(2 * max_d_kl / (search_dir @ HVP(search_dir)))
    max_step = max_length * search_dir

    def criterion(step):
        apply_update(step)

        with torch.no_grad():
            distribution_new = actor(obs)
            distribution_new = torch.distributions.utils.clamp_probs(distribution_new)
            probabilities_new = distribution_new[
                range(distribution_new.shape[0]), actions
            ]

            L_new = surrogate_loss(probabilities_new, probabilities, advantages)
            KL_new = kl_div(distribution, distribution_new)

        L_improvement = L_new - L

        if L_improvement > 0 and KL_new <= max_d_kl:
            return True

        apply_update(-step)
        return False

    i = 0
    while not criterion((0.9**i) * max_step) and i < 10:
        i += 1


def estimate_advantages(obs, last_obs, rewards):
    values = critic(obs)
    last_value = critic(last_obs.unsqueeze(0))
    next_values = torch.zeros_like(rewards)

    for i in reversed(range(rewards.shape[0])):
        last_value = next_values[i] = rewards[i] + 0.99 * last_value
    advantages = next_values - values
    return advantages


def surrogate_loss(new_probabilities, old_probabilies, advantages):
    return (new_probabilities / old_probabilies * advantages).mean()


def kl_div(p, q):
    return F.kl_div(p, q)


def flat_grad(y, x, retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True

    g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
    g = torch.cat([t.view(-1) for t in g])
    return g


def conjugate_gradient(A, b, delta=0):
    return cg(A, b, x0=torch.zeros_like(b), max_iter=1000)


def apply_update(grad_flattened):
    n = 0
    for p in actor.parameters():
        # Num elements in the tensor
        numel = p.numel()
        g = grad_flattened[n : n + numel].view(p.shape)
        p.data += g
        n += numel
