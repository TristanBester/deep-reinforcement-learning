import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam


def init_mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    """Initialize a feedforward neural network with given sizes."""
    layers = []

    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        act_func = activation if i < len(sizes) - 2 else output_activation
        layers.append(act_func())
    return nn.Sequential(*layers)


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    batch_size = 5000

    logits_net = init_mlp(sizes=[obs_dim] + [32] + [n_actions])
    optimizer = Adam(params=logits_net.parameters(), lr=1e-2)

    def get_policy(obs):
        """Given a state, return the action distribution."""
        logits = logits_net(obs)
        return Categorical(logits=logits)

    def get_action(obs):
        """Given a state, sample the action distribution and return result."""
        dist = get_policy(obs)
        action = dist.sample().item()
        return action

    def compute_loss(obs, act, weights):
        """Loss function with gradient equal to policy gradient."""
        action_distributions = get_policy(obs)
        selected_action_log_prop = action_distributions.log_prob(act)
        return -(selected_action_log_prop * weights).mean()

    def train_one_epoch():
        """Train the policy for one epoch."""
        batch_obs = []
        batch_acts = []
        # This is for R(tau) weighting in policy gradient
        batch_weights = []
        batch_rets = []
        batch_lens = []

        obs, _ = env.reset()
        done = False
        ep_rews = []

        # finished_rendering_this_epoch = False

        while True:
            # if not finished_rendering_this_epoch:
            #     env.render()

            batch_obs.append(obs.copy())

            obs = torch.as_tensor(obs, dtype=torch.float32)
            act = get_action(obs)
            obs, rew, done, _, _ = env.step(act)

            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                ep_ret = sum(ep_rews)
                ep_len = len(ep_rews)

                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # The weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # Reset episode-specific variables
                (obs, _), done, ep_rews = env.reset(), False, []

                # finished_rendering_this_epoch = True

                if len(batch_obs) > batch_size:
                    break

        optimizer.zero_grad()
        batch_loss = compute_loss(
            obs=torch.as_tensor(batch_obs, dtype=torch.float32),
            act=torch.as_tensor(batch_acts, dtype=torch.float32),
            weights=torch.as_tensor(batch_weights, dtype=torch.float32),
        )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

    for i in range(100):
        batch_loss_, batch_rets_, batch_lens_ = train_one_epoch()
        print(
            f"Epoch: {i}\tLoss: {batch_loss_.item():.3f}\t"
            f"Return: {np.mean(batch_rets_):.3f}\t"
            f"Lengths: {np.mean(batch_lens_):.3f}"
        )
