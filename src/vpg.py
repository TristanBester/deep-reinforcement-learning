import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.spaces import Box, Discrete
from torch import nn
from torch.optim import Adam

from core.buffer import GAEBuffer
from core.loss import pg_loss
from core.policies import ContinuousPolicy, DiscretePolicy, mlp_factory


def train_vpg(
    env: gym.Env,
    hidden_size: int,
    hidden_layers: int,
    activation: nn.Module = nn.ReLU,
    steps_per_epoch: int = 500,
    n_epochs: int = 5000,
    pi_lr: float = 1e-4,
    vf_lr: float = 1e-4,
    gamma=0.99,
    lambda_=0.95,
    max_ep_len=100,
    n_vf_iter=10,
    seed=0,
):
    """Vanilla Policy Gradient with GAE-Lambda for advantage estimation."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    obs_dim = env.observation_space.shape[0]
    hidden_sizes = [hidden_size] * hidden_layers

    if isinstance(env.action_space, Discrete):
        act_dim = env.action_space.n
        pi = DiscretePolicy(obs_dim, act_dim, hidden_sizes, activation)
    if isinstance(env.action_space, Box):
        act_dim = env.action_space.shape
        pi = ContinuousPolicy(obs_dim, act_dim, hidden_sizes, activation)

    vf_net_layers = [obs_dim] + hidden_sizes + [1]
    vf = mlp_factory(sizes=vf_net_layers, activation=activation)
    buffer = GAEBuffer(
        obs_dim, act_dim, size=steps_per_epoch, gamma=gamma, lambda_=lambda_
    )

    pi_optimiser = Adam(params=pi.parameters(), lr=pi_lr)
    vf_optimiser = Adam(params=vf.parameters(), lr=vf_lr)

    def rollout():
        """Rollout policy - simulate agent interaction with environment & record results."""
        buffer.reset()
        obs, _ = env.reset()
        ep_len, ep_return = 0, 0

        for t in range(steps_per_epoch):
            action = pi(obs).sample().item()
            logp_action = pi.logp_act(obs, torch.as_tensor(action, dtype=torch.float32))
            value = vf(torch.as_tensor(obs, dtype=torch.float32))

            obs_next, reward, done, _, _ = env.step(action)

            buffer.store(obs, action, reward, value, logp_action)
            ep_len += 1
            ep_return += reward

            obs = obs_next

            episode_terminated = ep_len == max_ep_len
            epoch_done = t == (steps_per_epoch - 1)

            if done or episode_terminated or epoch_done:
                if episode_terminated or epoch_done:
                    # Bootstrap state value as episode terminated early
                    v_term = (
                        vf(torch.as_tensor(obs, dtype=torch.float32)).detach().numpy()
                    )
                else:
                    # By definition terminal state
                    v_term = 0

                obs, _ = env.reset()
                ep_len, ep_return = 0, 0
                buffer.complete_trajectory(v_term=v_term)

    for epoch in range(n_epochs):
        rollout()

        obs, acts, _, advs, rets, logp_acts = buffer.get_data()

        logp_acts = pi.logp_act(
            torch.as_tensor(obs, dtype=torch.float32),
            torch.as_tensor(acts, dtype=torch.float32),
        )

        # Update the policy network
        pi_optimiser.zero_grad()
        loss_pi = pg_loss(
            logp_act=logp_acts,
            phi=torch.as_tensor(advs, dtype=torch.float32),
        )
        loss_pi.backward()
        pi_optimiser.step()

        # Update the value function network
        for _ in range(n_vf_iter):
            vf_optimiser.zero_grad()
            value_preds = vf(torch.as_tensor(obs, dtype=torch.float32)).flatten()
            loss_vf = F.mse_loss(
                value_preds, torch.as_tensor(rets, dtype=torch.float32)
            )
            loss_vf.backward()
            vf_optimiser.step()

        if epoch % 100 == 0:
            vals = []
            for _ in range(100):
                other_env = gym.make("CartPole-v1")

                obs, _ = other_env.reset()
                done = False
                ep_ret = 0

                while not done:
                    # other_env.render()
                    action = pi(obs).sample().item()

                    obs_next, reward, done, _, _ = other_env.step(action)
                    obs = obs_next
                    ep_ret += reward

                vals.append(ep_ret)

            val = np.mean(ep_ret)

            print("RETURN: ", val)

            if val > 300:
                input("Start...")
                for _ in range(100):
                    other_env = gym.make("CartPole-v1", render_mode="human")

                    obs, _ = other_env.reset()
                    done = False
                    ep_ret = 0

                    while not done:
                        other_env.render()
                        action = pi(obs).sample().item()

                        obs_next, reward, done, _, _ = other_env.step(action)
                        obs = obs_next
                        ep_ret += reward

            print()

        print(epoch, rets.sum())


if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    train_vpg(env=env, hidden_layers=2, hidden_size=32)
