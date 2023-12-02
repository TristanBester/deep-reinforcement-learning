import gymnasium as gym
import torch.nn.functional as F
from factories import Actor, Critic
from torch.optim import SGD

from core.buffer import Buffer


def work():
    n_epochs = 10000
    n_steps_per_epoch = 1000
    max_ep_len = 100
    iter_v_train = 100

    def rollout():
        """Rollout policy. Simulate agent interaction with environment & record results."""
        obs, _ = env.reset()
        ep_len, ep_return = 0, 0

        for t in range(n_steps_per_epoch):
            action, logp_action = actor.get_action_with_proba(obs)
            value = critic.get_value(obs)

            obs_next, reward, done, _, _ = env.step(action)

            buffer.store(obs, action, reward, value, logp_action)
            ep_len += 1
            ep_return += reward

            obs = obs_next

            episode_terminated = ep_len == max_ep_len
            epoch_done = t == (n_steps_per_epoch - 1)

            if done or episode_terminated or epoch_done:
                if episode_terminated or epoch_done:
                    # Bootstrap state value as episode terminated early
                    v_term = critic.get_value(obs)
                else:
                    # By definition terminal state
                    v_term = 0

                obs, _ = env.reset()
                ep_len, ep_return = 0, 0
                buffer.complete_trajectory(v_term=v_term)

    def loss_pi(obs, act, adv, logp_act):
        # TODO: Implement this as a torch loss so you can follow general code style
        _, logp_act = actor.get_action_with_proba(obs)
        # We must invert the sign as we want to turn the maximisation problem to minimisation for torch
        loss = -1 * (logp_act * adv).mean()
        return loss

    def update():
        """Optimise the actor and critic networks."""
        obs, act, adv, logp_act, rets = buffer.get_data()

        actor_optimiser.zero_grad()
        loss = loss_pi()
        loss.backward()
        actor_optimiser.step()

        for _ in range(iter_v_train):
            critic_optimiser.zero_grad()
            value_estimates = critic(obs)
            loss = F.mse_loss(value_estimates, rets)
            loss.backward()
            critic_optimiser.step()

    env = gym.make("CartPole-v1")

    actor = Actor()
    critic = Critic()
    buffer = Buffer()

    actor_optimiser = SGD(params=actor.pi.parameters(), lr=1e-3)
    critic_optimiser = SGD(params=critic.v.parameters(), lr=1e-3)

    for epoch in range(n_epochs):
        rollout()
        update()
