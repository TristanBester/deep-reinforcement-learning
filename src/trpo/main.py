import gymnasium as gym
import numpy as np
import scipy
import torch
import torch.nn.functional as F
from gymnasium.spaces import Box, Discrete
from scipy.sparse.linalg import cg
from torch import nn
from torch.distributions import Categorical


def mlp_factory(
    sizes: list[int],
    activation: nn.Module = nn.ReLU,
    output_activation: nn.Module = nn.Identity,
) -> nn.Sequential:
    """Create an MLP using the provided specification."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        activation_fn = activation if i < len(sizes) - 1 else output_activation
        layers.append(activation_fn())
    return nn.Sequential(*layers)


class DiscretePolicy(nn.Module):
    """Policy approximator for environments with discrete action spaces."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: list[int],
        activation: nn.Module = nn.ReLU,
    ) -> None:
        """Constructor."""
        super().__init__()
        layer_sizes = [obs_dim] + hidden_sizes + [act_dim]
        self.network = mlp_factory(
            sizes=layer_sizes,
            activation=activation,
        )

    def forward(self, obs) -> Categorical:
        """Return the action probability distribution conditioned on the given observation."""
        logits = self.network(torch.as_tensor(obs, dtype=torch.float32))
        return Categorical(logits=logits)


class Actor:
    def __init__(self, obs_dim, act_dim, is_discrete) -> None:
        self.network = DiscretePolicy(obs_dim, act_dim, hidden_sizes=[32])

    def get_action(self, obs):
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32)
            action_dist = self.network(x)
            return action_dist.sample().item()

    def get_action_probas(self, obs, actions):
        o = torch.as_tensor(obs, dtype=torch.float32)
        a = torch.as_tensor(actions, dtype=torch.float32)

        action_dist = self.network(o)
        return torch.exp(action_dist.log_prob(a))

    def get_distribution(self, obs):
        o = torch.as_tensor(obs, dtype=torch.float32)
        action_dist = self.network(o)
        return action_dist.probs


class Critic:
    def __init__(self, obs_dim):
        layer_sizes = [obs_dim] + [32] + [1]
        self.network = mlp_factory(sizes=layer_sizes)

    def get_value(self, obs):
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32)
            return self.network(x)


class GAEBuffer:
    """GAE Buffer."""

    def __init__(self, gamma, lambda_) -> None:
        self.gamma = gamma
        self.lambda_ = lambda_

        # Main buffers
        self.obs_buf = []
        self.action_buf = []
        self.obs_next_buf = []
        self.gae_buf = []
        self.returns_buf = []

        # Current trajectory buffers
        self.obs_t = []
        self.action_t = []
        self.obs_next_t = []
        self.reward_t = []
        self.value_t = []

    def start_trajectory(self) -> None:
        """Reset the trajectory buffers."""
        self.obs_t.clear()
        self.action_t.clear()
        self.obs_next_t.clear()
        self.reward_t.clear()
        self.value_t.clear()

    def end_trajectory(self) -> None:
        """Compute returns and GAE. Store trajectory in main buffers."""
        r = np.array(self.reward_t)
        v = np.array(self.value_t)
        v_next = np.array(self.value_t[1:] + [0])

        delta = r + self.gamma * v_next - v
        advantages = self.discount_cumsum(delta, self.gamma * self.lambda_)
        returns = self.discount_cumsum(self.reward_t, self.gamma)

        self.obs_buf += self.obs_t
        self.action_buf += self.action_t
        self.obs_next_buf += self.obs_next_t
        self.gae_buf += [advantages] * len(self.action_t)
        self.returns_buf += [returns] * len(self.action_t)

    def store_transition(self, obs, action, obs_next, reward, value) -> None:
        """Store the transition information in trajectory buffers."""
        self.obs_t.append(obs)
        self.action_t.append(action)
        self.obs_next_t.append(obs_next)
        self.reward_t.append(reward)
        self.value_t.append(value)

    def discount_cumsum(self, x: np.array, discount: float) -> np.array:
        """Compute discounted cumulative sums of vectors."""
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def get_advantages(self) -> np.array:
        """Return the advantages."""
        return np.array(self.gae_buf).squeeze()

    def get_actions(self) -> np.array:
        """Return actions."""
        return np.array(self.action_buf)

    def get_obs(self) -> np.array:
        """Return obs."""
        return np.array(self.obs_buf)


def rollout(env, actor, critic, n_episodes: int = 100) -> GAEBuffer:
    """Rollout the given policy and store trajectories."""
    buffer = GAEBuffer(gamma=0.99, lambda_=0.95)

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False

        buffer.start_trajectory()
        while not done:
            action = actor.get_action(obs)
            value = critic.get_value(obs)
            obs_next, done, reward, _, _ = env.step(action)

            buffer.store_transition(obs, action, obs_next, reward, value)
            obs = obs_next
        buffer.end_trajectory()
    return buffer


def surrogate_loss(old_action_probas, current_action_probas, advantages):
    """Estimate loss value with importance sampling."""
    return (old_action_probas / current_action_probas * advantages).mean()


def hessian_vector_product(grad_kl, x, parameters):
    product = grad_kl @ x
    result = torch.autograd.grad(
        product, parameters, retain_graph=True, create_graph=True
    )
    return torch.cat([t.view(-1) for t in result])


def solve_search_direction(g, grad_kl, parameters):
    x = torch.zeros_like(g)
    r = g.clone()
    p = g.clone()

    i = 0
    while i < 100:
        print(p)

        AVP = hessian_vector_product(grad_kl, p, parameters)
        print(AVP)

        dot_old = r @ r
        alpha = dot_old / (p @ AVP)

        x_new = x + alpha * p

        if (x - x_new).norm() <= 0:
            return x_new

        i += 1
        r = r - alpha * AVP

        beta = (r @ r) / dot_old
        p = r + beta * p

        x = x_new
    return x


def train_trpo(
    env,
    hidden_layers,
    hidden_size,
    n_epochs: int = 100,
    n_steps_per_epoch: int = 4,
    seed: int = 0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    obs_dim = env.observation_space.shape[0]
    hidden_sizes = [hidden_size] * hidden_layers

    if isinstance(env.action_space, Discrete):
        act_dim = env.action_space.n
        actor = Actor(obs_dim, act_dim, is_discrete=True)
    # if isinstance(env.action_space, Box):
    #     act_dim = env.action_space.shape
    #     pi = ContinuousPolicy(obs_dim, act_dim, hidden_sizes)

    critic = Critic(obs_dim=obs_dim)

    for _ in range(n_epochs):
        buffer = rollout(env, actor=actor, critic=critic, n_episodes=5)

        obs = torch.as_tensor(buffer.get_obs(), dtype=torch.float32)
        actions = torch.as_tensor(buffer.get_actions(), dtype=torch.float32)
        advantages = torch.as_tensor(buffer.get_advantages(), dtype=torch.float32)

        old_action_probas = actor.get_action_probas(obs, actions)
        old_action_distribution = actor.get_distribution(obs)

        for _ in range(n_steps_per_epoch):
            current_action_probas = actor.get_action_probas(obs, actions)
            current_action_distribution = actor.get_distribution(obs)

            # Solve for g
            loss_l = surrogate_loss(
                old_action_probas, current_action_probas, advantages
            )
            g = torch.autograd.grad(
                loss_l, actor.network.parameters(), retain_graph=True
            )
            g = torch.concat([t.flatten() for t in g])

            # Solve for search direction
            loss_kl = F.kl_div(
                current_action_distribution,
                old_action_distribution,
                reduction="batchmean",
            )
            grad_kl = torch.autograd.grad(
                loss_kl,
                actor.network.parameters(),
                retain_graph=True,
                create_graph=True,
            )
            grad_kl = torch.concat([t.flatten() for t in grad_kl])

            search_direction = solve_search_direction(
                g, grad_kl, actor.network.parameters()
            )

            print(search_direction)

            0 / 0


if __name__ == "__main__":
    environment = gym.make("CartPole-v1")

    train_trpo(env=environment, hidden_layers=2, hidden_size=32)
