import torch
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
        x = torch.as_tensor(obs, dtype=torch.float32)
        return self.network(x)
