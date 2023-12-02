import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def mlp_factory(
    sizes: list[int],
    activation: nn.Module,
    output_activation: nn.Module = nn.Identity,
) -> nn.Sequential:
    """Create an MLP using the provided specification."""


class DiscretePolicy(nn.Module):
    """Policy approximator for environments with discrete action spaces."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: list[int],
        activation: nn.Module,
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
        logits = self.network(obs)
        return Categorical(logits=logits)


class Actor:
    """Encapsulate the behaviour of an actor in actor-critic method."""

    def __init__(self, **kwargs: dict) -> None:
        self.pi = DiscretePolicy(**kwargs)

    def get_action_with_proba(self, obs):
        """Return an action sampled from the action distribution."""
        action_dist = self.pi(obs)
        action = action_dist.sample()
        logp_act = action_dist.log_prob(action)
        return action, logp_act


class Critic:
    """Encapsulate the behaviour of a critic in actor critic method."""

    def __init__(self, **kwargs: dict) -> None:
        self.v = mlp_factory(**kwargs)

    def get_value(self, obs):
        return self.v(obs)
