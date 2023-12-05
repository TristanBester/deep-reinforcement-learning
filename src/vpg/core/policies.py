import torch
from torch import nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal


def mlp_factory(
    sizes: list[int],
    activation: nn.Module,
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
        logits = self.network(torch.as_tensor(obs, dtype=torch.float32))
        return Categorical(logits=logits)

    def logp_act(self, obs, act) -> float:
        """Return the log prob of the given action."""
        return self.forward(obs).log_prob(act)


class ContinuousPolicy(nn.Module):
    """Policy approximator for environments with continuous action spaces."""

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation) -> None:
        """Constructor."""
        super().__init__()
        layer_sizes = [obs_dim] + hidden_sizes + [act_dim]
        self.mu_net = mlp_factory(sizes=layer_sizes, activation=activation)

        # We learn log(std) as it does not have constraints on the values it can take
        # whereas the std must be positive making optimisation more complicated
        self.log_std = nn.Parameter(torch.full(act_dim, -0.5, dtype=torch.float32))

    def forward(self, obs) -> Normal:
        """Return the action probability distribution conditioned on the given obsevation."""
        mu = self.mu_net(torch.as_tensor(obs, dtype=torch.float32))
        std = torch.exp(self.log_std)
        return Normal(loc=mu, scale=std)

    def logp_act(self, obs, act) -> float:
        """Return the log prob of the given action."""
        return self.forward(obs).log_prob(act)
