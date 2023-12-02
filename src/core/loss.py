import torch


def pg_loss(logp_act: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Loss function used in policy gradient methods.

    Args:
        logp_act (torch.Tensor): The log probability of each action.
        phi (torch.Tensor): The value of the weighting term in the VPG
            policy gradient expectation. A number of varients may be
            for phi such as GAE.

    Returns:
        torch.Tensor: Negative value of the policy gradient loss function. The
            negative value is added to convert from maximisation to minimisation.
    """
    return -1 * (logp_act * phi).mean()
