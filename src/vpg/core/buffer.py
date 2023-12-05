import numpy as np
import scipy


class GAEBuffer:
    """
    Buffer for storing trajectories experienced by a VPG agent interacting
    with the environment, using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(
        self, obs_dim: int, act_dim: int, size: int, gamma: float, lambda_: float
    ) -> None:
        """Constructor."""
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_act_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lambda_ = gamma, lambda_
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.obs_dim, self.act_dim = obs_dim, act_dim

    def reset(self):
        """Clear the buffer."""
        self.obs_buf = np.zeros(
            combined_shape(self.max_size, self.obs_dim), dtype=np.float32
        )
        self.act_buf = np.zeros(self.max_size, dtype=np.float32)
        self.adv_buf = np.zeros(self.max_size, dtype=np.float32)
        self.rew_buf = np.zeros(self.max_size, dtype=np.float32)
        self.ret_buf = np.zeros(self.max_size, dtype=np.float32)
        self.val_buf = np.zeros(self.max_size, dtype=np.float32)
        self.logp_act_buf = np.zeros(self.max_size, dtype=np.float32)
        self.ptr, self.path_start_idx = 0, 0

    def store(self, obs, action, reward, value, logp_act):
        """Store the given values in the appropriate buffers."""
        assert self.ptr < self.max_size, "Cannot insert into full buffer."
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = action
        self.rew_buf[self.ptr] = reward
        self.val_buf[self.ptr] = value
        self.logp_act_buf[self.ptr] = logp_act
        self.ptr += 1

    def complete_trajectory(self, v_term) -> None:
        """Compute relevant quantities at the end of an episode.

        This function should be called when the agent epsidode terminates.
        All data from the beginning of the active trajectory is processes.
        Advantage estimates are computed using GAE. The reward-to-go for each
        state is also computed.

        Args:
            v_term (float): Value of the terminal state, used for bootstapping
                in the event that a trajectory is terminated early.h
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], v_term)
        vals = np.append(self.val_buf[path_slice], v_term)

        # GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lambda_)

        # Reward-to-go for each state
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get_data(
        self,
    ) -> tuple[np.array, np.array, np.array, np.array, np.array, np.array]:
        """Return the data in the buffer."""
        return (
            self.obs_buf,
            self.act_buf,
            self.rew_buf,
            self.adv_buf,
            self.ret_buf,
            self.logp_act_buf,
        )


def combined_shape(length: int, shape: tuple | None = None) -> tuple:
    """Combine the shapes of the inputs"""
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def discount_cumsum(x: np.array, discount: float) -> np.array:
    """Compute discounted cumulative sums of vectors."""
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
