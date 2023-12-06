from dataclasses import dataclass, field

import torch


class Trajectory:
    def __init__(self) -> None:
        self._obs = []
        self._actions = []
        self._obs_next = []
        self._rewards = []

    def store(self, obs, action, obs_next, reward):
        self._obs.append(obs)
        self._actions.append(action)
        self._obs_next.append(obs_next)
        self._rewards.append(reward)

    @property
    def obs(self):
        return torch.as_tensor(self._obs, dtype=torch.float32)

    @property
    def obs_next(self):
        return torch.as_tensor(self._obs_next, dtype=torch.float32)

    @property
    def rewards(self):
        return torch.as_tensor(self._rewards, dtype=torch.float32)


# @dataclass
# class RolloutBuffer:
#     trajectories: list = field(default_factory=list)

#     def store(self, trajectory):
#         self.trajectories.append(trajectory)

#     @property
#     def obs(self):
#         return torch.concat(
#             [torch.as_tensor(t.obs, dtype=torch.float32) for t in self.trajectories],
#         )

#     @property
#     def actions(self):
#         return torch.concat(
#             [
#                 torch.as_tensor(t.actions, dtype=torch.float32)
#                 for t in self.trajectories
#             ],
#         )

#     @property
#     def obs_next(self):
#         return torch.concat(
#             [
#                 torch.as_tensor(t.obs_next, dtype=torch.float32)
#                 for t in self.trajectories
#             ],
#         )

#     @property
#     def rewards(self):
#         return torch.concat(
#             [
#                 torch.as_tensor(t.rewards, dtype=torch.float32)
#                 for t in self.trajectories
#             ],
#         )
