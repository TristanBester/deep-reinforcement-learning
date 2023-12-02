import gymnasium as gym
import numpy as np
import scipy
import torch
import torch.nn as nn
from gymnasium.spaces import Box, Discrete
from torch.distributions.categorical import Categorical


def mlp_factory(sizes, activation, output_activation=nn.Identity):
    """Create an MLP using the provided specification."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        activation_function = activation if i < len(sizes) - 1 else output_activation
        layers.append(activation_function)
    return nn.Sequential(*layers)


class MLPActor(nn.Module):
    """MLP Policy."""

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation) -> None:
        super().__init__()
        layer_sizes = [obs_dim] + hidden_sizes + [act_dim]
        self.logits_net = mlp_factory(sizes=layer_sizes, activation=activation)

    def forward(self, obs, act=None):
        """Forward."""
        pi = self._action_distribution(obs)
        logp_a = None
        if act is None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

    def _action_distribution(self, obs):
        """Compute the action probability distribution given the observation."""
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPCritic(nn.Module):
    """Thing."""

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        layer_sizes = [obs_dim] + hidden_sizes + [1]
        self.network = mlp_factory(sizes=layer_sizes, activation=activation)

    def forward(self, obs):
        """Thing."""
        return torch.squeeze(self.network(obs), -1)


class MLPActorCritic:
    """Thing."""

    def __init__(
        self, observation_space, action_space, hidden_sizes=[64, 64], activation=nn.ReLU
    ):
        obs_dim = observation_space.shape[0]

        if isinstance(action_space, Box):
            pass
        elif isinstance(action_space, Discrete):
            self.pi = MLPActor(
                obs_dim=obs_dim,
                act_dim=action_space.n,
                hidden_sizes=hidden_sizes,
                activation=activation,
            )
        self.v = MLPCritic(
            obs_dim=obs_dim, hidden_sizes=hidden_sizes, activation=activation
        )

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._action_distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class VPGBuffer:
    """Big man buffer."""

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lambda_=0.95):
        self.obs_buf = np.zeros((size,) + obs_dim)
        self.act_buf = np.zeros((size,) + act_dim)
        self.adv_buff = np.zeros(size, np.float32)
        self.rew_buff = np.zeros(size, np.float32)
        self.ret_buff = np.zeros(size, np.float32)
        self.val_buff = np.zeros(size, np.float32)
        self.logp_buff = np.zeros(size, np.float32)
        self.gamma, self.lambda_ = gamma, lambda_
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """Thing."""
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call at end of trajectory. Compute the GAE-lambda estimates as rewards-to-go
        for the optimisation of value function and policy.
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buff[path_slice], last_val)
        vals = np.append(self.vals_buff[path_slice], last_val)

        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buff[path_slice] = discount_cumsum(deltas, self.gamma * self.lambda_)

        self.ret_buff[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buff), np.std(self.adv_buff)
        self.adv_buff = (self.adv_buf - adv_mean) / adv_std
        data = dict(
            obs=self.obs_buff,
            act=self.act_buff,
            ret=self.ret_buff,
            adv=self.adv_buff,
            logp=self.logp_buff,
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


def vpg():
    env = gym.make("CartPole-v1")

    obs_space = env.observation_space
    act_space = env.action_space

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    ac = MLPActorCritic(observation_space=obs_space, action_space=act_space)
    buff = VPGBuffer(obs_dim=obs_dim, act_dim=act_dim, size=10000, 0.99, 0.95)

    
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    
    obs, _ = env.reset()
    ep_len, ep_ret = 0,0
    
    
    for epoch in range(100):
        for t in range(1000):
            a, v, logp_a = ac.step(obs)
            
            obs_next, reward, done, _, _ = env.step(a)
            
            ep_ret += reward 
            ep_len += 1 
            
            buff.store(obs, a, reward, v, logp_a)
            
            obs = obs_next
            
            terminal = done or ep_len == 1000
            epoch_done = t == 999
            
            if terminal or epoch_done:
                # if trajectory didn't reach terminal state, bootstrap value target
                if terminal or epoch_done:
                    _, v, _ = ac.step(torch.as_tensor(obs, dtype=torch.float32))
                else:
                    v = 0 
                    
                buff.finish_path(v)
                obs, _ = env.reset()
                ep_len, ep_ret = 0,0
                
        update()
                
            
            
    def update():
        data = buff.get()
        
        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()
        
        
        # Train policy with a single step of gradient descent
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        mpi_avg_grads(ac.pi)    # average grads across MPI processes
        pi_optimizer.step()

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()
        
    
    
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        
        # Policy loss 
        pi, logp = ac.pi(obs,act)
        loss_pi = -(logp * adv).mean()
        
        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent)

        return loss_pi, pi_info
    
    
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()