from stable_baselines3.common.buffers import DictRolloutBuffer

from typing import NamedTuple, Dict, Union
from gym import spaces
import numpy as np
import random

import torch


TensorDict = Dict[Union[str, int], torch.Tensor]

class Trajectory():
    def __init__(self):
        self.indices = []

def save_trajectory_indices(next_non_terminal, traj_dict, traj, step):
    if not next_non_terminal:
        traj[0] = Trajectory()
    traj[0].indices.append(step)
    traj_dict[step] = traj[0]

def sample_trajectory_indices(traj_dict, step):
    sampled_indices = random.sample(traj_dict[step].indices, k=1)
    return sampled_indices

class TorchDictRolloutBufferSamples(NamedTuple):
    observations: TensorDict
    next_observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

class TorchDictRolloutBufferSamples_2(NamedTuple):
    observations: TensorDict
    next_observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    observations_2: TensorDict
    next_observations_2: torch.Tensor
    actions_2: torch.Tensor

class TorchDictRolloutBuffer(DictRolloutBuffer):
    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        device,
        gae_lambda,
        gamma,
        n_envs,
        sample_trajectory=False,
    ):
        self.sample_trajectory = sample_trajectory

        super(DictRolloutBuffer, self).__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)

    def reset(self) -> None:
        assert isinstance(self.obs_shape, dict), "DictRolloutBuffer must be used with Dict obs space only"
        self.observations = {}
        for key, obs_input_shape in self.obs_shape.items():
            self.observations[key] = torch.zeros((self.buffer_size, self.n_envs) + obs_input_shape, dtype=torch.float32).to(self.device)
        self.next_observations = torch.zeros((self.buffer_size, self.n_envs) + self.obs_shape["obs"], dtype=torch.float32).to(self.device)
        self.actions = torch.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=torch.float32).to(self.device)
        self.rewards = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.float32).to(self.device)
        self.returns = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.float32).to(self.device)
        self.episode_starts = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.float32).to(self.device)
        self.values = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.float32).to(self.device)
        self.log_probs = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.float32).to(self.device)
        self.advantages = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.float32).to(self.device)
        self.generator_ready = False

        if self.sample_trajectory:
            self.observations_2 = {}
            for key, obs_input_shape in self.obs_shape.items():
                self.observations_2[key] = torch.zeros((self.buffer_size, self.n_envs) + obs_input_shape, dtype=torch.float32).to(self.device)
            self.next_observations_2 = torch.zeros((self.buffer_size, self.n_envs) + self.obs_shape["obs"], dtype=torch.float32).to(self.device)
            self.actions_2 = torch.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=torch.float32).to(self.device)

            self.traj_dicts = [{} for _ in range(self.n_envs)]
            self.trajs = [[Trajectory()] for _ in range(self.n_envs)]

        self.pos = 0
        self.full = False

    def compute_returns_and_advantage(self, last_values, dones) -> None:
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

            # save trajectory indices for latter use
            if self.sample_trajectory:
                temp = list(map(save_trajectory_indices, next_non_terminal, self.traj_dicts, self.trajs, [step]*self.n_envs)) ##########

        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

        if self.sample_trajectory:
            for step in reversed(range(self.buffer_size)):
                sampled_indices = list(map(sample_trajectory_indices, self.traj_dicts, [step]*self.n_envs)) #########
                sampled_indices = np.array(sampled_indices)
                sampled_indices = torch.from_numpy(sampled_indices).unsqueeze(0).to(self.device)
                
                for key in self.observations_2.keys():
                    expanded_sampled_indices = sampled_indices
                    for _ in range(len(self.obs_shape[key]) - 1):
                        expanded_sampled_indices = expanded_sampled_indices.unsqueeze(-1)
                    expanded_sampled_indices = expanded_sampled_indices.expand((-1, -1,) + self.obs_shape[key])
                    self.observations_2[key][step] = self.observations[key].gather(0, expanded_sampled_indices)
                self.next_observations_2[step] = self.next_observations.gather(0, sampled_indices)
                self.actions_2[step] = self.actions.gather(0, sampled_indices)

    def add(
        self,
        obs,
        next_obs,
        action,
        reward,
        episode_start,
        value,
        log_prob,
    ) -> None:  # pytype: disable=signature-mismatch
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        for key in self.observations.keys():
            obs_ = obs[key]
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs_ = obs_.reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = obs_

        if isinstance(self.observation_space.spaces["obs"], spaces.Discrete):
            next_obs = next_obs.reshape((self.n_envs,) + self.obs_shape["obs"])
        self.next_observations[self.pos] = next_obs

        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.episode_starts[self.pos] = episode_start
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def to_torch(self, array, copy=True):
        return array

    def get(self, batch_size=None):
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            _tensor_names = ["next_observations", "actions", "values", "log_probs", "advantages", "returns"]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

            if self.sample_trajectory:
                for key, obs in self.observations_2.items():
                    self.observations_2[key] = self.swap_and_flatten(obs)
                
                _tensor_names = ["next_observations_2", "actions_2"]
                for tensor in _tensor_names:
                    self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])


        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds, env=None) -> TorchDictRolloutBufferSamples:
        if not self.sample_trajectory:
            return TorchDictRolloutBufferSamples(
                observations={key: self.to_torch(obs[batch_inds]) for (key, obs) in self.observations.items()},
                next_observations=self.to_torch(self.next_observations[batch_inds]),
                actions=self.to_torch(self.actions[batch_inds]),
                old_values=self.to_torch(self.values[batch_inds].flatten()),
                old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
                advantages=self.to_torch(self.advantages[batch_inds].flatten()),
                returns=self.to_torch(self.returns[batch_inds].flatten()),
            )
        else:
            return TorchDictRolloutBufferSamples_2(
                observations={key: self.to_torch(obs[batch_inds]) for (key, obs) in self.observations.items()},
                next_observations=self.to_torch(self.next_observations[batch_inds]),
                actions=self.to_torch(self.actions[batch_inds]),
                old_values=self.to_torch(self.values[batch_inds].flatten()),
                old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
                advantages=self.to_torch(self.advantages[batch_inds].flatten()),
                returns=self.to_torch(self.returns[batch_inds].flatten()),
                observations_2={key: self.to_torch(obs[batch_inds]) for (key, obs) in self.observations_2.items()},
                next_observations_2=self.to_torch(self.next_observations_2[batch_inds]),
                actions_2=self.to_torch(self.actions_2[batch_inds]),
            )