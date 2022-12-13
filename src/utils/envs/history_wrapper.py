import gym
import torch
import numpy as np


class HistoryWrapper(gym.Wrapper):
    def __init__(self, env, len_history):
        super().__init__(env)

        self.len_history = len_history
        self.obs_dim = env.observation_space["obs"].shape if hasattr(env.observation_space["obs"], "shape") else tuple(env.observation_space["obs"].n)

        self.history_obs = torch.zeros((self.num_envs, self.len_history, *self.obs_dim), device=self.device)
        self.history_obs_delta = torch.zeros((self.num_envs, self.len_history, *self.obs_dim), device=self.device)
        self.history_act = torch.zeros((self.num_envs, self.len_history, *self.action_space.shape), device=self.device)
        self.prev_state = None
        
        history_observation_space = {
            "history_obs" : env.observation_space["obs"].__class__(-np.inf, np.inf, (len_history, *self.obs_dim)),
            "history_obs_delta" : env.observation_space["obs"].__class__(-np.inf, np.inf, (len_history, *self.obs_dim)),
            "history_act" : env.action_space.__class__(-np.inf, np.inf, (len_history, *self.action_space.shape)),
        }
        self.observation_space.spaces.update(history_observation_space)

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)

        self.history_obs = torch.cat([self.history_obs[:,1:], self.prev_state.unsqueeze(1)], dim=1)
        self.history_obs_delta = torch.cat([self.history_obs[:,1:], (observations["obs"] - self.prev_state).unsqueeze(1)], dim=1)
        self.history_act = torch.cat([self.history_act[:,1:], action.unsqueeze(1)], dim=1)
        self.prev_state = observations['obs']

        observations = {k: v for k, v in observations.items()}
        history = {
            "history_obs": self.history_obs.detach().clone(),
            "history_obs_delta": self.history_obs_delta.detach().clone(),
            "history_act": self.history_act.detach().clone(),
        }
        observations.update(history)
        
        self.history_obs[dones > 0] = 0
        self.history_obs_delta[dones > 0] = 0
        self.history_act[dones > 0] = 0
        self.prev_state[dones > 0] = 0
        
        return observations, rewards, dones, infos

    def reset(self):
        observations = super().reset()

        self.history_obs = torch.zeros((self.num_envs, self.len_history, *self.obs_dim), device=self.device)
        self.history_obs_delta = torch.zeros((self.num_envs, self.len_history, *self.obs_dim), device=self.device)
        self.history_act = torch.zeros((self.num_envs, self.len_history, *self.action_space.shape), device=self.device)
        self.prev_state = observations['obs']

        observations = {k: v for k, v in observations.items()}
        history = {
            "history_obs": self.history_obs.detach().clone(),
            "history_obs_delta": self.history_obs_delta.detach().clone(),
            "history_act": self.history_act.detach().clone(),
        }
        observations.update(history)

        return observations


if __name__ == "__main__":
    pass