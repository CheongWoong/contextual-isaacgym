import gym
import torch
import numpy as np

import random


class ContextWrapper(gym.Wrapper):
    def __init__(self, env, system_params=[], context_samples=[]):
        super().__init__(env)

        self.system_params = system_params
        self.system_params_idx = torch.tensor([self.all_system_params[key] for key in self.system_params]).to(self.device)
        self.context_samples = context_samples
        if self.context_samples:
            self.context_samples = [torch.tensor(sample, dtype=torch.float32).to(self.device) for sample in self.context_samples]
        else:
            for key in self.system_params:
                self.system_params[key] = torch.tensor(self.system_params[key], dtype=torch.float32).to(self.device)

        self.context_dim = len(system_params)
        if self.context_dim < 1:
            raise Exception("need to define system parameters for randomization.")
        
        context_observation_space = {
            "context" : env.observation_space["obs"].__class__(-np.inf, np.inf, (self.context_dim,)),
        }
        self.observation_space.spaces.update(context_observation_space)
        
    def step(self, action):
        observations, rewards, dones, infos = super().step(action)

        observations = {k: v for k, v in observations.items()}
        context = {
            "context": self.context.detach().clone(),
        }
        observations.update(context)
        
        len_dones = int(dones.sum())
        if len_dones > 0:
            new_context = self.sample_context(len_dones)
            self.set_context(new_context, dones)

        return observations, rewards, dones, infos

    def reset(self):
        observations = super().reset()

        new_context = self.sample_context()
        self.set_context(new_context)

        observations = {k: v for k, v in observations.items()}
        context = {
            "context": self.context.detach().clone(),
        }
        observations.update(context)

        return observations

    def sample_context(self, num_envs=None):
        if num_envs is None:
            num_envs = self.num_envs
        if self.context_samples:
            contexts = random.choices(self.context_samples, k=num_envs)
            contexts = torch.stack(contexts, dim=0)
        else:
            contexts = []
            for name, intervals in self.system_params.items():
                interval = random.choices(intervals, k=num_envs)
                interval = torch.stack(interval, dim=0)
                low, high = interval.T
                context = torch.distributions.uniform.Uniform(low, high).sample()

                contexts.append(context)
            contexts = torch.stack(contexts, dim=-1)
        return contexts
            
    def set_context(self, context, mask=None):
        if mask is None:
            env_ids = torch.arange(len(context)).to(self.device)
        else:
            env_ids = mask.nonzero(as_tuple=False).flatten()
        self.unwrapped.context_tensor[env_ids,self.system_params_idx.unsqueeze(-1)] = context.T

        system_params = {self.all_system_params_inverted[idx.item()]: idx.item() for idx in self.system_params_idx}
        self.apply_context(env_ids, system_params)

    def get_context(self):
        return dict(zip(self.system_params.keys(), self.context.T))

    @property
    def context(self):
        return self.context_tensor[:,self.system_params_idx]

    @property
    def unwrapped(self):
        env = self
        while hasattr(env, "env"):
            env = env.env
        return env


if __name__ == "__main__":
    pass