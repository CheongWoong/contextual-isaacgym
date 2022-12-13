from omegaconf import OmegaConf
import os

import gym
import isaacgym
import isaacgymenvs

from ...envs import contextual_isaacgymenvs
from .context_wrapper import ContextWrapper
from .history_wrapper import HistoryWrapper
from .record_episode_statistics_torch import RecordEpisodeStatisticsTorch


def make_env(env_id, seed, num_envs, capture_video, record_video_step_frequency, run_name, len_history, device, system_params=[], context_samples=[], gui=False):
    print(device)
    selected_envs = contextual_isaacgymenvs if "Contextual" in env_id else isaacgymenvs
    envs = selected_envs.make(
        seed=seed,
        task=env_id,
        num_envs=num_envs,
        sim_device=device,
        rl_device=device,
        graphics_device_id=0 if "cuda" in device else -1,
        headless=False if gui else True,
        multi_gpu=False,
        virtual_screen_capture=capture_video,
        force_render=False,
    )
    if capture_video:
        envs.is_vector_env = True
        envs = gym.wrappers.RecordVideo(
            envs,
            f"videos/{run_name}",
            step_trigger=lambda step: step % record_video_step_frequency == 0,
            video_length=100,  # for each video record up to 100 steps
        )
    envs = RecordEpisodeStatisticsTorch(envs, device)
    envs.observation_space = gym.spaces.Dict({"obs": envs.observation_space})
    envs = HistoryWrapper(envs, len_history)
    if "Contextual" in env_id:
        envs = ContextWrapper(envs, system_params, context_samples)
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    return envs