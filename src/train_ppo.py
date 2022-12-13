# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_action_isaacgympy
from .utils.envs import make_env
from .utils.arguments import parse_args
from .utils.rollout_buffer import TorchDictRolloutBuffer
from .models.policies.ppo import Agent

from copy import deepcopy
from omegaconf import OmegaConf
import json

import datetime
import time
import random
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    args = parse_args()
    args_for_save = deepcopy(args)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    config = OmegaConf.load(os.path.join(dir_path, "../configs/base_config.yaml"))
    if args.config_path:
        override_config = OmegaConf.load(args.config_path)
        config = OmegaConf.merge(config, override_config)
    vars(args).update(config)

    # run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    timestamp = datetime.datetime.now().strftime("%m%d-%H%M%S")
    if args.exp_name:
        run_name = f"{args.env_id}/{args.context_encoder_type}/{args.exp_name}"
    else:
        run_name = f"{args.env_id}/{args.context_encoder_type}/{timestamp}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    args.device = device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"

    # env setup
    system_params = []
    context_samples = []
    if "Contextual" in args.env_id:
        randomization_config = OmegaConf.load(os.path.join(dir_path, "../configs/randomization", args.env_id+".yaml"))
        randomization_config = getattr(randomization_config, args.randomization_type)
        context_samples = eval(randomization_config.context_samples)
        system_params = randomization_config.system_params if context_samples else dict(randomization_config.system_params)
    envs = make_env(args.env_id, args.seed, args.num_envs, args.capture_video, args.record_video_step_frequency, run_name, args.len_history, device, system_params, context_samples)

    args.obs_dim = np.array(envs.single_observation_space["obs"].shape).prod()
    args.action_dim = np.prod(envs.single_action_space.shape)
    if "Contextual" in args.env_id:
        args.context_dim = np.array(envs.single_observation_space["context"].shape).prod()

    agent = Agent(args).to(device)
    sample_trajectory = hasattr(agent.context_encoder.config, "sample_trajectory") and agent.context_encoder.config.sample_trajectory

    # ALGO Logic: Storage setup
    rb = TorchDictRolloutBuffer(
        args.num_steps,
        envs.single_observation_space,
        envs.single_action_space,
        args.device,
        args.gae_lambda,
        args.gamma,
        args.num_envs,
        sample_trajectory=sample_trajectory ###################
    )

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    last_obs = next_obs = envs.reset()
    last_done = next_done = torch.zeros(args.num_envs, dtype=torch.float).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            for optimizer in agent.optimizers:
                optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs

            # ALGO LOGIC: action logic
            with torch.no_grad():
                context, context_info = agent.get_context({"obs": next_obs, "training": False})
                action, logprob, _, value = agent.get_action_and_value(next_obs, context=context)
                value = value.flatten()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, next_done, info = envs.step(action)
            if 0 <= step <= 2:
                for idx, d in enumerate(next_done):
                    if d:
                        episodic_return = info["r"][idx].item()
                        print(f"global_step={global_step}, episodic_return={episodic_return}")
                        writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                        writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)
                        if "consecutive_successes" in info:  # ShadowHand and AllegroHand metric
                            writer.add_scalar(
                                "charts/consecutive_successes", info["consecutive_successes"].item(), global_step
                            )
                        break

            rb.add(last_obs, next_obs["obs"], action, reward, last_done, value, logprob)
            last_obs = next_obs
            last_done = next_done

        with torch.no_grad():
            context, context_info = agent.get_context({"obs": next_obs, "training": False})
            value = agent.get_value(next_obs, context=context).reshape(1, -1)
            rb.compute_returns_and_advantage(value, next_done)
        writer_dict = agent.learn(rb)
        rb.reset()
        
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for key, value in writer_dict.items():
            writer.add_scalar(key, value, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        #### Save periodically
        if update % (num_updates // args.save_freq) == 0 or update == num_updates:
            agent.save(os.path.join('runs', run_name, f'checkpoint_{global_step}.pt'))
            with open(os.path.join('runs', run_name, 'training_args.json'), 'w') as fout:
                json.dump(vars(args_for_save), fout, indent=2)
            with open(os.path.join('runs', run_name, 'training_config.yaml'), 'w') as fout:
                OmegaConf.save(config=config, f=fout)
            print('[INFO] Checkpoint is saved')

    # envs.close()
    writer.close()
