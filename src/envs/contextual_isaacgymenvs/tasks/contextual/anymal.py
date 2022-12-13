from ..anymal import Anymal

import torch


class ContextualAnymal(Anymal):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        super().__init__(cfg=cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.rigid_body_params = ["mass"]
        self.dof_params = ["damping"]
        self.isaacgym_params = self.rigid_body_params + self.dof_params

        self.original_rigid_body_params = self.gym.get_actor_rigid_body_properties(self.envs[0], 0)
        self.original_dof_params = self.gym.get_actor_dof_properties(self.envs[0], 0)

        self.all_system_params = {}
        self.all_system_params.update({f"action_noise_{i}": i + len(self.all_system_params) for i in range(self.num_actions)})
        self.all_system_params.update({f"mass_{i}": i + len(self.all_system_params) for i in range(4)})
        self.all_system_params.update({f"damping_{i}": i + len(self.all_system_params) for i in range(12)})

        self.all_system_params_inverted = dict(map(reversed, self.all_system_params.items()))

        self.context_tensor = torch.ones(self.num_envs, len(self.all_system_params)).to(self.device)

    def apply_context(self, env_ids, system_params):
        for env_id in env_ids:
            new_rigid_body_params = self.gym.get_actor_rigid_body_properties(self.envs[env_id], 0)
            new_dof_params = self.gym.get_actor_dof_properties(self.envs[env_id], 0)
            for param_name, param_idx in system_params.items():
                param_name = param_name.split("_")
                attr_name, attr_idx = "_".join(param_name[:-1]), int(param_name[-1])

                if attr_name in self.rigid_body_params:
                    for idx in range(attr_idx*3 + 1, attr_idx*3 + 1 + 3):
                        new_param_value = self.context_tensor[env_id, param_idx] * getattr(self.original_rigid_body_params[idx], attr_name)
                        setattr(new_rigid_body_params[idx], attr_name, new_param_value)
                elif attr_name in self.dof_params:
                    for idx in range(attr_idx, attr_idx + 1):
                        new_param_value = self.context_tensor[env_id, param_idx] * self.original_dof_params[attr_name][idx]
                        new_dof_params[attr_name][idx] = new_param_value
                else:
                    continue
            
            self.gym.set_actor_rigid_body_properties(self.envs[env_id], 0, new_rigid_body_params, True)
            self.gym.set_actor_dof_properties(self.envs[env_id], 0, new_dof_params)