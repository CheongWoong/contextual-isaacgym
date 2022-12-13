import torch


def get_contextual_task(base):

    class ContextualTask(base):
        def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
            super().__init__(cfg=cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

            self.all_system_params = {}
            self.all_system_params.update({f"action_noise_{i}": i + len(self.all_system_params) for i in range(self.num_actions)})
            self.all_system_params_inverted = dict(map(reversed, self.all_system_params.items()))

            self.isaacgym_params = []

            self.context_tensor = torch.ones(self.num_envs, len(self.all_system_params)).to(self.device)

    return ContextualTask