import hydra
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from isaacgymenvs.utils.reformat import omegaconf_to_dict

from typing import Callable
from .tasks import isaacgym_task_map

# OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower())
# OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
# OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
# OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)


def get_rlgames_env_creator(
        # used to create the vec task
        seed: int,
        task_config: dict,
        task_name: str,
        sim_device: str,
        rl_device: str,
        graphics_device_id: int,
        headless: bool,
        # Used to handle multi-gpu case
        multi_gpu: bool = False,
        post_create_hook: Callable = None,
        virtual_screen_capture: bool = False,
        force_render: bool = False,
):
    """Parses the configuration parameters for the environment task and creates a VecTask

    Args:
        task_config: environment configuration.
        task_name: Name of the task, used to evaluate based on the imported name (eg 'Trifinger')
        sim_device: The type of env device, eg 'cuda:0'
        rl_device: Device that RL will be done on, eg 'cuda:0'
        graphics_device_id: Graphics device ID.
        headless: Whether to run in headless mode.
        multi_gpu: Whether to use multi gpu
        post_create_hook: Hooks to be called after environment creation.
            [Needed to setup WandB only for one of the RL Games instances when doing multiple GPUs]
        virtual_screen_capture: Set to True to allow the users get captured screen in RGB array via `env.render(mode='rgb_array')`. 
        force_render: Set to True to always force rendering in the steps (if the `control_freq_inv` is greater than 1 we suggest stting this arg to True)
    Returns:
        A VecTaskPython object.
    """
    def create_rlgpu_env():
        """
        Creates the task from configurations and wraps it using RL-games wrappers if required.
        """

        # create native task and pass custom config
        env = isaacgym_task_map[task_name](
            cfg=task_config,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        if post_create_hook is not None:
            post_create_hook()

        return env
    return create_rlgpu_env


def make(
    seed: int, 
    task: str, 
    num_envs: int, 
    sim_device: str,
    rl_device: str,
    graphics_device_id: int = -1,
    headless: bool = False,
    multi_gpu: bool = False,
    virtual_screen_capture: bool = False,
    force_render: bool = True,
    cfg: DictConfig = None
): 
    # create hydra config if no config passed in
    if cfg is None:
        # reset current hydra config if already parsed (but not passed in here)
        if HydraConfig.initialized():
            task = HydraConfig.get().runtime.choices['task']
            hydra.core.global_hydra.GlobalHydra.instance().clear()

        with initialize(config_path="./cfg"):
            cfg = compose(config_name="config", overrides=[f"task={task}"])
            cfg_dict = omegaconf_to_dict(cfg.task)
            cfg_dict['env']['numEnvs'] = num_envs
    # reuse existing config
    else:
        cfg_dict = omegaconf_to_dict(cfg.task)

    create_rlgpu_env = get_rlgames_env_creator(
        seed=seed,
        task_config=cfg_dict,
        task_name=cfg_dict["name"],
        sim_device=sim_device,
        rl_device=rl_device,
        graphics_device_id=graphics_device_id,
        headless=headless,
        multi_gpu=multi_gpu,
        virtual_screen_capture=virtual_screen_capture,
        force_render=force_render,
    )
    return create_rlgpu_env()