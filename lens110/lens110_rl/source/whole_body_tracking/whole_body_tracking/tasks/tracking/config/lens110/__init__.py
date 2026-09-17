import gymnasium as gym

from . import flat_env_cfg
from .agents import rsl_rl_ppo_cfg

gym.register(
    id="Tracking-Flat-Lens110-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Lens110FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{rsl_rl_ppo_cfg.__name__}:Lens110FlatPPORunnerCfg",
    },
)
