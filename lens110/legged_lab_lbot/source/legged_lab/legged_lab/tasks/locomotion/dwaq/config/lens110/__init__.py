import gymnasium as gym
from legged_lab.tasks.locomotion.dwaq.config.lens110.lens110_dwaq_env_cfg import (
    Lens110DwaqEnvCfg,
    Lens110DwaqEnvCfg_PLAY,
)
from legged_lab.tasks.locomotion.dwaq.config.lens110.lens110_stairs_dwaq_env_cfg import (
    Lens110StairsDwaqEnvCfg,
    Lens110StairsDwaqEnvCfg_PLAY,
)
from legged_lab.tasks.locomotion.dwaq.config.lens110.agents.rsl_rl_ppo_cfg import (
    Lens110DwaqRunnerCfg,
)

gym.register(
    id="LeggedLab-Isaac--DWAQ-Lens110-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Lens110DwaqEnvCfg,
        "rsl_rl_cfg_entry_point": Lens110DwaqRunnerCfg,
    },
)

gym.register(
    id="LeggedLab-Isaac--DWAQ-Lens110-PLAY-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Lens110DwaqEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": Lens110DwaqRunnerCfg,
    },
)

gym.register(
    id="LeggedLab-Isaac--DWAQ-Lens110-Stairs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Lens110StairsDwaqEnvCfg,
        "rsl_rl_cfg_entry_point": Lens110DwaqRunnerCfg,
    },
)

gym.register(
    id="LeggedLab-Isaac--DWAQ-Lens110-Stairs-PLAY-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Lens110StairsDwaqEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": Lens110DwaqRunnerCfg,
    },
)
