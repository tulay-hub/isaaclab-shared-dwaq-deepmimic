import gymnasium as gym

from . import agents


gym.register(
    id="LeggedLab-Isaac-DWAQ-Unitree-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_dwaq_env_cfg:G1DwaqEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DwaqRunnerCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-DWAQ-Unitree-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_dwaq_env_cfg:G1DwaqEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DwaqRunnerCfg",
    },
)
