import gymnasium as gym

from . import agents

gym.register(
    id="LeggedLab-Isaac--Deepmimic-Lens110-v0",
    entry_point="legged_lab.envs:ManagerBasedAnimationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lens110_deepmimic_env_cfg:Lens110DeepMimicEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Lens110DeepMimicPPORunnerCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac--Deepmimic-Lens110-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAnimationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lens110_deepmimic_env_cfg:Lens110DeepMimicEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Lens110DeepMimicPPORunnerCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac--Deepmimic-Lens110-Debug-v0",
    entry_point="legged_lab.envs:ManagerBasedAnimationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lens110_deepmimic_env_cfg:Lens110DeepMimicEnvCfg_DEBUG",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Lens110DeepMimicPPORunnerCfg",
    },
)

# F 版备份（含关键点观测，251 维，训练效果最优）：
# 配置文件见 lens110_deepmimic_env_cfg_keybody.py；当前主配置（H 版，161 维）
# 删除了关键点观测，用于真机部署。
gym.register(
    id="LeggedLab-Isaac--Deepmimic-Lens110-KeyBody-v0",
    entry_point="legged_lab.envs:ManagerBasedAnimationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lens110_deepmimic_env_cfg_keybody:Lens110DeepMimicEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Lens110DeepMimicPPORunnerCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac--Deepmimic-Lens110-KeyBody-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAnimationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lens110_deepmimic_env_cfg_keybody:Lens110DeepMimicEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Lens110DeepMimicPPORunnerCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac--Deepmimic-Lens110-159-v0",
    entry_point="legged_lab.envs:ManagerBasedAnimationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lens110_deepmimic_env_cfg_159:Lens110DeepMimicEnvCfg159",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Lens110DeepMimicPPORunnerCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac--Deepmimic-Lens110-159-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAnimationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lens110_deepmimic_env_cfg_159:Lens110DeepMimicEnvCfg159_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Lens110DeepMimicPPORunnerCfg",
    },
)

# 侧滚（动作5 重定向数据，100Hz）：termination 放宽 + RSI 开启
gym.register(
    id="LeggedLab-Isaac--Deepmimic-Lens110-SideRoll-v0",
    entry_point="legged_lab.envs:ManagerBasedAnimationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lens110_deepmimic_env_cfg_sideroll:Lens110DeepMimicSideRollEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Lens110DeepMimicPPORunnerCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac--Deepmimic-Lens110-SideRoll-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAnimationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lens110_deepmimic_env_cfg_sideroll:Lens110DeepMimicSideRollEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Lens110DeepMimicPPORunnerCfg",
    },
)
