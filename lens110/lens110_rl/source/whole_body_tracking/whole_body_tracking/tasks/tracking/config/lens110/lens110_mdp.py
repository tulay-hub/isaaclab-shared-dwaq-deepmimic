"""Lens110 H 版 (161 维观测) 的观测/终止函数。

观测维度保持 H 版 161 维不变; 奖励直接使用 whole_body_tracking_engineai
框架自带的 T800 那套 (tracking_env_cfg.RewardsCfg), 这里不再重复定义。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import matrix_from_quat

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


KEY_BODY_NAMES = [
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_elbow_link", "right_elbow_link",
    "left_shoulder_roll_link", "right_shoulder_roll_link",
]


# ---------------- 观测 ----------------


def root_rot_tan_norm(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """机身旋转矩阵 tan/norm 两列 (6 维姿态表示)。"""
    robot = env.scene[asset_cfg.name]
    rotm = matrix_from_quat(robot.data.root_quat_w)
    return torch.cat([rotm[..., 0], rotm[..., 2]], dim=-1)


def foot_contact_flags(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """左右脚触地标志 (2 维)。"""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if sensor_cfg.body_ids == slice(None):
        raise ValueError("foot_contact_flags requires explicit body_names")
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    return (torch.norm(forces, dim=-1) > threshold).to(torch.float32)


def ref_root_rot_tan_norm(
    env: ManagerBasedRLEnv,
    command_name: str,
    num_steps: int = 4,
) -> torch.Tensor:
    """参考运动朝向 tan/norm (24 = 4 步 x 6)。"""
    command: MotionCommand = env.command_manager.get_term(command_name)
    motion = command.motion
    max_i = motion.body_quat_w.shape[0] - 1
    steps = torch.stack([(command.time_steps + s).clamp(max=max_i) for s in range(num_steps)], dim=1)
    quat = motion.body_quat_w[steps, 0]
    rotm = matrix_from_quat(quat)
    return torch.cat([rotm[..., 0], rotm[..., 2]], dim=-1).reshape(env.num_envs, -1)


def ref_joint_pos(
    env: ManagerBasedRLEnv,
    command_name: str,
    num_steps: int = 4,
) -> torch.Tensor:
    """参考运动关节位置 (84 = 4 步 x 21)。"""
    command: MotionCommand = env.command_manager.get_term(command_name)
    motion = command.motion
    max_i = motion.joint_pos.shape[0] - 1
    steps = torch.stack([(command.time_steps + s).clamp(max=max_i) for s in range(num_steps)], dim=1)
    return motion.joint_pos[steps].reshape(env.num_envs, -1)


# ---------------- 终止 ----------------


def motion_data_finish(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """舞蹈参考耗尽 -> 跳完 (正常结束)。"""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.time_steps >= command.motion.time_step_total - 1


def deviation_root_pos_w(
    env: ManagerBasedRLEnv, command_name: str, threshold: float
) -> torch.Tensor:
    robot = env.scene["robot"]
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(robot.data.root_pos_w - command.anchor_pos_w, dim=-1) > threshold


def deviation_key_body_pos_w(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=KEY_BODY_NAMES, preserve_order=True
    ),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_ids = asset_cfg.body_ids if asset_cfg.body_ids != slice(None) else \
        robot.find_bodies(list(asset_cfg.body_names), preserve_order=True)[0]
    ref_ids = [command.cfg.body_names.index(name) for name in asset_cfg.body_names]
    dist = torch.norm(robot.data.body_pos_w[:, body_ids] - command.body_pos_w[:, ref_ids], dim=-1)
    return torch.max(dist, dim=-1)[0] > threshold
