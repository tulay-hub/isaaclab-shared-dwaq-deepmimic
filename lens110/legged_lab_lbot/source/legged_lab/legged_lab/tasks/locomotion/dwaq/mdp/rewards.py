"""DWAQ-specific reward functions adapted from DreamWaQ.
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Constant reward for staying alive (not terminated)."""
    return torch.ones(env.num_envs, device=env.device, dtype=torch.float)


def energy_norm(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize energy consumption: ``||torque * joint_vel||``."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.norm(
        torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]),
        dim=-1,
    )


def fly(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0
) -> torch.Tensor:
    """Penalize having no feet in contact (airborne / flying)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history
    is_contact = (
        torch.max(torch.norm(net_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    )
    return (torch.sum(is_contact, dim=-1) < 0.5).float()


def body_orientation_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize non-upright orientation of a specific body (e.g. torso).

    Projects gravity into the body frame and penalises the xy-components.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    from isaaclab.utils.math import quat_apply_inverse as quat_inv
    body_gravity = quat_inv(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :],
        asset.data.GRAVITY_VEC_W,
    )
    return torch.sum(torch.square(body_gravity[:, :2]), dim=1)


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize stumbling (lateral contact force > 5x vertical force)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    lateral = torch.norm(forces[:, :, :2], dim=2)
    vertical = torch.abs(forces[:, :, 2])
    return torch.any(lateral > 5.0 * vertical, dim=1).float()


def feet_too_near(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.2,
) -> torch.Tensor:
    """Penalize feet being closer together than *threshold* (bipeds)."""
    assert len(asset_cfg.body_ids) == 2, "feet_too_near requires exactly 2 body_ids"
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_heading_alignment(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    min_speed: float = 0.15,
    max_yaw_rate: float = 0.2,
    max_lateral_command: float = 0.1,
) -> torch.Tensor:
    """Penalize toe-in/toe-out during straight forward or backward motion.

    The ankle-roll body frame has its longitudinal axis along ``+X`` in the
    LENS110 URDF.  We compare that axis with the yaw-only torso frame and
    penalize its relative yaw only when the command is a straight segment.
    Turning and lateral segments are excluded so the policy can reorient its
    feet when the command explicitly asks it to turn.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    straight = (
        (command[:, 0].abs() > min_speed)
        & (command[:, 1].abs() < max_lateral_command)
        & (command[:, 2].abs() < max_yaw_rate)
    )

    foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    forward_axis = torch.zeros(
        foot_quat_w.shape[0], foot_quat_w.shape[1], 3, device=foot_quat_w.device, dtype=foot_quat_w.dtype
    )
    forward_axis[..., 0] = 1.0
    foot_forward_w = quat_apply(foot_quat_w, forward_axis)

    torso_yaw = yaw_quat(asset.data.root_quat_w).unsqueeze(1).expand_as(foot_quat_w)
    foot_forward_b = quat_apply_inverse(torso_yaw, foot_forward_w)
    foot_yaw = torch.atan2(foot_forward_b[..., 1], foot_forward_b[..., 0])
    error = torch.mean(torch.square(foot_yaw), dim=1)
    return error * straight.float()


def body_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 500.0,
    max_reward: float = 400.0,
) -> torch.Tensor:
    """Penalize excessive vertical contact force on feet."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward = force_z.clone()
    reward[reward < threshold] = 0
    reward[reward >= threshold] -= threshold
    return reward.clamp(min=0, max=max_reward)


def joint_deviation_l1_always(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint deviation from default pose at all times.

    Unlike ``joint_deviation_l1`` (which only penalises when standing),
    this applies regardless of the velocity command.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def idle_when_commanded(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.2,
    vel_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize being idle when a non-zero velocity command is active.

    Returns 1.0 when the robot is commanded to move (command magnitude >
    *cmd_threshold*) but its actual velocity is below *vel_threshold*.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd_xy = env.command_manager.get_command(command_name)[:, :2]
    cmd_mag = torch.linalg.norm(cmd_xy, dim=-1)

    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    vel_mag = torch.linalg.norm(vel_yaw[:, :2], dim=-1)

    return ((cmd_mag > cmd_threshold) & (vel_mag < vel_threshold)).float()


def feet_swing_height(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = 0.08,
) -> torch.Tensor:
    """Penalize swing-foot height deviation from *target_height*.

    Only applies when the foot is **not** in contact (swing phase).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    contact = torch.norm(net_forces, dim=-1) > 1.0

    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    error = torch.square(feet_z - target_height) * (~contact).float()
    return torch.sum(error, dim=-1)


def leg_ref_joint_pos(
    env: ManagerBasedRLEnv,
    left_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    right_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    period: float = 0.8,
    scales: tuple[float, ...] = (-0.2, 0.4, -0.2),
    double_support_threshold: float = 0.1,
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.1,
) -> torch.Tensor:
    """Reference joint-position tracking reward for leg pitch joints.

    Adapted from humanoid-gym ``compute_ref_state`` / ``_reward_joint_pos``.

    Uses two separate ``SceneEntityCfg`` (*left_cfg*, *right_cfg*) so the
    left/right assignment is explicit and independent of the robot's internal
    joint ordering.

    *scales* gives the signed amplitude for each joint type within one leg
    (e.g. hip-pitch, knee, ankle-pitch).  Both legs share the same scales.

    During the **swing phase** the reference position is::

        ref = default_pos + |sin_half_wave| × scale

    During stance and double-support (|sin| < *double_support_threshold*)
    the reference equals the default position.

    Left leg swings when ``sin(2π·t/period) < 0``; right leg swings when
    ``sin(2π·t/period) > 0``.

    Returns ``exp(-2·‖diff‖) − 0.2·clamp(‖diff‖, 0, 0.5)``.
    Use with a **positive** weight.
    """
    asset: Articulation = env.scene[left_cfg.name]

    t = env.episode_length_buf.float() * env.step_dt
    sin_pos = torch.sin(2 * math.pi * t / period)  # (N,)

    left_pos = asset.data.joint_pos[:, left_cfg.joint_ids]
    left_default = asset.data.default_joint_pos[:, left_cfg.joint_ids]
    right_pos = asset.data.joint_pos[:, right_cfg.joint_ids]
    right_default = asset.data.default_joint_pos[:, right_cfg.joint_ids]

    num_per_leg = left_pos.shape[1]
    scales_t = torch.tensor(
        list(scales[:num_per_leg]), device=left_pos.device, dtype=left_pos.dtype
    )

    # Left leg swings when sin < 0
    swing_l = (-sin_pos).clamp(min=0)
    # Right leg swings when sin > 0
    swing_r = sin_pos.clamp(min=0)

    # Double-support: both legs stay at default
    ds_mask = torch.abs(sin_pos) < double_support_threshold
    swing_l = swing_l.masked_fill(ds_mask, 0.0)
    swing_r = swing_r.masked_fill(ds_mask, 0.0)

    left_ref = left_default + swing_l.unsqueeze(1) * scales_t.unsqueeze(0)
    right_ref = right_default + swing_r.unsqueeze(1) * scales_t.unsqueeze(0)

    diff = torch.cat([left_pos - left_ref, right_pos - right_ref], dim=-1)
    diff_norm = torch.norm(diff, dim=-1)
    reward = torch.exp(-2.0 * diff_norm) - 0.2 * diff_norm.clamp(0, 0.5)

    # Only apply when a velocity command is active
    cmd = env.command_manager.get_command(command_name)[:, :2]
    moving = (torch.linalg.norm(cmd, dim=-1) > cmd_threshold).float()

    return reward * moving
