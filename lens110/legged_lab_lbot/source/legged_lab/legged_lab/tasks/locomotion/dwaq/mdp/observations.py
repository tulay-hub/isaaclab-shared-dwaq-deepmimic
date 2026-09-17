"""DWAQ-specific observation functions.
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase_sin_cos(
    env: ManagerBasedRLEnv,
    period: float = 0.8,
    offset: list[float] | tuple[float, ...] = (0.0, 0.5),
) -> torch.Tensor:
    """Sinusoidal encoding of the bipedal gait phase.

    Computes ``[sin(2π φ_i), cos(2π φ_i)]`` for each leg, where ``φ_i`` is
    the normalised phase for leg *i* derived from the episode clock.

    Returns shape ``(num_envs, 2 * num_legs)`` — for two legs this is 4.
    """
    t = env.episode_length_buf.float() * env.step_dt
    global_phase = (t % period) / period  # (num_envs,)

    phases = torch.stack([(global_phase + off) % 1.0 for off in offset], dim=-1)  # (num_envs, num_legs)
    sin_phase = torch.sin(2 * math.pi * phases)
    cos_phase = torch.cos(2 * math.pi * phases)
    return torch.cat([sin_phase, cos_phase], dim=-1)  # (num_envs, 4)


def feet_contact_binary(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Binary foot-contact indicator (1 = in contact, 0 = in air).

    Returns shape ``(num_envs, num_feet)``.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history
    is_contact = (
        torch.max(torch.norm(net_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
        > threshold
    )
    return is_contact.float()


def feet_pos_body_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Feet positions in the robot body frame.

    Returns shape ``(num_envs, num_feet * 3)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w  # (N, 3)
    root_quat = asset.data.root_quat_w  # (N, 4)
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]  # (N, F, 3)

    rel = feet_pos_w - root_pos.unsqueeze(1)
    num_feet = rel.shape[1]
    parts = [quat_apply_inverse(root_quat, rel[:, i]) for i in range(num_feet)]
    return torch.cat(parts, dim=-1)  # (N, F*3)


def feet_vel_body_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Feet linear velocities in the robot body frame.

    Returns shape ``(num_envs, num_feet * 3)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    root_vel = asset.data.root_lin_vel_w  # (N, 3)
    root_quat = asset.data.root_quat_w  # (N, 4)
    feet_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]  # (N, F, 3)

    rel = feet_vel_w - root_vel.unsqueeze(1)
    num_feet = rel.shape[1]
    parts = [quat_apply_inverse(root_quat, rel[:, i]) for i in range(num_feet)]
    return torch.cat(parts, dim=-1)  # (N, F*3)


def feet_contact_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """3-D contact force for each foot (world frame).

    Returns shape ``(num_envs, num_feet * 3)``.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # (N, F, 3)
    return forces.reshape(forces.shape[0], -1)
