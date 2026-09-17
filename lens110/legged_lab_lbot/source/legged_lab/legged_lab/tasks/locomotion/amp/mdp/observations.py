from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from dataclasses import MISSING

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from legged_lab.envs import ManagerBasedAnimationEnv
    from legged_lab.managers import AnimationTerm
    


def root_local_rot_tan_norm(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    
    root_quat = robot.data.root_quat_w
    yaw_quat = math_utils.yaw_quat(root_quat)
    
    root_quat_local = math_utils.quat_mul(math_utils.quat_conjugate(yaw_quat), root_quat)
    
    root_rotm_local = math_utils.matrix_from_quat(root_quat_local)
    # use the first and last column of the rotation matrix as the tangent and normal vectors
    tan_vec = root_rotm_local[:, :, 0]  # (N, 3)
    norm_vec = root_rotm_local[:, :, 2]  # (N, 3)
    obs = torch.cat([tan_vec, norm_vec], dim=-1)  # (N, 6)

    return obs


def key_body_pos_b(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=MISSING, preserve_order=True),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]

    key_body_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :]
    root_pos_w = robot.data.root_pos_w
    root_quat = robot.data.root_quat_w

    num_key_bodies = key_body_pos_w.shape[1]
    num_envs = root_pos_w.shape[0]
    key_body_pos = math_utils.quat_apply_inverse(
        root_quat.unsqueeze(1).expand(-1, num_key_bodies, -1),
        key_body_pos_w - root_pos_w.unsqueeze(1).expand(-1, num_key_bodies, -1),
    )

    return key_body_pos.reshape(num_envs, -1)


def ref_root_local_rot_tan_norm(
    env: ManagerBasedAnimationEnv, 
    animation: str, 
    flatten_steps_dim: bool = True,
) -> torch.Tensor:

    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    num_envs = env.num_envs
    
    ref_root_quat = animation_term.get_root_quat() # shape: (num_envs, num_steps, 4)
    ref_yaw_quat = math_utils.yaw_quat(ref_root_quat)
    ref_root_quat_local = math_utils.quat_mul(
        math_utils.quat_conjugate(ref_yaw_quat), ref_root_quat
    )  # shape: (num_envs, num_steps, 4)
    ref_root_rotm_local = math_utils.matrix_from_quat(ref_root_quat_local) # shape: (num_envs, num_steps, 3, 3)
    
    tan_vec = ref_root_rotm_local[:, :, :, 0]  # (num_envs, num_steps, 3)
    norm_vec = ref_root_rotm_local[:, :, :, 2]  # (num_envs, num_steps, 3)
    obs = torch.cat([tan_vec, norm_vec], dim=-1)  # (num_envs, num_steps, 6)
    
    if flatten_steps_dim:
        return obs.reshape(num_envs, -1)
    else:
        return obs

def ref_projected_gravity(
    env: ManagerBasedAnimationEnv, 
    animation: str, 
) -> torch.Tensor:
    
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    num_envs = env.num_envs
        
    ref_root_quat = animation_term.get_root_quat() # shape: (num_envs, num_steps, 4)
    
    gravity_vec_w = torch.tensor([0.0, 0.0, -1.0], device=ref_root_quat.device)
    gravity_vec_w = gravity_vec_w.view(1, 1, 3).expand(ref_root_quat.shape[0], ref_root_quat.shape[1], 3)
    projected_gravity = math_utils.quat_apply_inverse(ref_root_quat, gravity_vec_w)
    return projected_gravity        


def ref_root_ang_vel_b(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
) -> torch.Tensor:
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    ref_root_ang_vel_w = animation_term.get_root_ang_vel_w()
    ref_root_quat = animation_term.get_root_quat()
    ref_root_ang_vel = math_utils.quat_apply_inverse(ref_root_quat, ref_root_ang_vel_w)

    if flatten_steps_dim:
        return ref_root_ang_vel.reshape(env.num_envs, -1)
    return ref_root_ang_vel


def ref_root_lin_vel_b(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
) -> torch.Tensor:
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    ref_root_lin_vel_w = animation_term.get_root_vel_w()
    ref_root_quat = animation_term.get_root_quat()
    ref_root_lin_vel = math_utils.quat_apply_inverse(ref_root_quat, ref_root_lin_vel_w)

    if flatten_steps_dim:
        return ref_root_lin_vel.reshape(env.num_envs, -1)
    return ref_root_lin_vel


def ref_joint_pos(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
    joint_ids: list[int] | tuple[int, ...] | None = None,
) -> torch.Tensor:
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    ref_dof_pos = animation_term.get_dof_pos()
    if joint_ids is not None:
        ref_dof_pos = ref_dof_pos[:, :, joint_ids]

    if flatten_steps_dim:
        return ref_dof_pos.reshape(env.num_envs, -1)
    return ref_dof_pos


def ref_joint_vel(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
    joint_ids: list[int] | tuple[int, ...] | None = None,
) -> torch.Tensor:
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    ref_dof_vel = animation_term.get_dof_vel()
    if joint_ids is not None:
        ref_dof_vel = ref_dof_vel[:, :, joint_ids]

    if flatten_steps_dim:
        return ref_dof_vel.reshape(env.num_envs, -1)
    return ref_dof_vel


def ref_key_body_pos_b(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
    body_ids: list[int] | tuple[int, ...] | None = None,
) -> torch.Tensor:
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    ref_key_body_pos = animation_term.get_key_body_pos_b()
    if body_ids is not None:
        ref_key_body_pos = ref_key_body_pos[:, :, body_ids]

    if flatten_steps_dim:
        return ref_key_body_pos.reshape(env.num_envs, -1)
    return ref_key_body_pos.reshape(ref_key_body_pos.shape[0], ref_key_body_pos.shape[1], -1)
