from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]

# =====================================================================
# Lens110 仅下肢（12 DOF）配置
#
# 观测布局（与 lens110_amp_env_cfg.py 中的观测项顺序一一对应）：
#   policy: base_ang_vel(3) + projected_gravity(3) + velocity_commands(3)
#           + joint_pos(12) + joint_vel(12) + actions(12)              = 45
#   critic: 历史长度 3，每个 term 的 history 连续拼接:
#           3 * [base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3)
#           + velocity_commands(3) + joint_pos(12) + joint_vel(12) + actions(12)] = 144
# =====================================================================
JOINT_NUM = 12
ACTION_NUM = 12
POLICY_HISTORY_LEN = 1
CRITIC_HISTORY_LEN = 3
POLICY_OBS_DIM = 45
CRITIC_OBS_DIM = 144

# 12 个下肢关节（URDF/机器人 dof 顺序）: 0-5 左腿, 6-11 右腿
LEFT_JOINT_INDICES = [0, 1, 2, 3, 4, 5]
RIGHT_JOINT_INDICES = [6, 7, 8, 9, 10, 11]
# 左右镜像时需要翻转符号的关节（roll/yaw）:
# 左: hip_roll(1), hip_yaw(2), ankle_roll(5); 右: hip_roll(7), hip_yaw(8), ankle_roll(11)
SIGN_FLIP_JOINT_INDICES = [1, 2, 5, 7, 8, 11]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Apply left-right symmetry augmentation for Lens110 leg-only (12DOF) observations and actions."""
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)

        obs_aug["policy"][:batch_size] = obs["policy"][:]
        obs_aug["policy"][batch_size : 2 * batch_size] = _transform_policy_obs_left_right(obs["policy"])

        obs_aug["critic"][:batch_size] = obs["critic"][:]
        obs_aug["critic"][batch_size : 2 * batch_size] = _transform_critic_obs_left_right(obs["critic"])
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size : 2 * batch_size] = _switch_lens110_joints_left_right(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


def _transform_policy_obs_left_right(obs: torch.Tensor) -> torch.Tensor:
    _validate_last_dim(obs, POLICY_OBS_DIM, "policy observations")
    obs = obs.clone()
    end_idx = 0

    end_idx = _flip_repeated_vectors(obs, end_idx, 3, POLICY_HISTORY_LEN, [-1, 1, -1])  # base_ang_vel
    end_idx = _flip_repeated_vectors(obs, end_idx, 3, POLICY_HISTORY_LEN, [1, -1, 1])  # projected_gravity
    end_idx = _flip_repeated_vectors(obs, end_idx, 3, POLICY_HISTORY_LEN, [1, -1, -1])  # velocity_commands
    end_idx = _switch_repeated_joints(obs, end_idx, POLICY_HISTORY_LEN, JOINT_NUM)  # joint_pos
    end_idx = _switch_repeated_joints(obs, end_idx, POLICY_HISTORY_LEN, JOINT_NUM)  # joint_vel
    _switch_repeated_joints(obs, end_idx, POLICY_HISTORY_LEN, ACTION_NUM)  # actions

    return obs


def _transform_critic_obs_left_right(obs: torch.Tensor) -> torch.Tensor:
    _validate_last_dim(obs, CRITIC_OBS_DIM, "critic observations")
    obs = obs.clone()
    end_idx = 0

    end_idx = _flip_repeated_vectors(obs, end_idx, 3, CRITIC_HISTORY_LEN, [1, -1, 1])  # base_lin_vel
    end_idx = _flip_repeated_vectors(obs, end_idx, 3, CRITIC_HISTORY_LEN, [-1, 1, -1])  # base_ang_vel
    end_idx = _flip_repeated_vectors(obs, end_idx, 3, CRITIC_HISTORY_LEN, [1, -1, 1])  # projected_gravity
    end_idx = _flip_repeated_vectors(obs, end_idx, 3, CRITIC_HISTORY_LEN, [1, -1, -1])  # velocity_commands
    end_idx = _switch_repeated_joints(obs, end_idx, CRITIC_HISTORY_LEN, JOINT_NUM)  # joint_pos
    end_idx = _switch_repeated_joints(obs, end_idx, CRITIC_HISTORY_LEN, JOINT_NUM)  # joint_vel
    _switch_repeated_joints(obs, end_idx, CRITIC_HISTORY_LEN, ACTION_NUM)  # actions

    return obs


def _flip_repeated_vectors(
    obs: torch.Tensor,
    start_idx: int,
    dim: int,
    history_len: int,
    signs: list[int],
) -> int:
    signs_tensor = torch.tensor(signs, device=obs.device, dtype=obs.dtype)
    end_idx = start_idx
    for _ in range(history_len):
        next_idx = end_idx + dim
        obs[:, end_idx:next_idx] = obs[:, end_idx:next_idx] * signs_tensor
        end_idx = next_idx
    return end_idx


def _switch_repeated_joints(obs: torch.Tensor, start_idx: int, history_len: int, block_dim: int) -> int:
    end_idx = start_idx
    for _ in range(history_len):
        next_idx = end_idx + block_dim
        obs[:, end_idx:next_idx] = _switch_lens110_joints_left_right(obs[:, end_idx:next_idx])
        end_idx = next_idx
    return end_idx


def _switch_lens110_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Mirror Lens110 leg-only joint data (12 dims).

    Joint order: left leg 0-5, right leg 6-11.
    """
    _validate_last_dim(joint_data, JOINT_NUM, "Lens110 joint data")
    joint_data_switched = torch.zeros_like(joint_data)

    joint_data_switched[..., LEFT_JOINT_INDICES] = joint_data[..., RIGHT_JOINT_INDICES]
    joint_data_switched[..., RIGHT_JOINT_INDICES] = joint_data[..., LEFT_JOINT_INDICES]
    joint_data_switched[..., SIGN_FLIP_JOINT_INDICES] *= -1.0

    return joint_data_switched


def _validate_last_dim(tensor: torch.Tensor, expected_dim: int, name: str) -> None:
    if tensor.shape[-1] != expected_dim:
        raise ValueError(
            f"Expected {name} last dimension to be {expected_dim}, got {tensor.shape[-1]}. "
            "Update legged_lab/tasks/locomotion/amp/mdp/symmetry/lens110.py if Lens110 "
            "joint order or observation terms changed."
        )
