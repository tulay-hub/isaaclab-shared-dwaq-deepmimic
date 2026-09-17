from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_apply_inverse,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _normalize_quat(quat: torch.Tensor) -> torch.Tensor:
    return quat / torch.clamp(torch.norm(quat, dim=-1, keepdim=True), min=1e-8)


def _lerp_tensor(a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    view_shape = [blend.shape[0]] + [1] * a.dim()
    alpha = blend.view(*view_shape)
    return a.unsqueeze(0) * (1.0 - alpha) + b.unsqueeze(0) * alpha


def _quat_slerp_batch(q0: torch.Tensor, q1: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    q0 = _normalize_quat(q0)
    q1 = _normalize_quat(q1)

    dot = torch.sum(q0 * q1, dim=-1)
    q1 = torch.where(dot.unsqueeze(-1) < 0.0, -q1, q1)
    dot = torch.sum(q0 * q1, dim=-1).clamp(-1.0, 1.0)

    linear_mask = torch.abs(dot) > 0.9995
    lerp = _normalize_quat(q0 * (1.0 - blend.unsqueeze(-1)) + q1 * blend.unsqueeze(-1))

    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta = theta_0 * blend
    sin_theta = torch.sin(theta)

    s0 = torch.sin(theta_0 - theta) / torch.clamp(sin_theta_0, min=1e-8)
    s1 = sin_theta / torch.clamp(sin_theta_0, min=1e-8)
    slerp = _normalize_quat(s0.unsqueeze(-1) * q0 + s1.unsqueeze(-1) * q1)
    return torch.where(linear_mask.unsqueeze(-1), lerp, slerp)


def _bridge_quat_sequence(last_quat: torch.Tensor, first_quat: torch.Tensor, bridge_frames: int) -> torch.Tensor:
    if bridge_frames <= 0:
        return torch.empty((0,) + last_quat.shape, dtype=last_quat.dtype, device=last_quat.device)
    flat_last = last_quat.reshape(-1, 4)
    flat_first = first_quat.reshape(-1, 4)
    blend = torch.linspace(0.0, 1.0, bridge_frames + 2, device=last_quat.device, dtype=last_quat.dtype)[1:-1]
    bridge = []
    for tau in blend:
        tau_vec = torch.full((flat_last.shape[0],), tau, dtype=last_quat.dtype, device=last_quat.device)
        bridge.append(_quat_slerp_batch(flat_last, flat_first, tau_vec).reshape(last_quat.shape))
    return torch.stack(bridge, dim=0)


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        body_indexes: Sequence[int],
        device: str = "cpu",
        min_traj_duration: float | None = None,
        bridge_frames: int = 20,
    ):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        fps = data["fps"]
        self.fps = float(np.asarray(fps).reshape(-1)[0])
        motion_tensors = {
            "joint_pos": torch.tensor(data["joint_pos"], dtype=torch.float32, device=device),
            "joint_vel": torch.tensor(data["joint_vel"], dtype=torch.float32, device=device),
            "body_pos_w": torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device),
            "body_quat_w": torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device),
            "body_lin_vel_w": torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device),
            "body_ang_vel_w": torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device),
        }
        if min_traj_duration is not None and min_traj_duration > 0.0:
            min_frames = int(min_traj_duration * self.fps) + 1
            motion_tensors = self._extend_short_trajectory(
                motion_tensors,
                min_frames=min_frames,
                bridge_frames=bridge_frames,
                source_file=motion_file,
            )
        self.joint_pos = motion_tensors["joint_pos"]
        self.joint_vel = motion_tensors["joint_vel"]
        self._body_pos_w = motion_tensors["body_pos_w"]
        self._body_quat_w = motion_tensors["body_quat_w"]
        self._body_lin_vel_w = motion_tensors["body_lin_vel_w"]
        self._body_ang_vel_w = motion_tensors["body_ang_vel_w"]
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    def _extend_short_trajectory(
        self,
        motion: dict[str, torch.Tensor],
        min_frames: int,
        bridge_frames: int,
        source_file: str,
    ) -> dict[str, torch.Tensor]:
        frame_count = motion["joint_pos"].shape[0]
        if frame_count >= min_frames:
            return motion

        first = {k: v[0] for k, v in motion.items()}
        last = {k: v[-1] for k, v in motion.items()}
        blend = torch.linspace(
            0.0, 1.0, bridge_frames + 2, device=motion["joint_pos"].device, dtype=motion["joint_pos"].dtype
        )[1:-1]

        bridge = {
            "joint_pos": _lerp_tensor(last["joint_pos"], first["joint_pos"], blend),
            "joint_vel": _lerp_tensor(last["joint_vel"], first["joint_vel"], blend),
            "body_pos_w": _lerp_tensor(last["body_pos_w"], first["body_pos_w"], blend),
            "body_quat_w": _bridge_quat_sequence(last["body_quat_w"], first["body_quat_w"], bridge_frames),
            "body_lin_vel_w": _lerp_tensor(last["body_lin_vel_w"], first["body_lin_vel_w"], blend),
            "body_ang_vel_w": _lerp_tensor(last["body_ang_vel_w"], first["body_ang_vel_w"], blend),
        }

        pieces = [{k: v for k, v in motion.items()}]
        total_frames = frame_count
        while total_frames < min_frames:
            if bridge_frames > 0:
                pieces.append(bridge)
                total_frames += bridge_frames
            pieces.append(motion)
            total_frames += frame_count

        result = {key: torch.cat([piece[key] for piece in pieces], dim=0) for key in motion.keys()}
        orig_duration = max(frame_count - 1, 0) / self.fps
        new_duration = max(result["joint_pos"].shape[0] - 1, 0) / self.fps
        print(
            f"[INFO] Extended short motion '{os.path.basename(source_file)}': "
            f"{frame_count} -> {result['joint_pos'].shape[0]} frames "
            f"({orig_duration:.2f}s -> {new_duration:.2f}s, bridge={bridge_frames})"
        )
        return result

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        motion_body_names = self.cfg.motion_body_names if self.cfg.motion_body_names is not None else self.cfg.body_names
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )
        self.motion_body_indexes = torch.tensor(
            [motion_body_names.index(name) for name in self.cfg.body_names], dtype=torch.long, device=self.device
        )

        if self.cfg.motion_joint_names is not None:
            self.robot_joint_indexes = torch.tensor(
                self.robot.find_joints(self.cfg.motion_joint_names, preserve_order=True)[0],
                dtype=torch.long,
                device=self.device,
            )
        else:
            self.robot_joint_indexes = None

        self.motion = MotionLoader(
            self.cfg.motion_file,
            self.motion_body_indexes,
            device=self.device,
            min_traj_duration=self.cfg.min_traj_duration,
            bridge_frames=self.cfg.bridge_frames,
        )
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        if self.robot_joint_indexes is None:
            return self.robot.data.joint_pos
        return self.robot.data.joint_pos[:, self.robot_joint_indexes]

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        if self.robot_joint_indexes is None:
            return self.robot.data.joint_vel
        return self.robot.data.joint_vel[:, self.robot_joint_indexes]

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _refresh_relative_motion_state(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids_tensor = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids_tensor = env_ids
        if env_ids_tensor.numel() == 0:
            return

        anchor_pos_w_repeat = self.anchor_pos_w[env_ids_tensor][:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[env_ids_tensor][:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[env_ids_tensor][:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[env_ids_tensor][:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat.clone()
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w[env_ids_tensor] = quat_mul(delta_ori_w, self.body_quat_w[env_ids_tensor])
        self.body_pos_relative_w[env_ids_tensor] = delta_pos_w + quat_apply(
            delta_ori_w, self.body_pos_w[env_ids_tensor] - anchor_pos_w_repeat
        )

    def _pd_stand_reset_would_terminate(self, env_ids: torch.Tensor) -> torch.Tensor:
        if env_ids.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)

        terminated = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        terminations_cfg = self._env.cfg.terminations

        anchor_pos_cfg = getattr(terminations_cfg, "anchor_pos", None)
        if anchor_pos_cfg is not None:
            threshold = float(anchor_pos_cfg.params["threshold"])
            terminated |= torch.abs(self.anchor_pos_w[env_ids, -1] - self.robot_anchor_pos_w[env_ids, -1]) > threshold

        anchor_ori_cfg = getattr(terminations_cfg, "anchor_ori", None)
        if anchor_ori_cfg is not None:
            threshold = float(anchor_ori_cfg.params["threshold"])
            gravity_vec_w = self.robot.data.GRAVITY_VEC_W
            if gravity_vec_w.ndim > 1:
                gravity_vec_w = gravity_vec_w[env_ids]
            motion_projected_gravity_b = quat_apply_inverse(self.anchor_quat_w[env_ids], gravity_vec_w)
            robot_projected_gravity_b = quat_apply_inverse(self.robot_anchor_quat_w[env_ids], gravity_vec_w)
            terminated |= torch.abs(motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]) > threshold

        ee_body_pos_cfg = getattr(terminations_cfg, "ee_body_pos", None)
        if ee_body_pos_cfg is not None:
            threshold = float(ee_body_pos_cfg.params["threshold"])
            body_names = ee_body_pos_cfg.params.get("body_names")
            body_indexes = [
                i for i, name in enumerate(self.cfg.body_names) if (body_names is None) or (name in body_names)
            ]
            if body_indexes:
                body_indexes = torch.tensor(body_indexes, dtype=torch.long, device=self.device)
                error_z = torch.abs(
                    self.body_pos_relative_w[env_ids][:, body_indexes, -1] - self.robot_body_pos_w[env_ids][:, body_indexes, -1]
                )
                terminated |= torch.any(error_z > threshold, dim=-1)

        return terminated

    def _reset_envs_from_motion(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return

        root_pos = self.anchor_pos_w[env_ids].clone()
        root_ori = self.anchor_quat_w[env_ids].clone()
        root_lin_vel = self.anchor_lin_vel_w[env_ids].clone()
        root_ang_vel = self.anchor_ang_vel_w[env_ids].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori = quat_mul(orientations_delta, root_ori)

        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel += rand_samples[:, :3]
        root_ang_vel += rand_samples[:, 3:]

        joint_pos = self.joint_pos[env_ids].clone()
        joint_vel = self.joint_vel[env_ids].clone()
        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)

        if self.robot_joint_indexes is None:
            soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
            joint_pos = torch.clip(joint_pos, soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1])
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        else:
            full_joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
            full_joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
            soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids][:, self.robot_joint_indexes]
            clipped_joint_pos = torch.clip(joint_pos, soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1])
            full_joint_pos[:, self.robot_joint_indexes] = clipped_joint_pos
            full_joint_vel[:, self.robot_joint_indexes] = joint_vel
            self.robot.write_joint_state_to_sim(full_joint_pos, full_joint_vel, env_ids=env_ids)

        root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    def _reset_envs_from_pd_stand(self, env_ids: torch.Tensor) -> torch.Tensor:
        if env_ids.numel() == 0:
            return env_ids

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self._env.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self._env.scene.update(self._env.physics_dt)
        self._refresh_relative_motion_state(env_ids)

        return env_ids[self._pd_stand_reset_would_terminate(env_ids)]

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._adaptive_sampling(env_ids)

        pd_stand_ratio = float(self.cfg.pd_stand_reset_ratio)
        if pd_stand_ratio <= 0.0:
            self._reset_envs_from_motion(env_ids)
            return

        pd_stand_mask = torch.rand(len(env_ids), device=self.device) < pd_stand_ratio
        motion_env_ids = env_ids[~pd_stand_mask]
        pd_stand_env_ids = env_ids[pd_stand_mask]

        self._reset_envs_from_motion(motion_env_ids)
        fallback_env_ids = self._reset_envs_from_pd_stand(pd_stand_env_ids)
        if fallback_env_ids.numel() > 0:
            self._reset_envs_from_motion(fallback_env_ids)

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        self._resample_command(env_ids)
        self._refresh_relative_motion_state()

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_w[:, i], self.body_quat_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING
    motion_body_names: list[str] | None = None
    motion_joint_names: list[str] | None = None

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    min_traj_duration: float | None = None
    bridge_frames: int = 20

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    pd_stand_reset_ratio: float = 0.0

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
