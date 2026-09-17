import argparse
from copy import deepcopy
from pathlib import Path
import re
import sys


# Avoid importing the local utils/math.py when third-party packages expect the stdlib math module.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))

import joblib
import numpy as np


DEFAULT_INPUT_DIR = Path("legged_lab/source/legged_lab/legged_lab/data/MotionData/lens110_lab")
DEFAULT_OUTPUT_DIR = Path("legged_lab/source/legged_lab/legged_lab/data/MotionData/lens110_lab_tuned")

RUN_TAGS = ("run", "walk_to_run", "stand_to_run")
TURN_TAGS = ("turn", "xuanzhuan", "change_direction")
DYNAMIC_TAGS = ("walk", "run", "move_", "side_step", "turn", "xuanzhuan")

JOINT_LIMITS = {
    "left_hip_pitch_joint": (-2.10, 1.80),
    "right_hip_pitch_joint": (-2.10, 1.80),
    "left_shoulder_roll_joint": (-0.35, 2.70),
    "right_shoulder_roll_joint": (-2.70, 0.35),
    "left_knee_joint": (-0.10, 2.30),
    "right_knee_joint": (-0.10, 2.30),
    "left_elbow_joint": (-2.30, 0.05),
    "right_elbow_joint": (-2.30, 0.05),
    "left_shoulder_pitch_joint": (-3.40, 1.57),
    "right_shoulder_pitch_joint": (-3.40, 1.57),
    "left_hip_roll_joint": (-0.65, 0.65),
    "right_hip_roll_joint": (-0.65, 0.65),
    "left_hip_yaw_joint": (-0.75, 0.75),
    "right_hip_yaw_joint": (-0.75, 0.75),
    "left_ankle_pitch_joint": (-1.20, 0.00),
    "right_ankle_pitch_joint": (-1.20, 0.00),
    "left_ankle_roll_joint": (-0.80, 0.80),
    "right_ankle_roll_joint": (-0.80, 0.80),
}

STAND_POSE_TARGET = {
    "left_shoulder_pitch_joint": 0.02,
    "right_shoulder_pitch_joint": 0.02,
    "left_shoulder_roll_joint": 0.14,
    "right_shoulder_roll_joint": -0.14,
    "left_elbow_joint": -1.10,
    "right_elbow_joint": -1.10,
    "left_hip_pitch_joint": -0.03,
    "right_hip_pitch_joint": -0.03,
    "left_hip_roll_joint": 0.03,
    "right_hip_roll_joint": -0.03,
    "left_hip_yaw_joint": -0.08,
    "right_hip_yaw_joint": 0.08,
    "left_knee_joint": 0.04,
    "right_knee_joint": 0.04,
    "left_ankle_pitch_joint": -0.08,
    "right_ankle_pitch_joint": -0.08,
    "left_ankle_roll_joint": -0.02,
    "right_ankle_roll_joint": 0.02,
}

LOWER_ARM_POSE_TARGET_0202 = {
    "left_shoulder_pitch_joint": 0.50,
    "right_shoulder_pitch_joint": 0.50,
    "left_shoulder_roll_joint": 0.10,
    "right_shoulder_roll_joint": -0.10,
    "left_elbow_joint": -1.02,
    "right_elbow_joint": -1.02,
}

LOWER_ARM_POSE_TARGET_1634 = {
    "left_shoulder_pitch_joint": 0.50,
    "right_shoulder_pitch_joint": 0.50,
    "left_shoulder_roll_joint": 0.06,
    "right_shoulder_roll_joint": -0.06,
    "left_elbow_joint": -0.84,
    "right_elbow_joint": -0.84,
}

FINAL_1634_POSE_TARGET = {
    "left_shoulder_pitch_joint": 0.50,
    "right_shoulder_pitch_joint": 0.50,
    "left_shoulder_roll_joint": 0.05,
    "right_shoulder_roll_joint": -0.05,
    "left_elbow_joint": -0.82,
    "right_elbow_joint": -0.82,
    "left_hip_pitch_joint": 0.00,
    "right_hip_pitch_joint": 0.00,
    "left_knee_joint": 0.015,
    "right_knee_joint": 0.015,
    "left_ankle_pitch_joint": -0.03,
    "right_ankle_pitch_joint": -0.03,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process lens110 motion clips without overwriting the source data.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--shoulder-center", type=float, default=0.40)
    parser.add_argument("--shoulder-pitch-max", type=float, default=0.70)
    parser.add_argument("--shoulder-pitch-min", type=float, default=-0.40)
    parser.add_argument("--arm-forward-scale", type=float, default=1.48)
    parser.add_argument("--arm-backward-scale", type=float, default=0.88)
    parser.add_argument("--run-forward-scale", type=float, default=1.18)
    parser.add_argument("--run-backward-scale", type=float, default=0.52)
    parser.add_argument("--run-forward-bias", type=float, default=0.10)
    parser.add_argument("--run-symmetry-blend", type=float, default=0.40)
    parser.add_argument("--run-elbow-symmetry-blend", type=float, default=0.30)
    parser.add_argument("--numeric-forward-bias", type=float, default=0.28)
    parser.add_argument("--numeric-forward-scale", type=float, default=1.28)
    parser.add_argument("--numeric-motion-gain", type=float, default=1.20)
    parser.add_argument("--numeric-hip-forward-bias", type=float, default=0.12)
    parser.add_argument("--numeric-hip-forward-scale", type=float, default=1.08)
    parser.add_argument("--clip-0202-hip-pitch-scale", type=float, default=0.65)
    parser.add_argument("--walk-turn-shoulder-min-range", type=float, default=0.62)
    parser.add_argument("--walk-turn-shoulder-max-gain", type=float, default=1.85)
    parser.add_argument("--walk-turn-shoulder-forward-bias", type=float, default=0.06)
    parser.add_argument("--walk-turn-leg-arm-coupling", type=float, default=0.18)
    parser.add_argument("--c-series-shoulder-min-range", type=float, default=0.66)
    parser.add_argument("--c-series-shoulder-max-gain", type=float, default=1.70)
    parser.add_argument("--c-series-shoulder-forward-bias", type=float, default=0.05)
    parser.add_argument("--c-series-leg-arm-coupling", type=float, default=0.16)
    parser.add_argument("--elbow-bend", type=float, default=0.12)
    parser.add_argument("--run-elbow-bend", type=float, default=0.22)
    parser.add_argument("--hip-roll-width-offset", type=float, default=0.045)
    parser.add_argument("--hip-roll-width-scale", type=float, default=1.08)
    parser.add_argument("--turn-hip-yaw-scale", type=float, default=0.90)
    parser.add_argument("--turn-root-scale", type=float, default=0.90)
    parser.add_argument("--ankle-pitch-max", type=float, default=0.00)
    return parser.parse_args()


def classify_clip(clip_name: str, root_pos: np.ndarray) -> dict[str, bool]:
    lower_name = clip_name.lower()
    planar_span = float(np.linalg.norm(np.ptp(root_pos[:, :2], axis=0)))
    is_stand_like = "stand" in lower_name and planar_span < 0.15 and "run" not in lower_name and "walk" not in lower_name
    is_numeric_labeled = re.fullmatch(r"[0-9_]+", clip_name) is not None
    return {
        "is_c_series": clip_name.startswith("C"),
        "is_run": any(tag in lower_name for tag in RUN_TAGS),
        "is_turn": any(tag in lower_name for tag in TURN_TAGS),
        "is_walk_turn": "walk_turn" in lower_name,
        "is_dynamic": (any(tag in lower_name for tag in DYNAMIC_TAGS) or planar_span > 0.15) and not is_stand_like,
        "is_stand_like": is_stand_like,
        "is_numeric_labeled": is_numeric_labeled,
    }


def scale_around_center(values: np.ndarray, center: float, positive_scale: float, negative_scale: float) -> np.ndarray:
    delta = values - center
    return center + np.where(delta >= 0.0, delta * positive_scale, delta * negative_scale)


def scale_shoulder_pitch(
    values: np.ndarray, center: float, forward_scale: float, backward_scale: float, forward_bias: float
) -> np.ndarray:
    # For lens110 shoulder_pitch_joint, smaller / more negative values mean the arm swings forward.
    delta = values - center
    scaled = center + np.where(delta >= 0.0, delta * backward_scale, delta * forward_scale)
    return scaled - forward_bias


def scale_hip_pitch_forward(values: np.ndarray, forward_scale: float, forward_bias: float) -> np.ndarray:
    # For lens110 hip_pitch_joint, more negative values move the thigh further forward.
    return np.where(values < 0.0, values * forward_scale - forward_bias, values - forward_bias)


def scale_about_mean(values: np.ndarray, gain: float) -> np.ndarray:
    center = float(values.mean())
    return center + (values - center) * gain


def enforce_min_motion_range(values: np.ndarray, min_range: float, max_gain: float) -> np.ndarray:
    current_range = float(values.max() - values.min())
    if current_range <= 1e-6 or current_range >= min_range:
        return values
    gain = min(max_gain, min_range / current_range)
    return scale_about_mean(values, gain)


def leg_motion_score(dof_pos: np.ndarray, idx: dict[str, int]) -> float:
    hip_ranges = []
    knee_ranges = []
    for joint_name in ("left_hip_pitch_joint", "right_hip_pitch_joint"):
        values = dof_pos[:, idx[joint_name]]
        hip_ranges.append(float(values.max() - values.min()))
    for joint_name in ("left_knee_joint", "right_knee_joint"):
        values = dof_pos[:, idx[joint_name]]
        knee_ranges.append(float(values.max() - values.min()))
    return 0.5 * float(np.mean(hip_ranges)) + 0.35 * float(np.mean(knee_ranges))


def symmetrize_run_arms(
    dof_pos: np.ndarray,
    idx: dict[str, int],
    shoulder_blend: float,
    elbow_blend: float,
    ankle_pitch_max: float,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    changed: list[tuple[str, np.ndarray, np.ndarray]] = []

    l_idx = idx["left_shoulder_pitch_joint"]
    r_idx = idx["right_shoulder_pitch_joint"]
    left_before = dof_pos[:, l_idx].copy()
    right_before = dof_pos[:, r_idx].copy()

    left_center = float(left_before.mean())
    right_center = float(right_before.mean())
    shared_center = 0.5 * (left_center + right_center)
    left_dev = left_before - left_center
    right_dev = -(right_before - right_center)
    shared_dev = 0.5 * (left_dev + right_dev)

    left_after = (1.0 - shoulder_blend) * left_before + shoulder_blend * (shared_center + shared_dev)
    right_after = (1.0 - shoulder_blend) * right_before + shoulder_blend * (shared_center - shared_dev)
    left_after = clip_joint(left_after, "left_shoulder_pitch_joint", ankle_pitch_max)
    right_after = clip_joint(right_after, "right_shoulder_pitch_joint", ankle_pitch_max)
    dof_pos[:, l_idx] = left_after
    dof_pos[:, r_idx] = right_after
    changed.append(("left_shoulder_pitch_joint [run_sym]", left_before, left_after))
    changed.append(("right_shoulder_pitch_joint [run_sym]", right_before, right_after))

    le_idx = idx["left_elbow_joint"]
    re_idx = idx["right_elbow_joint"]
    left_elbow_before = dof_pos[:, le_idx].copy()
    right_elbow_before = dof_pos[:, re_idx].copy()
    elbow_shared = 0.5 * (left_elbow_before + right_elbow_before)
    left_elbow_after = (1.0 - elbow_blend) * left_elbow_before + elbow_blend * elbow_shared
    right_elbow_after = (1.0 - elbow_blend) * right_elbow_before + elbow_blend * elbow_shared
    left_elbow_after = clip_joint(left_elbow_after, "left_elbow_joint", ankle_pitch_max)
    right_elbow_after = clip_joint(right_elbow_after, "right_elbow_joint", ankle_pitch_max)
    dof_pos[:, le_idx] = left_elbow_after
    dof_pos[:, re_idx] = right_elbow_after
    changed.append(("left_elbow_joint [run_sym]", left_elbow_before, left_elbow_after))
    changed.append(("right_elbow_joint [run_sym]", right_elbow_before, right_elbow_after))

    return changed


def blend_pose_targets(
    dof_pos: np.ndarray,
    idx: dict[str, int],
    target_pose: dict[str, float],
    frame_slice: slice,
    blend: float | np.ndarray,
    ankle_pitch_max: float,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    frame_indices = np.arange(dof_pos.shape[0])[frame_slice]
    if frame_indices.size == 0:
        return []
    if np.isscalar(blend):
        blend_values = np.full((frame_indices.size, 1), float(blend), dtype=dof_pos.dtype)
    else:
        blend_array = np.asarray(blend, dtype=dof_pos.dtype)
        if blend_array.shape[0] != frame_indices.size:
            raise ValueError("Blend array length must match selected frame count.")
        blend_values = blend_array.reshape(-1, 1)

    changed: list[tuple[str, np.ndarray, np.ndarray]] = []
    for joint_name, target_value in target_pose.items():
        joint_idx = idx[joint_name]
        before = dof_pos[frame_indices, joint_idx].copy()
        after = (1.0 - blend_values[:, 0]) * before + blend_values[:, 0] * target_value
        after = clip_joint(after, joint_name, ankle_pitch_max)
        dof_pos[frame_indices, joint_idx] = after
        changed.append((joint_name, before, after))
    return changed


def scale_planar_root_trajectory(root_pos: np.ndarray, scale: float) -> np.ndarray:
    scaled_root_pos = root_pos.copy()
    planar_delta = np.diff(root_pos[:, :2], axis=0)
    scaled_root_pos[1:, :2] = root_pos[0, :2] + np.cumsum(planar_delta * scale, axis=0)
    return scaled_root_pos


def normalize_quaternions(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat / np.clip(norm, 1e-8, None)


def yaw_only_quaternions(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = normalize_quaternions(quat_wxyz)
    w = quat_wxyz[:, 0]
    x = quat_wxyz[:, 1]
    y = quat_wxyz[:, 2]
    z = quat_wxyz[:, 3]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    yaw_half = 0.5 * yaw
    yaw_only = np.zeros_like(quat_wxyz)
    yaw_only[:, 0] = np.cos(yaw_half)
    yaw_only[:, 3] = np.sin(yaw_half)
    return yaw_only


def blend_root_upright(root_rot: np.ndarray, frame_slice: slice, blend: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame_indices = np.arange(root_rot.shape[0])[frame_slice]
    if frame_indices.size == 0:
        return np.empty((0, 4), dtype=root_rot.dtype), np.empty((0, 4), dtype=root_rot.dtype)
    if np.isscalar(blend):
        blend_values = np.full((frame_indices.size, 1), float(blend), dtype=root_rot.dtype)
    else:
        blend_array = np.asarray(blend, dtype=root_rot.dtype)
        if blend_array.shape[0] != frame_indices.size:
            raise ValueError("Blend array length must match selected frame count.")
        blend_values = blend_array.reshape(-1, 1)

    before = root_rot[frame_indices].copy()
    upright_target = yaw_only_quaternions(before)
    after = normalize_quaternions((1.0 - blend_values) * before + blend_values * upright_target)
    root_rot[frame_indices] = after
    return before, after


def clip_joint(values: np.ndarray, joint_name: str, ankle_pitch_max: float) -> np.ndarray:
    lower, upper = JOINT_LIMITS[joint_name]
    if "ankle_pitch_joint" in joint_name:
        upper = min(upper, ankle_pitch_max)
    return np.clip(values, lower, upper)


def summarize_joint(values: np.ndarray) -> tuple[float, float, float]:
    return float(values.min()), float(values.max()), float(values.mean())


def format_stats(label: str, before: np.ndarray, after: np.ndarray) -> str:
    b_min, b_max, b_mean = summarize_joint(before)
    a_min, a_max, a_mean = summarize_joint(after)
    return (
        f"  {label}: "
        f"before[min={b_min:.3f}, max={b_max:.3f}, mean={b_mean:.3f}] "
        f"after[min={a_min:.3f}, max={a_max:.3f}, mean={a_mean:.3f}]"
    )


def postprocess_clip(motion_data: dict, clip_name: str, args: argparse.Namespace) -> tuple[dict, list[str]]:
    processed = deepcopy(motion_data)
    dof_names = processed["dof_names"]
    dof_pos = processed["dof_pos"].copy()
    root_pos = processed["root_pos"].copy()
    root_rot = processed["root_rot"].copy()
    idx = {name: i for i, name in enumerate(dof_names)}
    clip_flags = classify_clip(clip_name, root_pos)
    stats_lines: list[str] = []

    shoulder_joints = ["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"]
    hip_pitch_joints = ["left_hip_pitch_joint", "right_hip_pitch_joint"]
    elbow_joints = ["left_elbow_joint", "right_elbow_joint"]
    hip_roll_pairs = [("left_hip_roll_joint", 1.0), ("right_hip_roll_joint", -1.0)]
    hip_yaw_joints = ["left_hip_yaw_joint", "right_hip_yaw_joint"]
    ankle_pitch_joints = ["left_ankle_pitch_joint", "right_ankle_pitch_joint"]
    leg_score = leg_motion_score(dof_pos, idx) if (clip_flags["is_walk_turn"] or clip_flags["is_c_series"]) else 0.0

    shoulder_backward_scale = args.run_backward_scale if clip_flags["is_run"] else args.arm_backward_scale
    shoulder_forward_bias = args.run_forward_bias if clip_flags["is_run"] else 0.0
    shoulder_forward_scale = args.run_forward_scale if clip_flags["is_run"] else args.arm_forward_scale
    numeric_relaxed_clips = {"02_02", "16_34"}
    is_numeric_special = clip_flags["is_numeric_labeled"] and clip_name not in numeric_relaxed_clips
    if clip_flags["is_numeric_labeled"]:
        shoulder_forward_scale *= args.numeric_forward_scale
        shoulder_forward_bias += args.numeric_forward_bias
    elbow_bend = args.run_elbow_bend if clip_flags["is_run"] else args.elbow_bend

    if clip_flags["is_dynamic"]:
        for joint_name in shoulder_joints:
            joint_idx = idx[joint_name]
            before = dof_pos[:, joint_idx].copy()
            after = scale_shoulder_pitch(
                before,
                center=args.shoulder_center,
                forward_scale=shoulder_forward_scale,
                backward_scale=shoulder_backward_scale,
                forward_bias=shoulder_forward_bias,
            )
            if is_numeric_special:
                after = scale_about_mean(after, args.numeric_motion_gain)
            if clip_flags["is_walk_turn"]:
                after = scale_about_mean(after, 1.0 + args.walk_turn_leg_arm_coupling * leg_score)
                after = enforce_min_motion_range(
                    after,
                    min_range=args.walk_turn_shoulder_min_range,
                    max_gain=args.walk_turn_shoulder_max_gain,
                )
                after = after - args.walk_turn_shoulder_forward_bias
            if clip_flags["is_c_series"]:
                after = scale_about_mean(after, 1.0 + args.c_series_leg_arm_coupling * leg_score)
                after = enforce_min_motion_range(
                    after,
                    min_range=args.c_series_shoulder_min_range,
                    max_gain=args.c_series_shoulder_max_gain,
                )
                after = after - args.c_series_shoulder_forward_bias
            dof_pos[:, joint_idx] = clip_joint(after, joint_name, args.ankle_pitch_max)
            stats_lines.append(format_stats(joint_name, before, dof_pos[:, joint_idx]))

        if clip_flags["is_numeric_labeled"] and clip_name != "02_02":
            for joint_name in hip_pitch_joints:
                joint_idx = idx[joint_name]
                before = dof_pos[:, joint_idx].copy()
                after = scale_hip_pitch_forward(
                    before,
                    forward_scale=args.numeric_hip_forward_scale,
                    forward_bias=args.numeric_hip_forward_bias,
                )
                dof_pos[:, joint_idx] = clip_joint(after, joint_name, args.ankle_pitch_max)
                stats_lines.append(format_stats(joint_name, before, dof_pos[:, joint_idx]))

        for joint_name in elbow_joints:
            joint_idx = idx[joint_name]
            before = dof_pos[:, joint_idx].copy()
            after = before - elbow_bend
            dof_pos[:, joint_idx] = clip_joint(after, joint_name, args.ankle_pitch_max)
            stats_lines.append(format_stats(joint_name, before, dof_pos[:, joint_idx]))

        for joint_name, direction in hip_roll_pairs:
            joint_idx = idx[joint_name]
            before = dof_pos[:, joint_idx].copy()
            after = before * args.hip_roll_width_scale + direction * args.hip_roll_width_offset
            dof_pos[:, joint_idx] = clip_joint(after, joint_name, args.ankle_pitch_max)
            stats_lines.append(format_stats(joint_name, before, dof_pos[:, joint_idx]))

    if clip_flags["is_turn"]:
        before_root = root_pos.copy()
        root_pos = scale_planar_root_trajectory(root_pos, args.turn_root_scale)
        root_span_before = np.ptp(before_root[:, :2], axis=0)
        root_span_after = np.ptp(root_pos[:, :2], axis=0)
        stats_lines.append(
            "  root_xy_span: "
            f"before[x={root_span_before[0]:.3f}, y={root_span_before[1]:.3f}] "
            f"after[x={root_span_after[0]:.3f}, y={root_span_after[1]:.3f}]"
        )
        for joint_name in hip_yaw_joints:
            joint_idx = idx[joint_name]
            before = dof_pos[:, joint_idx].copy()
            center = float(before.mean())
            after = center + (before - center) * args.turn_hip_yaw_scale
            dof_pos[:, joint_idx] = clip_joint(after, joint_name, args.ankle_pitch_max)
            stats_lines.append(format_stats(joint_name, before, dof_pos[:, joint_idx]))

    for joint_name in ankle_pitch_joints:
        joint_idx = idx[joint_name]
        before = dof_pos[:, joint_idx].copy()
        after = clip_joint(before, joint_name, args.ankle_pitch_max)
        dof_pos[:, joint_idx] = after
        stats_lines.append(format_stats(joint_name, before, after))

    if clip_flags["is_run"]:
        for joint_name, before, after in symmetrize_run_arms(
            dof_pos,
            idx,
            shoulder_blend=args.run_symmetry_blend,
            elbow_blend=args.run_elbow_symmetry_blend,
            ankle_pitch_max=args.ankle_pitch_max,
        ):
            stats_lines.append(format_stats(joint_name, before, after))

    # Clip-specific cleanups to preserve nicer poses after the generic pass.
    if clip_name == "A1-_Stand_stageii":
        for joint_name, before, after in blend_pose_targets(
            dof_pos, idx, STAND_POSE_TARGET, slice(None), blend=0.90, ankle_pitch_max=args.ankle_pitch_max
        ):
            stats_lines.append(format_stats(f"{joint_name} [stand_pose]", before, after))

    if clip_name == "02_02":
        for joint_name, before, after in blend_pose_targets(
            dof_pos, idx, LOWER_ARM_POSE_TARGET_0202, slice(None), blend=0.55, ankle_pitch_max=args.ankle_pitch_max
        ):
            stats_lines.append(format_stats(f"{joint_name} [02_02_arm]", before, after))
        for joint_name in hip_pitch_joints:
            joint_idx = idx[joint_name]
            before = dof_pos[:, joint_idx].copy()
            after = scale_about_mean(before, args.clip_0202_hip_pitch_scale)
            dof_pos[:, joint_idx] = clip_joint(after, joint_name, args.ankle_pitch_max)
            stats_lines.append(format_stats(f"{joint_name} [02_02_hip_pitch_damp]", before, dof_pos[:, joint_idx]))

    if clip_name == "16_34":
        for joint_name, before, after in blend_pose_targets(
            dof_pos, idx, LOWER_ARM_POSE_TARGET_1634, slice(None), blend=0.68, ankle_pitch_max=args.ankle_pitch_max
        ):
            stats_lines.append(format_stats(f"{joint_name} [16_34_arm]", before, after))
        tail_len = min(60, dof_pos.shape[0])
        tail_blend = np.linspace(0.25, 0.95, tail_len, dtype=dof_pos.dtype)
        for joint_name, before, after in blend_pose_targets(
            dof_pos,
            idx,
            FINAL_1634_POSE_TARGET,
            slice(dof_pos.shape[0] - tail_len, dof_pos.shape[0]),
            blend=tail_blend,
            ankle_pitch_max=args.ankle_pitch_max,
        ):
            stats_lines.append(format_stats(f"{joint_name} [16_34_tail]", before, after))
        root_before, root_after = blend_root_upright(
            root_rot,
            slice(dof_pos.shape[0] - tail_len, dof_pos.shape[0]),
            blend=np.linspace(0.20, 0.85, tail_len, dtype=root_rot.dtype),
        )
        if root_before.size:
            stats_lines.append(
                "  root_rot [16_34_tail_upright]: "
                f"before_last={np.array2string(root_before[-1], precision=4, suppress_small=True)} "
                f"after_last={np.array2string(root_after[-1], precision=4, suppress_small=True)}"
            )

    for joint_name in shoulder_joints:
        joint_idx = idx[joint_name]
        before = dof_pos[:, joint_idx].copy()
        after = np.clip(before, args.shoulder_pitch_min, args.shoulder_pitch_max)
        after = clip_joint(after, joint_name, args.ankle_pitch_max)
        dof_pos[:, joint_idx] = after
        stats_lines.append(format_stats(f"{joint_name} [shoulder_pitch_range]", before, after))

    processed["dof_pos"] = dof_pos
    processed["root_pos"] = root_pos
    processed["root_rot"] = root_rot
    processed.setdefault("retarget_info", {})
    processed["retarget_info"]["postprocess"] = {
        "tool": "postprocess_lens110_motion.py",
        "clip_name": clip_name,
        "notes": [
            "Increased arm swing while reducing excessive run backswing.",
            "Added elbow flexion and wider hip roll stance for dynamic clips.",
            "Reduced turn root trajectory span and hip yaw excursion for turn clips.",
            "Clamped ankle_pitch_joint to avoid upward toe lift.",
            "key_body_pos was left unchanged; recompute if downstream tasks rely on key-body targets.",
        ],
        "parameters": {
            "shoulder_center": args.shoulder_center,
            "shoulder_pitch_max": args.shoulder_pitch_max,
            "shoulder_pitch_min": args.shoulder_pitch_min,
            "arm_forward_scale": args.arm_forward_scale,
            "arm_backward_scale": args.arm_backward_scale,
            "run_forward_scale": args.run_forward_scale,
            "run_backward_scale": args.run_backward_scale,
            "run_forward_bias": args.run_forward_bias,
            "run_symmetry_blend": args.run_symmetry_blend,
            "run_elbow_symmetry_blend": args.run_elbow_symmetry_blend,
            "numeric_forward_bias": args.numeric_forward_bias,
            "numeric_forward_scale": args.numeric_forward_scale,
            "numeric_motion_gain": args.numeric_motion_gain,
            "numeric_hip_forward_bias": args.numeric_hip_forward_bias,
            "numeric_hip_forward_scale": args.numeric_hip_forward_scale,
            "clip_0202_hip_pitch_scale": args.clip_0202_hip_pitch_scale,
            "walk_turn_shoulder_min_range": args.walk_turn_shoulder_min_range,
            "walk_turn_shoulder_max_gain": args.walk_turn_shoulder_max_gain,
            "walk_turn_shoulder_forward_bias": args.walk_turn_shoulder_forward_bias,
            "walk_turn_leg_arm_coupling": args.walk_turn_leg_arm_coupling,
            "c_series_shoulder_min_range": args.c_series_shoulder_min_range,
            "c_series_shoulder_max_gain": args.c_series_shoulder_max_gain,
            "c_series_shoulder_forward_bias": args.c_series_shoulder_forward_bias,
            "c_series_leg_arm_coupling": args.c_series_leg_arm_coupling,
            "elbow_bend": args.elbow_bend,
            "run_elbow_bend": args.run_elbow_bend,
            "hip_roll_width_offset": args.hip_roll_width_offset,
            "hip_roll_width_scale": args.hip_roll_width_scale,
            "turn_hip_yaw_scale": args.turn_hip_yaw_scale,
            "turn_root_scale": args.turn_root_scale,
            "ankle_pitch_max": args.ankle_pitch_max,
        },
    }
    return processed, stats_lines


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    motion_files = sorted(input_dir.glob("*.pkl"))
    if not motion_files:
        raise FileNotFoundError(f"No .pkl motion clips found in {input_dir}")

    if not args.report_only:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Report only:      {args.report_only}")

    for motion_path in motion_files:
        motion_data = joblib.load(motion_path)
        processed, stats_lines = postprocess_clip(motion_data, motion_path.stem, args)
        print(f"\n[{motion_path.name}]")
        for line in stats_lines:
            print(line)
        if not args.report_only:
            joblib.dump(processed, output_dir / motion_path.name)


if __name__ == "__main__":
    main()
