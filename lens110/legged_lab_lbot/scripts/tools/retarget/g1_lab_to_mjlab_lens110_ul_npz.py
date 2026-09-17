"""Convert Legged Lab G1 pickle motions to MJLab Lens110 upper/lower npz files.

The pipeline is:
1. Reuse the existing G1 29DoF -> Lens110 21DoF IK retargeter.
2. Insert Lens110 ankle upper/lower joints with the weighted MLP PR->UL state model.
3. Run MuJoCo forward kinematics and save the npz schema used by AMP_mjlab.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch

import g1_lab_to_lens110_scaled_ik as g1_to_lens


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = next(
    (
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "projects").is_dir() and (parent / "frameworks").is_dir()
    ),
    REPO_ROOT,
)
DEFAULT_SOURCE_ROOT = REPO_ROOT / "source/legged_lab/legged_lab/data/MotionData/g1_lab"
DEFAULT_XML = REPO_ROOT / "source/legged_lab/legged_lab/data/Robots/model_humanoid_lens110/mjcf/lens110.xml"
DEFAULT_OUTPUT_ROOT = Path(
    WORKSPACE_ROOT
    / "projects/04_fall_to_stand/framework/amp_mjlab/AMP_mjlab/src/assets/motions/lens110_upper_lower_direct/WalkandRun_g1_lab"
)
DEFAULT_MLP_MODEL_DIR = REPO_ROOT / "pitchRoll2UpperLower/t_p_v/models/weighted_mlp"

MJLAB_LENS110_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "left_ankle_upper_joint",
    "left_ankle_lower_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "right_ankle_upper_joint",
    "right_ankle_lower_joint",
    "torso_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
)

PITCH_ROLL_JOINT_NAMES = (
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)

UPPER_LOWER_JOINT_NAMES = (
    "left_ankle_upper_joint",
    "left_ankle_lower_joint",
    "right_ankle_upper_joint",
    "right_ankle_lower_joint",
)

MJLAB_JOINT_SIGN_OVERRIDES = {
    # The Legged Lab retargeter emits the right elbow in the URDF/training sign.
    # MJLab's Lens110 MJCF uses the opposite sign; the reference MJLab motions
    # have right_elbow positive while left_elbow is negative.
    "right_elbow_joint": -1.0,
}

UL_TRAINING_LIMITS = np.asarray(
    [
        [-0.70, 1.60],
        [-0.55, 1.50],
        [-0.75, 1.60],
        [-0.60, 1.50],
    ],
    dtype=np.float32,
)


def _as_float_fps(value) -> float:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return float(array[0])


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat / np.maximum(norm, 1e-12)


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def _quat_inv_wxyz(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def _finite_diff(values: np.ndarray, dt: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    vel = np.zeros_like(values)
    if values.shape[0] <= 1:
        return vel
    vel[:-1] = (values[1:] - values[:-1]) / dt
    vel[-1] = vel[-2]
    return vel


def _angular_velocity_wxyz(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    quat = _normalize_quat_wxyz(quat_wxyz)
    omega = np.zeros((quat.shape[0], 3), dtype=np.float64)
    if quat.shape[0] <= 1:
        return omega
    delta = _quat_mul_wxyz(quat[1:], _quat_inv_wxyz(quat[:-1]))
    delta = _normalize_quat_wxyz(delta)
    delta[delta[:, 0] < 0.0] *= -1.0
    rotvec = Rotation.from_quat(delta[:, [1, 2, 3, 0]]).as_rotvec()
    omega[:-1] = rotvec / dt
    omega[-1] = omega[-2]
    return omega


def _resample_motion(motion: dict, output_fps: float | None) -> dict:
    input_fps = _as_float_fps(motion["fps"])
    if output_fps is None or abs(output_fps - input_fps) < 1e-6:
        return motion

    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    root_rot = _normalize_quat_wxyz(np.asarray(motion["root_rot"], dtype=np.float64))
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)
    duration = (dof_pos.shape[0] - 1) / input_fps
    in_times = np.arange(dof_pos.shape[0], dtype=np.float64) / input_fps
    out_times = np.arange(0.0, duration + 1e-9, 1.0 / output_fps)
    out_times = np.clip(out_times, in_times[0], in_times[-1])

    root_pos_out = np.stack([np.interp(out_times, in_times, root_pos[:, i]) for i in range(3)], axis=-1)
    dof_out = np.stack([np.interp(out_times, in_times, dof_pos[:, i]) for i in range(dof_pos.shape[1])], axis=-1)
    slerp = Slerp(in_times, Rotation.from_quat(root_rot[:, [1, 2, 3, 0]]))
    root_rot_out = slerp(out_times).as_quat()[:, [3, 0, 1, 2]]

    out = dict(motion)
    out["fps"] = float(output_fps)
    out["root_pos"] = root_pos_out.astype(np.float32)
    out["root_rot"] = _normalize_quat_wxyz(root_rot_out).astype(np.float32)
    out["dof_pos"] = dof_out.astype(np.float32)
    return out


class WeightedMlpAnkleConverter:
    def __init__(self, model_dir: Path, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = torch.jit.load(str(model_dir / "pr_to_ul_state" / "scripted.pth"), map_location=self.device).eval()

    def pr_to_ul_state(self, pr_pos: np.ndarray, pr_vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.concatenate((pr_pos, pr_vel), axis=-1).astype(np.float32)
        with torch.no_grad():
            y = self.model(torch.from_numpy(x).to(self.device)).cpu().numpy()
        return y[:, 0:4].astype(np.float32), y[:, 4:8].astype(np.float32)


def _insert_upper_lower_joints_mlp(
    motion: dict,
    joint_vel_pr_source: np.ndarray,
    converter: WeightedMlpAnkleConverter,
    clip_ul_to_training_limits: bool,
) -> tuple[np.ndarray, np.ndarray]:
    source_names = list(motion["dof_names"])
    source_index = {name: i for i, name in enumerate(source_names)}
    pr = np.stack([motion["dof_pos"][:, source_index[name]] for name in PITCH_ROLL_JOINT_NAMES], axis=-1)
    pr_vel = np.stack([joint_vel_pr_source[:, source_index[name]] for name in PITCH_ROLL_JOINT_NAMES], axis=-1)
    ul, ul_vel = converter.pr_to_ul_state(pr.astype(np.float32), pr_vel.astype(np.float32))
    if clip_ul_to_training_limits:
        ul = np.clip(ul, UL_TRAINING_LIMITS[:, 0], UL_TRAINING_LIMITS[:, 1])

    values_by_name = {name: motion["dof_pos"][:, i] for i, name in enumerate(source_names)}
    vel_by_name = {name: joint_vel_pr_source[:, i] for i, name in enumerate(source_names)}
    for i, name in enumerate(UPPER_LOWER_JOINT_NAMES):
        values_by_name[name] = ul[:, i]
        vel_by_name[name] = ul_vel[:, i]

    missing = [name for name in MJLAB_LENS110_JOINT_NAMES if name not in values_by_name]
    if missing:
        raise ValueError(f"Cannot build MJLab joint_pos, missing joints: {missing}")
    joint_pos = np.stack([values_by_name[name] for name in MJLAB_LENS110_JOINT_NAMES], axis=-1).astype(np.float32)
    joint_vel = np.stack([vel_by_name[name] for name in MJLAB_LENS110_JOINT_NAMES], axis=-1).astype(np.float32)
    for name, sign in MJLAB_JOINT_SIGN_OVERRIDES.items():
        index = MJLAB_LENS110_JOINT_NAMES.index(name)
        joint_pos[:, index] *= sign
        joint_vel[:, index] *= sign
    return joint_pos, joint_vel


def _fk_body_arrays(
    model: mujoco.MjModel,
    root_pos: np.ndarray,
    root_rot: np.ndarray,
    joint_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    qpos_addr = {}
    for name in MJLAB_LENS110_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Joint {name} not found in {DEFAULT_XML}")
        qpos_addr[name] = int(model.jnt_qposadr[joint_id])

    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody)]
    body_pos = np.zeros((joint_pos.shape[0], len(body_names), 3), dtype=np.float32)
    body_quat = np.zeros((joint_pos.shape[0], len(body_names), 4), dtype=np.float32)

    for frame in range(joint_pos.shape[0]):
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.qpos[:3] = root_pos[frame]
        data.qpos[3:7] = _normalize_quat_wxyz(root_rot[frame])
        for joint_idx, name in enumerate(MJLAB_LENS110_JOINT_NAMES):
            data.qpos[qpos_addr[name]] = joint_pos[frame, joint_idx]
        mujoco.mj_forward(model, data)
        body_pos[frame] = data.xpos[1:].astype(np.float32)
        body_quat[frame] = data.xquat[1:].astype(np.float32)

    return body_pos, body_quat


def _convert_one(
    motion_file: Path,
    source_tree,
    target_tree,
    target_dof_names: list[str],
    converter: WeightedMlpAnkleConverter,
    model: mujoco.MjModel,
    args: argparse.Namespace,
) -> dict:
    source_motion = g1_to_lens._load_pickle(motion_file)
    target_motion = g1_to_lens._retarget_motion(
        motion_dict=source_motion,
        clip_name=motion_file.stem,
        source_tree=source_tree,
        target_tree=target_tree,
        target_dof_names=target_dof_names,
        stride_scale_factor=args.stride_scale_factor,
        root_upright_blend=args.root_upright_blend,
        stand_root_upright_blend=args.stand_root_upright_blend,
        stand_root_position_blend=args.stand_root_position_blend,
        root_smoothing_window=args.root_smoothing_window,
        arm_smoothing_window=args.arm_smoothing_window,
        leg_max_step=args.leg_max_step,
        arm_max_step=args.arm_max_step,
    )
    target_motion = _resample_motion(target_motion, args.output_fps)

    fps = _as_float_fps(target_motion["fps"])
    dt = 1.0 / fps
    root_pos = np.asarray(target_motion["root_pos"], dtype=np.float32)
    root_rot = _normalize_quat_wxyz(np.asarray(target_motion["root_rot"], dtype=np.float32)).astype(np.float32)
    source_joint_vel = _finite_diff(np.asarray(target_motion["dof_pos"], dtype=np.float32), dt).astype(np.float32)
    joint_pos, joint_vel = _insert_upper_lower_joints_mlp(
        target_motion, source_joint_vel, converter, args.clip_ul_to_training_limits
    )
    body_pos, body_quat = _fk_body_arrays(model, root_pos, root_rot, joint_pos)

    return {
        "fps": np.asarray([fps], dtype=np.float64),
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": _finite_diff(body_pos, dt).astype(np.float32),
        "body_ang_vel_w": _angular_velocity_wxyz(body_quat.reshape(-1, 4), dt)
        .reshape(body_quat.shape[0], body_quat.shape[1], 3)
        .astype(np.float32),
        "joint_names": np.asarray(MJLAB_LENS110_JOINT_NAMES),
        "body_names": np.asarray([mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody)]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--source_urdf",
        type=Path,
        default=WORKSPACE_ROOT / "tools/retargeting/gmr_lens110/assets/unitree_g1/g1_custom_collision_29dof.urdf",
    )
    parser.add_argument("--target_urdf", type=Path, default=REPO_ROOT / "source/legged_lab/legged_lab/data/Robots/model_humanoid_lens110/lens110_21dof.urdf")
    parser.add_argument("--target_xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--mlp_model_dir", type=Path, default=DEFAULT_MLP_MODEL_DIR)
    parser.add_argument("--output_fps", type=float, default=50.0)
    parser.add_argument("--clip_ul_to_training_limits", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clean_output", action="store_true")
    parser.add_argument("--stride_scale_factor", type=float, default=0.9)
    parser.add_argument("--root_upright_blend", type=float, default=0.10)
    parser.add_argument("--stand_root_upright_blend", type=float, default=0.92)
    parser.add_argument("--stand_root_position_blend", type=float, default=0.85)
    parser.add_argument("--root_smoothing_window", type=int, default=5)
    parser.add_argument("--arm_smoothing_window", type=int, default=5)
    parser.add_argument("--leg_max_step", type=float, default=0.18)
    parser.add_argument("--arm_max_step", type=float, default=0.10)
    args = parser.parse_args()

    motion_files = sorted(args.source_root.glob("*.pkl"))
    if not motion_files:
        raise FileNotFoundError(f"No .pkl files found in {args.source_root}")

    if args.clean_output and args.output_root.exists():
        for file in args.output_root.glob("*.npz"):
            file.unlink()
    args.output_root.mkdir(parents=True, exist_ok=True)

    source_tree = g1_to_lens.KinematicTree(args.source_urdf)
    target_tree = g1_to_lens.KinematicTree(args.target_urdf)
    target_dof_names = target_tree.joint_order
    if len(target_dof_names) != 21:
        raise ValueError(f"Expected 21 target Lens110 DoFs, got {len(target_dof_names)}")

    converter = WeightedMlpAnkleConverter(args.mlp_model_dir)
    model = mujoco.MjModel.from_xml_path(str(args.target_xml))

    for index, motion_file in enumerate(motion_files, start=1):
        npz = _convert_one(motion_file, source_tree, target_tree, target_dof_names, converter, model, args)
        output_path = args.output_root / f"{motion_file.stem}.npz"
        np.savez_compressed(output_path, **npz)
        print(f"[{index:02d}/{len(motion_files):02d}] {motion_file.name} -> {output_path.name} frames={npz['joint_pos'].shape[0]}")

    print(f"[DONE] Wrote {len(motion_files)} npz files to {args.output_root}")


if __name__ == "__main__":
    main()
