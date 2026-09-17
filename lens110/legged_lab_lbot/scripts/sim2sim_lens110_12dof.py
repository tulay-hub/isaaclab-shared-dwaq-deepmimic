# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lens110 lower-body 12-DoF sim2sim in MuJoCo.

This entry point mirrors the current Lens110 lower-body velocity task:
12 policy actions drive the leg joints directly, while torso and arms are held
at their default pose with a separate PD target.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

if "--headless" in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

import mujoco
import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is only cosmetic.
    tqdm = lambda x, **_: x

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = next(
    (
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "projects").is_dir() and (parent / "frameworks").is_dir()
    ),
    ROOT,
)
PROJECT02_FRAMEWORK_ROOT = WORKSPACE_ROOT / "projects/02_dance_half_body/framework/legged_lab_upper_lower"
DEFAULT_MJCF_CANDIDATES = [
    ROOT / "source/legged_lab/legged_lab/data/Robots/model_humanoid_lens110/mjcf/lens110_21dof.xml",
    PROJECT02_FRAMEWORK_ROOT / "source/legged_lab/legged_lab/data/Robots/model_humanoid_lens110/mjcf/lens110_21dof.xml",
]
DEFAULT_POLICY_ROOT = PROJECT02_FRAMEWORK_ROOT / "logs/rsl_rl/lens110_amp_leg_pitch_roll"

LEG_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

HOLD_JOINT_NAMES = [
    "torso_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]

DEFAULT_JOINT_POS = {
    "left_hip_pitch_joint": -0.14,
    "left_hip_roll_joint": 0.01,
    "left_hip_yaw_joint": -0.10,
    "left_knee_joint": 0.30,
    "left_ankle_pitch_joint": -0.15,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.14,
    "right_hip_roll_joint": -0.01,
    "right_hip_yaw_joint": 0.10,
    "right_knee_joint": 0.30,
    "right_ankle_pitch_joint": -0.15,
    "right_ankle_roll_joint": 0.0,
    "torso_yaw_joint": 0.0,
    "left_shoulder_pitch_joint": 0.4,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": -0.8,
    "right_shoulder_pitch_joint": 0.4,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": -0.8,
}

KP = {
    ".*_hip_.*": 40.0,
    ".*_knee_joint": 40.0,
    ".*_ankle_pitch_joint": 14.5,
    ".*_ankle_roll_joint": 7.0,
    "torso_yaw_joint": 100.0,
    ".*_shoulder_.*": 20.0,
    ".*_elbow_joint": 20.0,
}

KD = {
    ".*_hip_.*": 5.0,
    ".*_knee_joint": 5.0,
    ".*_ankle_pitch_joint": 14.5,
    ".*_ankle_roll_joint": 7.0,
    "torso_yaw_joint": 5.0,
    ".*_shoulder_.*": 1.0,
    ".*_elbow_joint": 1.0,
}

TAU_LIMIT = {
    ".*_hip_pitch_joint": 80.0,
    ".*_hip_roll_joint": 80.0,
    ".*_hip_yaw_joint": 36.0,
    ".*_knee_joint": 80.0,
    ".*_ankle_.*": 36.0,
    "torso_yaw_joint": 36.0,
    ".*_shoulder_.*": 36.0,
    ".*_elbow_joint": 36.0,
}

TRAIN_DT = 0.005
TRAIN_DECIMATION = 4
ACTION_SCALE = 0.25
DEFAULT_ROOT_HEIGHT = 0.68
FOOT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
FOOT_CONTACT_GEOMS = [
    "left_ankle_roll_collision",
    "right_ankle_roll_collision",
    "left_ankle_pitch_collision",
    "right_ankle_pitch_collision",
]
TRAINING_FOOT_BOXES = {
    "left_ankle_roll_collision": ((0.019, 0.009, -0.030), (0.0975, 0.0475, 0.010)),
    "right_ankle_roll_collision": ((0.019, -0.009, -0.030), (0.0975, 0.0475, 0.010)),
    "left_ankle_pitch_collision": ((0.019, 0.009, -0.030), (0.0975, 0.0475, 0.010)),
    "right_ankle_pitch_collision": ((0.019, -0.009, -0.030), (0.0975, 0.0475, 0.010)),
}
SHOULDER_PITCH_GAIN = -2.2
SHOULDER_PITCH_LIMIT = 0.55
ELBOW_GAIN = 1.0
ELBOW_LIMIT = 0.25
ARM_SWING_SMOOTHING = 0.35


def resolve_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (ROOT / p).resolve()


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def regex_value(table: dict[str, float], name: str) -> float:
    for pattern, value in table.items():
        if re.fullmatch(pattern, name):
            return float(value)
    raise KeyError(f"No value configured for joint {name}")


def name_to_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise ValueError(f"Missing {obj_type.name}: {name}")
    return obj_id


class JointSet:
    def __init__(self, model: mujoco.MjModel, names: list[str]):
        self.names = names
        self.joint_ids = np.array([name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names])
        self.qpos_ids = model.jnt_qposadr[self.joint_ids].astype(np.int32)
        self.qvel_ids = model.jnt_dofadr[self.joint_ids].astype(np.int32)
        self.actuator_ids = self._find_actuators(model)

    def _find_actuators(self, model: mujoco.MjModel) -> np.ndarray:
        actuator_ids = np.full(len(self.names), -1, dtype=np.int32)
        for actuator_id in range(model.nu):
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            matches = np.where(self.joint_ids == joint_id)[0]
            if len(matches):
                actuator_ids[matches[0]] = actuator_id
        if np.any(actuator_ids < 0):
            missing = [name for name, actuator_id in zip(self.names, actuator_ids, strict=True) if actuator_id < 0]
            raise ValueError(f"Missing actuators for joints: {missing}")
        return actuator_ids

    def qpos(self, data: mujoco.MjData) -> np.ndarray:
        return data.qpos[self.qpos_ids].copy()

    def qvel(self, data: mujoco.MjData) -> np.ndarray:
        return data.qvel[self.qvel_ids].copy()


class SensorIMU:
    def __init__(self, model: mujoco.MjModel):
        self.orientation = self._sensor(model, "orientation", 4)
        self.gyro = self._sensor(model, "angular-velocity", 3)
        self.lin_vel = self._optional_sensor(model, "linear-velocity", 3)

    @staticmethod
    def _sensor(model: mujoco.MjModel, name: str, dim: int) -> tuple[int, int]:
        sensor_id = name_to_id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        actual_dim = int(model.sensor_dim[sensor_id])
        if actual_dim < dim:
            raise ValueError(f"Sensor {name} dim={actual_dim}, expected at least {dim}")
        return int(model.sensor_adr[sensor_id]), dim

    @staticmethod
    def _optional_sensor(model: mujoco.MjModel, name: str, dim: int) -> tuple[int, int] | None:
        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sensor_id < 0 or int(model.sensor_dim[sensor_id]) < dim:
            return None
        return int(model.sensor_adr[sensor_id]), dim

    def quat(self, data: mujoco.MjData) -> np.ndarray:
        adr, dim = self.orientation
        quat = data.sensordata[adr : adr + dim].copy()
        return quat / max(np.linalg.norm(quat), 1.0e-12)

    def gyro_body(self, data: mujoco.MjData) -> np.ndarray:
        adr, dim = self.gyro
        return data.sensordata[adr : adr + dim].copy()

    def linear_velocity_body(self, data: mujoco.MjData) -> np.ndarray:
        if self.lin_vel is not None:
            adr, dim = self.lin_vel
            return data.sensordata[adr : adr + dim].copy()
        return quat_to_rotmat_wxyz(self.quat(data)).T @ data.qvel[:3]


def quat_to_rotmat_wxyz(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def joint_ranges(model: mujoco.MjModel, joints: JointSet) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(len(joints.names), -np.inf, dtype=np.float64)
    upper = np.full(len(joints.names), np.inf, dtype=np.float64)
    for i, joint_id in enumerate(joints.joint_ids):
        if model.jnt_limited[joint_id]:
            lower[i], upper[i] = model.jnt_range[joint_id]
    return lower, upper


def patch_model(model: mujoco.MjModel, controlled: JointSet) -> None:
    model.opt.timestep = TRAIN_DT
    for name, (pos, size) in TRAINING_FOOT_BOXES.items():
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id >= 0:
            model.geom_pos[geom_id] = np.asarray(pos, dtype=np.float64)
            model.geom_size[geom_id] = np.asarray(size, dtype=np.float64)
            model.geom_type[geom_id] = mujoco.mjtGeom.mjGEOM_BOX

    foot_geom_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in FOOT_CONTACT_GEOMS
    }
    foot_geom_ids.discard(-1)
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        active = geom_name in {"floor", "ground", "plane"} or geom_id in foot_geom_ids
        model.geom_contype[geom_id] = int(active)
        model.geom_conaffinity[geom_id] = int(active)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id >= 0:
        model.geom_friction[floor_id] = [1.0, 0.08, 0.004]

    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        dof_id = model.jnt_dofadr[joint_id]
        model.dof_damping[dof_id] = 0.0
        model.dof_frictionloss[dof_id] = 0.1
        model.dof_armature[dof_id] = 0.01

    for i, actuator_id in enumerate(controlled.actuator_ids):
        kp = regex_value(KP, controlled.names[i])
        kd = regex_value(KD, controlled.names[i])
        limit = regex_value(TAU_LIMIT, controlled.names[i])
        model.actuator_gaintype[actuator_id] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_biastype[actuator_id] = mujoco.mjtBias.mjBIAS_AFFINE
        model.actuator_gainprm[actuator_id, :] = 0.0
        model.actuator_gainprm[actuator_id, 0] = kp
        model.actuator_biasprm[actuator_id, :] = 0.0
        model.actuator_biasprm[actuator_id, 1] = -kp
        model.actuator_biasprm[actuator_id, 2] = -kd
        model.actuator_ctrllimited[actuator_id] = 0
        model.actuator_forcelimited[actuator_id] = 1
        model.actuator_forcerange[actuator_id] = [-limit, limit]


def geom_box_zmin(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float:
    pos = data.geom_xpos[geom_id]
    mat = data.geom_xmat[geom_id].reshape(3, 3)
    size = model.geom_size[geom_id]
    zmin = np.inf
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corner = pos + mat @ (size * np.array([sx, sy, sz], dtype=np.float64))
                zmin = min(zmin, float(corner[2]))
    return zmin


def settle_base_height(model: mujoco.MjModel, data: mujoco.MjData, clearance: float = 0.002) -> None:
    zmins = []
    for name in FOOT_CONTACT_GEOMS:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id >= 0:
            zmins.append(geom_box_zmin(model, data, geom_id))
    if zmins:
        data.qpos[2] += clearance - min(zmins)
        mujoco.mj_forward(model, data)


def set_default_pose(model: mujoco.MjModel, data: mujoco.MjData, all_joints: JointSet, root_height: float) -> None:
    data.qpos[:] = 0.0
    if model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        data.qpos[:3] = [0.0, 0.0, root_height]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for name, qpos_id in zip(all_joints.names, all_joints.qpos_ids, strict=True):
        data.qpos[qpos_id] = DEFAULT_JOINT_POS[name]
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    settle_base_height(model, data)


def latest_policy(policy_root: Path) -> Path:
    candidates = []
    if policy_root.exists():
        candidates = [p for p in policy_root.glob("*/exported/policy.pt") if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No exported 12-DoF leg-pitch-roll policy.pt found under: {policy_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


class TorchScriptPolicy:
    def __init__(self, path: Path, device: torch.device):
        self.model = torch.jit.load(str(path), map_location=device)
        self.model.eval()
        self.device = device

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.from_numpy(obs.astype(np.float32, copy=False)).to(self.device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
        with torch.inference_mode():
            action = self.model(obs_t)
        return action.squeeze(0).detach().cpu().numpy()


def build_observation(
    data: mujoco.MjData,
    leg_joints: JointSet,
    imu: SensorIMU,
    default_leg_pos: np.ndarray,
    last_action: np.ndarray,
    command: np.ndarray,
) -> np.ndarray:
    rot = quat_to_rotmat_wxyz(imu.quat(data))
    projected_gravity = rot.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return np.concatenate(
        [
        imu.gyro_body(data).astype(np.float32),
        projected_gravity.astype(np.float32),
        command.astype(np.float32),
        (leg_joints.qpos(data) - default_leg_pos).astype(np.float32),
        leg_joints.qvel(data).astype(np.float32),
        last_action.astype(np.float32),
        ]
    ).astype(np.float32)


class ScriptedArmSwing:
    def __init__(self, model: mujoco.MjModel, hold_joint_names: list[str], default_hold_pos: np.ndarray):
        self.body_ids = np.array([name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in FOOT_BODY_NAMES])
        self.root_body_id = 1
        self.local_id = {name: i for i, name in enumerate(hold_joint_names)}
        self.default = default_hold_pos.copy()
        self.target = default_hold_pos.copy()

    def reset(self) -> None:
        self.target = self.default.copy()

    def __call__(self, data: mujoco.MjData) -> np.ndarray:
        root_pos = data.xpos[self.root_body_id]
        root_mat = data.xmat[self.root_body_id].reshape(3, 3)
        foot_pos_b = (data.xpos[self.body_ids] - root_pos) @ root_mat
        foot_center_x = 0.5 * (foot_pos_b[0, 0] + foot_pos_b[1, 0])
        left_foot_swing = foot_pos_b[0, 0] - foot_center_x
        right_foot_swing = foot_pos_b[1, 0] - foot_center_x

        target = self.default.copy()
        left_arm_pitch = np.clip(
            SHOULDER_PITCH_GAIN * right_foot_swing,
            -SHOULDER_PITCH_LIMIT,
            SHOULDER_PITCH_LIMIT,
        )
        right_arm_pitch = np.clip(
            SHOULDER_PITCH_GAIN * left_foot_swing,
            -SHOULDER_PITCH_LIMIT,
            SHOULDER_PITCH_LIMIT,
        )
        left_elbow = np.clip(ELBOW_GAIN * abs(right_foot_swing), 0.0, ELBOW_LIMIT)
        right_elbow = np.clip(ELBOW_GAIN * abs(left_foot_swing), 0.0, ELBOW_LIMIT)
        target[self.local_id["left_shoulder_pitch_joint"]] += left_arm_pitch
        target[self.local_id["right_shoulder_pitch_joint"]] += right_arm_pitch
        target[self.local_id["left_elbow_joint"]] += left_elbow
        target[self.local_id["right_elbow_joint"]] += right_elbow
        self.target = ARM_SWING_SMOOTHING * target + (1.0 - ARM_SWING_SMOOTHING) * self.target
        return self.target.copy()


def pd_torque(joints: JointSet, data: mujoco.MjData, target: np.ndarray) -> np.ndarray:
    q = joints.qpos(data)
    dq = joints.qvel(data)
    kp = np.array([regex_value(KP, name) for name in joints.names], dtype=np.float64)
    kd = np.array([regex_value(KD, name) for name in joints.names], dtype=np.float64)
    limit = np.array([regex_value(TAU_LIMIT, name) for name in joints.names], dtype=np.float64)
    return np.clip(kp * (target - q) - kd * dq, -limit, limit)


def init_viewer(args: argparse.Namespace, model: mujoco.MjModel, data: mujoco.MjData):
    if args.no_render:
        return None, None, None, None
    if not args.headless:
        from mujoco import viewer

        handle = viewer.launch_passive(model, data)
        handle.cam.distance = 3.5
        handle.cam.azimuth = 90.0
        handle.cam.elevation = -18.0
        handle.cam.lookat = [0.0, 0.0, 0.75]
        return None, None, None, handle

    import cv2

    model.vis.global_.offwidth = args.viewer_width
    model.vis.global_.offheight = args.viewer_height
    renderer = mujoco.Renderer(model, width=args.viewer_width, height=args.viewer_height)
    camera = mujoco.MjvCamera()
    camera.distance = 3.5
    camera.azimuth = 90.0
    camera.elevation = -18.0
    camera.lookat = [0.0, 0.0, 0.75]
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        1.0 / (TRAIN_DT * TRAIN_DECIMATION),
        (args.viewer_width, args.viewer_height),
    )
    return renderer, camera, writer, None


def render(args: argparse.Namespace, data: mujoco.MjData, renderer, camera, writer, viewer) -> bool:
    if args.no_render:
        return True
    if args.headless:
        renderer.update_scene(data, camera=camera)
        frame = renderer.render()
        writer.write(frame[:, :, ::-1])
        return True
    if not viewer.is_running():
        return False
    viewer.sync()
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lens110 12-DoF lower-body sim2sim.")
    parser.add_argument("--policy", type=Path, default=None, help="TorchScript policy.pt. Defaults to latest leg-pitch-roll export.")
    parser.add_argument("--mjcf", type=Path, default=None, help="Direct-ankle Lens110 MJCF.")
    parser.add_argument("--cmd_vel", type=float, nargs=3, default=[0.4, 0.0, 0.0], help="vx vy wz command.")
    parser.add_argument("--duration", type=float, default=30.0, help="Simulation duration in seconds.")
    parser.add_argument("--stand_warmup", type=float, default=0.5, help="Hold default pose before policy starts.")
    parser.add_argument("--command_ramp", type=float, default=1.0, help="Seconds to ramp command after warmup.")
    parser.add_argument("--root_height", type=float, default=DEFAULT_ROOT_HEIGHT, help="Initial root height.")
    parser.add_argument("--device", default="auto", help="cuda, cpu, or auto.")
    parser.add_argument("--print_every", type=int, default=25, help="Print every N control steps.")
    parser.add_argument("--hold_default", action="store_true", help="Run PD hold without policy actions.")
    parser.add_argument("--headless", action="store_true", help="Render to mp4 using EGL.")
    parser.add_argument("--no_render", action="store_true", help="Run physics only.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/lens110_12dof_sim2sim.mp4")
    parser.add_argument("--viewer_width", type=int, default=1280)
    parser.add_argument("--viewer_height", type=int, default=720)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mjcf_path = resolve_path(args.mjcf) if args.mjcf is not None else first_existing(DEFAULT_MJCF_CANDIDATES)
    policy_path = resolve_path(args.policy) if args.policy is not None else latest_policy(DEFAULT_POLICY_ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    leg_joints = JointSet(model, LEG_JOINT_NAMES)
    all_controlled = JointSet(model, LEG_JOINT_NAMES + HOLD_JOINT_NAMES)
    patch_model(model, all_controlled)
    lower, upper = joint_ranges(model, leg_joints)
    set_default_pose(model, data, all_controlled, args.root_height)

    imu = SensorIMU(model)
    policy = TorchScriptPolicy(policy_path, device)

    default_leg_pos = np.array([DEFAULT_JOINT_POS[name] for name in LEG_JOINT_NAMES], dtype=np.float64)
    default_hold_pos = np.array([DEFAULT_JOINT_POS[name] for name in HOLD_JOINT_NAMES], dtype=np.float64)
    command = np.asarray(args.cmd_vel, dtype=np.float32)
    last_action = np.zeros(len(LEG_JOINT_NAMES), dtype=np.float32)
    arm_swing = ScriptedArmSwing(model, HOLD_JOINT_NAMES, default_hold_pos)

    renderer, camera, writer, viewer = init_viewer(args, model, data)

    print(f"[INFO] policy: {policy_path}")
    print(f"[INFO] MJCF: {mjcf_path}")
    print(f"[INFO] device: {device}")
    print(f"[INFO] controlled policy joints: {len(LEG_JOINT_NAMES)}")
    print(f"[INFO] held upper-body joints: {len(HOLD_JOINT_NAMES)}")
    print(f"[INFO] physics dt={TRAIN_DT:.4f}s, control dt={TRAIN_DT * TRAIN_DECIMATION:.4f}s")
    print(f"[INFO] command: ({command[0]:+.2f}, {command[1]:+.2f}, {command[2]:+.2f})")

    steps = int(args.duration / TRAIN_DT)
    warmup_steps = int(round(args.stand_warmup / TRAIN_DT))
    ramp_steps = int(round(args.command_ramp / TRAIN_DT))
    leg_target = default_leg_pos.copy()
    hold_target = default_hold_pos.copy()
    min_root_z = float("inf")
    max_abs_yaw_rate = 0.0
    start = time.time()

    for step in tqdm(range(steps), desc="Simulating"):
        if step % TRAIN_DECIMATION == 0:
            enabled = (not args.hold_default) and step >= warmup_steps
            if enabled and ramp_steps > 0:
                alpha = float(np.clip((step - warmup_steps) / ramp_steps, 0.0, 1.0))
            else:
                alpha = 1.0 if enabled else 0.0
            obs_command = command * alpha
            obs = build_observation(data, leg_joints, imu, default_leg_pos, last_action, obs_command)

            if enabled:
                action = policy(obs).reshape(-1)
                if action.shape[0] != len(LEG_JOINT_NAMES):
                    raise ValueError(f"Policy output dim {action.shape[0]} != 12")
                last_action = action.astype(np.float32)
                leg_target = default_leg_pos + ACTION_SCALE * last_action
                np.clip(leg_target, lower, upper, out=leg_target)
            else:
                last_action.fill(0.0)
                leg_target = default_leg_pos.copy()
                arm_swing.reset()
            hold_target = arm_swing(data) if enabled else default_hold_pos.copy()

            policy_step = step // TRAIN_DECIMATION
            if policy_step % max(args.print_every, 1) == 0:
                lin_vel = imu.linear_velocity_body(data)
                gyro = imu.gyro_body(data)
                print(
                    f"t={data.time:5.2f}s "
                    f"cmd=({obs_command[0]:+.2f},{obs_command[1]:+.2f},{obs_command[2]:+.2f}) "
                    f"vel=({lin_vel[0]:+.2f},{lin_vel[1]:+.2f},{gyro[2]:+.2f}) "
                    f"z={data.qpos[2]:+.3f} "
                    f"act_abs_max={np.max(np.abs(last_action)):.3f}"
                )

            if not render(args, data, renderer, camera, writer, viewer):
                break

        full_target = np.concatenate([leg_target, hold_target])
        data.ctrl[all_controlled.actuator_ids] = full_target
        mujoco.mj_step(model, data)
        min_root_z = min(min_root_z, float(data.qpos[2]))
        max_abs_yaw_rate = max(max_abs_yaw_rate, abs(float(imu.gyro_body(data)[2])))

        if not args.headless and not args.no_render:
            target_time = (step + 1) * TRAIN_DT
            elapsed = time.time() - start
            if elapsed < target_time:
                time.sleep(target_time - elapsed)

    print(f"[SUMMARY] min_root_z={min_root_z:.3f}, max_abs_yaw_rate={max_abs_yaw_rate:.3f}")
    if writer is not None:
        writer.release()
        print(f"[INFO] saved video: {args.output}")
    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
