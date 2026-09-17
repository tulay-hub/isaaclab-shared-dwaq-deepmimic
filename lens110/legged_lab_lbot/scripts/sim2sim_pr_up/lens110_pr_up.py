# SPDX-License-Identifier: BSD-3-Clause
"""Lens110 pitch/roll policy deployed on upper/lower ankle MJCF.

Policy observation/action semantics stay pitch/roll.  Physical ankle commands
are converted to upper/lower motor targets with side-specific pkl models.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

if "--headless" in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

import joblib
import mujoco
import numpy as np
from tqdm import tqdm


DEFAULT_CONFIG = Path(__file__).with_name("pr_up.json")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "outputs" / "lens110_pr_up.mp4"
UPPER_LOWER = (
    "left_ankle_upper_joint",
    "left_ankle_lower_joint",
    "right_ankle_upper_joint",
    "right_ankle_lower_joint",
)
ANKLE_PR = (
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
POLICY_ANKLE_PR_LIMITS = {
    "left_ankle_pitch_joint": (-1.50, 0.90),
    "left_ankle_roll_joint": (-0.60, 0.60),
    "right_ankle_pitch_joint": (-1.50, 0.90),
    "right_ankle_roll_joint": (-0.60, 0.60),
}
TRAINING_TO_MUJOCO_SIGN = {
    # lens110.xml right elbow axis is flipped relative to the URDF/training semantics.
    "right_elbow_joint": -1.0,
}
GROUND_GEOM_NAMES = {"ground", "floor", "plane"}
TRAINING_FOOT_BOXES = {
    "left_ankle_roll_collision": ((0.019, 0.009, -0.030), (0.0975, 0.0475, 0.010)),
    "right_ankle_roll_collision": ((0.019, -0.009, -0.030), (0.0975, 0.0475, 0.010)),
    "left_ankle_pitch_collision": ((0.019, 0.009, -0.030), (0.0975, 0.0475, 0.010)),
    "right_ankle_pitch_collision": ((0.019, -0.009, -0.030), (0.0975, 0.0475, 0.010)),
}


def repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in (current.parent, *current.parents):
        if (parent / "source" / "legged_lab").is_dir():
            return parent
    return Path.cwd().resolve()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, Path(__file__).resolve().parent / path, repo_root() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (repo_root() / path).resolve()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def name_to_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise ValueError(f"Missing {obj_type.name}: {name}")
    return obj_id


def make_mapping(source_names: list[str], target_names: list[str]) -> np.ndarray:
    target_index = {name: i for i, name in enumerate(target_names)}
    return np.array([target_index[name] for name in source_names], dtype=np.int32)


def quat_to_rotmat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / max(np.linalg.norm(quat), 1.0e-12)
    w, x, y, z = quat
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class OnnxPolicy:
    def __init__(self, path: Path):
        try:
            import onnxruntime as ort

            self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.backend = "onnxruntime"
            self._run = self._run_onnxruntime
        except ImportError:
            import onnx
            from onnx.reference import ReferenceEvaluator

            model = onnx.load(str(path))
            onnx.checker.check_model(model)
            self.session = ReferenceEvaluator(model)
            self.input_name = model.graph.input[0].name
            self.output_name = model.graph.output[0].name
            self.backend = "onnx-reference"
            self._run = self._run_reference

    def _run_onnxruntime(self, obs: np.ndarray) -> np.ndarray:
        return self.session.run([self.output_name], {self.input_name: obs})[0]

    def _run_reference(self, obs: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: obs})[0]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self._run(obs.astype(np.float32, copy=False))[0]


class Joints:
    def __init__(self, model: mujoco.MjModel, joint_names: list[str], require_actuators: bool = True):
        self.names = joint_names
        self.joint_ids = np.array([name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names], dtype=np.int32)
        self.qpos_ids = model.jnt_qposadr[self.joint_ids].astype(np.int32)
        self.qvel_ids = model.jnt_dofadr[self.joint_ids].astype(np.int32)
        self.actuator_ids = self._find_actuators(model) if require_actuators else None

    def _find_actuators(self, model: mujoco.MjModel) -> np.ndarray:
        actuator_ids = np.full(len(self.joint_ids), -1, dtype=np.int32)
        for actuator_id in range(model.nu):
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            matches = np.where(self.joint_ids == joint_id)[0]
            if len(matches):
                actuator_ids[matches[0]] = actuator_id
        if np.any(actuator_ids < 0):
            missing = [name for name, actuator_id in zip(self.names, actuator_ids) if actuator_id < 0]
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

    @staticmethod
    def _sensor(model: mujoco.MjModel, name: str, dim: int) -> tuple[int, int]:
        sensor_id = name_to_id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        actual_dim = int(model.sensor_dim[sensor_id])
        if actual_dim < dim:
            raise ValueError(f"Sensor {name} dim={actual_dim}, expected at least {dim}")
        return int(model.sensor_adr[sensor_id]), dim

    def quat(self, data: mujoco.MjData) -> np.ndarray:
        adr, dim = self.orientation
        return data.sensordata[adr : adr + dim].copy()

    def gyro_body(self, data: mujoco.MjData) -> np.ndarray:
        adr, dim = self.gyro
        return data.sensordata[adr : adr + dim].copy()


class PrToUpperLower:
    def __init__(self, left_path: Path, right_path: Path, default_pr: np.ndarray):
        self.left = joblib.load(left_path)
        self.right = joblib.load(right_path)
        self.default_ul = self(default_pr)

    def __call__(self, pr: np.ndarray) -> np.ndarray:
        pr = np.asarray(pr, dtype=np.float64)
        left_ul = np.asarray(self.left.predict(pr[[0, 1]].reshape(1, 2))[0], dtype=np.float64)
        right_ul = np.asarray(self.right.predict(pr[[2, 3]].reshape(1, 2))[0], dtype=np.float64)
        return np.array([left_ul[0], left_ul[1], right_ul[0], right_ul[1]], dtype=np.float64)

    def target_with_default_offset(self, target_pr: np.ndarray, default_control_ul: np.ndarray) -> np.ndarray:
        return default_control_ul + (self(target_pr) - self.default_ul)


def patch_contacts_and_actuators(model: mujoco.MjModel, joints: Joints, cfg: dict) -> None:
    physics = cfg.get("physics", {})
    robot = cfg["robot"]
    if physics.get("use_training_foot_collision_boxes", True):
        for name, (pos, size) in TRAINING_FOOT_BOXES.items():
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id >= 0:
                model.geom_pos[geom_id] = np.asarray(pos, dtype=np.float64)
                model.geom_size[geom_id] = np.asarray(size, dtype=np.float64)
                model.geom_type[geom_id] = mujoco.mjtGeom.mjGEOM_BOX

    foot_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in robot["foot_contact_geoms"]
    }
    foot_ids.discard(-1)
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        active = name in GROUND_GEOM_NAMES or geom_id in foot_ids
        model.geom_contype[geom_id] = int(active)
        model.geom_conaffinity[geom_id] = int(active)

    static_friction = physics.get("static_friction")
    dynamic_friction = physics.get("dynamic_friction", static_friction)
    if static_friction is not None:
        for ground in sorted(GROUND_GEOM_NAMES):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, ground)
            if gid >= 0:
                model.geom_condim[gid] = 4
                model.geom_margin[gid] = 0.001
                model.geom_friction[gid] = [float(dynamic_friction), 0.2, 0.2]
                break

    armature = np.asarray(robot["armature"], dtype=np.float64)
    model.dof_damping[:] = 0.0
    model.dof_armature[joints.qvel_ids] = armature
    tau_limit = np.asarray(robot["tau_limit"], dtype=np.float64)
    for i, actuator_id in enumerate(joints.actuator_ids):
        effort = float(tau_limit[i])
        model.actuator_gaintype[actuator_id] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_biastype[actuator_id] = mujoco.mjtBias.mjBIAS_NONE
        model.actuator_gainprm[actuator_id, :] = 0.0
        model.actuator_gainprm[actuator_id, 0] = 1.0
        model.actuator_biasprm[actuator_id, :] = 0.0
        model.actuator_ctrllimited[actuator_id] = 1
        model.actuator_ctrlrange[actuator_id] = [-effort, effort]
        model.actuator_forcelimited[actuator_id] = 1
        model.actuator_forcerange[actuator_id] = [-effort, effort]


def joint_ranges(model: mujoco.MjModel, joints: Joints) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(len(joints.names), -np.inf, dtype=np.float64)
    upper = np.full(len(joints.names), np.inf, dtype=np.float64)
    for i, joint_id in enumerate(joints.joint_ids):
        if model.jnt_limited[joint_id]:
            lower[i], upper[i] = model.jnt_range[joint_id]
    return lower, upper


def apply_policy_ankle_pr_limits(
    lower: np.ndarray,
    upper: np.ndarray,
    joint_names: list[str],
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    lower = lower.copy()
    upper = upper.copy()
    limits = cfg.get("robot", {}).get("policy_ankle_pr_limits", POLICY_ANKLE_PR_LIMITS)
    for i, name in enumerate(joint_names):
        if name in limits:
            lower[i], upper[i] = limits[name]
    return lower, upper


def training_defaults_to_mujoco(default_pos: np.ndarray, joint_names: list[str]) -> np.ndarray:
    converted = default_pos.astype(np.float64, copy=True)
    for i, name in enumerate(joint_names):
        converted[i] *= TRAINING_TO_MUJOCO_SIGN.get(name, 1.0)
    return converted


def reset_robot(
    data: mujoco.MjData,
    observe_joints: Joints,
    control_joints: Joints,
    cfg: dict,
    default_pr: np.ndarray,
    control_default: np.ndarray,
) -> None:
    default_pos = np.asarray(cfg["robot"]["default_pos"], dtype=np.float64)
    init_cfg = cfg["robot"].get("init_height", 0.68)
    init_height = 0.68 if isinstance(init_cfg, str) and init_cfg.lower() == "auto" else float(init_cfg)
    data.qpos[:3] = [0.0, 0.0, init_height]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[observe_joints.qpos_ids] = default_pr
    data.qpos[control_joints.qpos_ids] = control_default
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(data.model, data)
    settle_base_height(data, cfg)


def geom_box_zmin(data: mujoco.MjData, geom_id: int) -> float:
    model = data.model
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


def settle_base_height(data: mujoco.MjData, cfg: dict) -> None:
    init = cfg["robot"].get("init_height", 0.68)
    if isinstance(init, str) and init.lower() == "auto":
        foot_geoms = cfg["robot"]["foot_contact_geoms"]
        zmins = []
        for name in foot_geoms:
            geom_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id >= 0:
                zmins.append(geom_box_zmin(data, geom_id))
        if zmins:
            clearance = float(cfg["robot"].get("init_foot_clearance", -0.002))
            data.qpos[2] += clearance - min(zmins)
            mujoco.mj_forward(data.model, data)


def build_observation(
    data: mujoco.MjData,
    observe_joints: Joints,
    imu: SensorIMU,
    policy_to_mujoco: np.ndarray,
    policy_action_sign: np.ndarray,
    default_pos: np.ndarray,
    last_action: np.ndarray,
    command: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    quat = imu.quat(data)
    rot = quat_to_rotmat_wxyz(quat)
    omega = imu.gyro_body(data)
    gravity = rot.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
    q_obs = (observe_joints.qpos(data) - default_pos)[policy_to_mujoco] * policy_action_sign
    dq_obs = observe_joints.qvel(data)[policy_to_mujoco] * policy_action_sign
    obs = np.concatenate([omega, gravity, command, q_obs, dq_obs, last_action]).astype(np.float32)
    return obs.reshape(1, -1), gravity


def pd_torque(control_q: np.ndarray, control_dq: np.ndarray, target: np.ndarray, cfg: dict) -> np.ndarray:
    kp = np.asarray(cfg["robot"]["kp"], dtype=np.float64)
    kd = np.asarray(cfg["robot"]["kd"], dtype=np.float64)
    limit = np.asarray(cfg["robot"]["tau_limit"], dtype=np.float64)
    tau = kp * (target - control_q) - kd * control_dq
    return np.clip(tau, -limit, limit)


def target_pr_to_control(
    target_pr_full: np.ndarray,
    cfg: dict,
    control_default: np.ndarray,
    mapper: PrToUpperLower,
    default_pr_full: np.ndarray | None = None,
    ankle_pr_scale: float = 1.0,
    ankle_pr_for_mapping: np.ndarray | None = None,
    control_lower: np.ndarray | None = None,
    control_upper: np.ndarray | None = None,
) -> np.ndarray:
    observe_index = {name: i for i, name in enumerate(cfg["robot"]["mujoco_joint_names"])}
    control_index = {name: i for i, name in enumerate(cfg["robot"]["control_joint_names"])}
    control = control_default.copy()
    for name in cfg["robot"]["control_joint_names"]:
        if name in UPPER_LOWER:
            continue
        if name in observe_index:
            if default_pr_full is None:
                control[control_index[name]] = target_pr_full[observe_index[name]]
            else:
                control[control_index[name]] = (
                    control_default[control_index[name]]
                    + target_pr_full[observe_index[name]]
                    - default_pr_full[observe_index[name]]
                )

    ankle_pr = (
        np.asarray(ankle_pr_for_mapping, dtype=np.float64).copy()
        if ankle_pr_for_mapping is not None
        else np.array([target_pr_full[observe_index[name]] for name in ANKLE_PR], dtype=np.float64)
    )
    if default_pr_full is not None and ankle_pr_scale != 1.0:
        default_ankle_pr = np.array([default_pr_full[observe_index[name]] for name in ANKLE_PR], dtype=np.float64)
        ankle_pr = default_ankle_pr + (ankle_pr - default_ankle_pr) * float(ankle_pr_scale)
    ankle_ul = mapper.target_with_default_offset(
        ankle_pr,
        control_default[[control_index[name] for name in UPPER_LOWER]],
    )
    control[[control_index[name] for name in UPPER_LOWER]] = ankle_ul
    if control_lower is not None and control_upper is not None:
        np.clip(control, control_lower, control_upper, out=control)
    return control


def policy_signs(policy_joint_names: list[str], args: argparse.Namespace) -> np.ndarray:
    flip_names: set[str] = set()
    if args.flip_leg_pitch:
        flip_names.update(
            [
                "left_hip_pitch_joint",
                "right_hip_pitch_joint",
                "left_knee_joint",
                "right_knee_joint",
                "left_ankle_pitch_joint",
                "right_ankle_pitch_joint",
            ]
        )
    if args.flip_hip_pitch:
        flip_names.update(["left_hip_pitch_joint", "right_hip_pitch_joint"])
    if args.flip_knee:
        flip_names.update(["left_knee_joint", "right_knee_joint"])
    if args.flip_ankle_pitch:
        flip_names.update(["left_ankle_pitch_joint", "right_ankle_pitch_joint"])
    if args.flip_roll:
        flip_names.update(
            [
                "left_hip_roll_joint",
                "right_hip_roll_joint",
                "left_ankle_roll_joint",
                "right_ankle_roll_joint",
            ]
        )
    signs = []
    for name in policy_joint_names:
        sign = TRAINING_TO_MUJOCO_SIGN.get(name, 1.0)
        if name in flip_names:
            sign *= -1.0
        signs.append(sign)
    return np.asarray(signs, dtype=np.float64)


def limit_upper_lower_command(
    desired: np.ndarray,
    previous: np.ndarray,
    default: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    safety = cfg.get("ankle_command_safety", {})
    rel_clip = safety.get("rel_clip", 0.45)
    rate_limit = safety.get("rate_limit", 4.0)
    dt = float(cfg["simulation"]["dt"])
    decimation = int(cfg["simulation"]["decimation"])
    limited = desired.copy()
    if rel_clip is not None and rel_clip >= 0.0:
        limited = default + np.clip(limited - default, -float(rel_clip), float(rel_clip))
    if rate_limit is not None and rate_limit >= 0.0:
        max_delta = float(rate_limit) * dt * decimation
        limited = previous + np.clip(limited - previous, -max_delta, max_delta)
    return limited


def compensated_ankle_pr_for_mapping(
    target_pr_full: np.ndarray,
    actual_pr_full: np.ndarray,
    default_pr_full: np.ndarray,
    observe_lower: np.ndarray,
    observe_upper: np.ndarray,
    observe_index: dict[str, int],
    gain: float,
    clip: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    pr_ids = np.array([observe_index[name] for name in ANKLE_PR], dtype=np.int32)
    target_pr = target_pr_full[pr_ids].copy()
    actual_pr = actual_pr_full[pr_ids].copy()
    error = target_pr - actual_pr
    if clip is not None and clip >= 0.0:
        error = np.clip(error, -float(clip), float(clip))
    mapped_pr = target_pr + float(gain) * error
    mapped_pr = np.clip(mapped_pr, observe_lower[pr_ids], observe_upper[pr_ids])
    if np.any(~np.isfinite(mapped_pr)):
        mapped_pr = default_pr_full[pr_ids].copy()
    return mapped_pr, error


def init_render(args: argparse.Namespace, model: mujoco.MjModel, data: mujoco.MjData, cfg: dict):
    if not args.headless:
        from mujoco import viewer

        handle = viewer.launch_passive(model, data)
        handle.cam.distance = 3.5
        handle.cam.azimuth = 90
        handle.cam.elevation = -18
        return None, None, None, handle

    import cv2

    width, height = args.viewer_width, args.viewer_height
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    renderer = mujoco.Renderer(model, width=width, height=height)
    camera = mujoco.MjvCamera()
    camera.distance = 3.5
    camera.azimuth = 90
    camera.elevation = -18
    camera.lookat = [0.0, 0.0, 0.8]
    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        1.0 / cfg["simulation"]["dt"],
        (width, height),
    )
    return renderer, camera, writer, None


def render(args: argparse.Namespace, data: mujoco.MjData, renderer, camera, writer, viewer) -> bool:
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
    parser = argparse.ArgumentParser(description="Lens110 PR policy on upper/lower ankle MJCF.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--policy", default=None)
    parser.add_argument("--pos_pr2ul_left", default=None)
    parser.add_argument("--pos_pr2ul_right", default=None)
    parser.add_argument("--cmd_vel", type=float, nargs=3, default=None)
    parser.add_argument("--sim_duration", type=float, default=None)
    parser.add_argument("--print_every", type=int, default=None)
    parser.add_argument("--stand_warmup", type=float, default=None)
    parser.add_argument("--command_ramp", type=float, default=None)
    parser.add_argument("--ankle_action_scale_mult", type=float, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--viewer_width", type=int, default=1280)
    parser.add_argument("--viewer_height", type=int, default=720)
    parser.add_argument("--hold_default", action="store_true")
    parser.add_argument("--ankle_command_rel_clip", type=float, default=None)
    parser.add_argument("--ankle_command_rate_limit", type=float, default=None)
    parser.add_argument("--ankle_pr_track_gain", type=float, default=None)
    parser.add_argument("--ankle_pr_track_clip", type=float, default=None)
    parser.add_argument("--flip_leg_pitch", action="store_true")
    parser.add_argument("--flip_hip_pitch", action="store_true")
    parser.add_argument("--flip_knee", action="store_true")
    parser.add_argument("--flip_ankle_pitch", action="store_true")
    parser.add_argument("--flip_roll", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = resolve_path(args.config)
    cfg = load_config(cfg_path)
    policy_path = resolve_path(args.policy or cfg["policy"]["onnx"])
    xml_path = resolve_path(cfg["model"]["path"])
    left_model = resolve_path(args.pos_pr2ul_left or cfg["model"]["pos_pr2ul_left"])
    right_model = resolve_path(args.pos_pr2ul_right or cfg["model"]["pos_pr2ul_right"])
    for label, model_path in (("left", left_model), ("right", right_model)):
        if model_path.suffix != ".pkl":
            raise ValueError(f"PR->UL {label} model must be a .pkl file, got: {model_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"PR->UL {label} model not found: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    model.opt.timestep = float(cfg["simulation"]["dt"])
    data = mujoco.MjData(model)
    observe_joints = Joints(model, cfg["robot"]["mujoco_joint_names"], require_actuators=False)
    control_joints = Joints(model, cfg["robot"]["control_joint_names"], require_actuators=True)
    patch_contacts_and_actuators(model, control_joints, cfg)
    observe_lower, observe_upper = joint_ranges(model, observe_joints)
    observe_lower, observe_upper = apply_policy_ankle_pr_limits(
        observe_lower,
        observe_upper,
        cfg["robot"]["mujoco_joint_names"],
        cfg,
    )
    control_lower, control_upper = joint_ranges(model, control_joints)

    default_pr_training = np.asarray(cfg["robot"]["default_pos"], dtype=np.float64)
    stand_pr_training = np.asarray(cfg["robot"].get("stand_default_pos", cfg["robot"]["default_pos"]), dtype=np.float64)
    default_pr = training_defaults_to_mujoco(default_pr_training, cfg["robot"]["mujoco_joint_names"])
    stand_pr = training_defaults_to_mujoco(stand_pr_training, cfg["robot"]["mujoco_joint_names"])
    observe_index = {name: i for i, name in enumerate(cfg["robot"]["mujoco_joint_names"])}
    control_index = {name: i for i, name in enumerate(cfg["robot"]["control_joint_names"])}
    default_ankle_pr = np.array([default_pr[observe_index[name]] for name in ANKLE_PR], dtype=np.float64)
    mapper = PrToUpperLower(left_model, right_model, default_ankle_pr)
    control_seed = np.zeros(len(cfg["robot"]["control_joint_names"]), dtype=np.float64)
    control_seed[[control_index[name] for name in UPPER_LOWER]] = mapper.default_ul
    training_control_default = target_pr_to_control(
        default_pr,
        cfg,
        control_seed,
        mapper,
        control_lower=control_lower,
        control_upper=control_upper,
    )
    control_default = target_pr_to_control(
        stand_pr,
        cfg,
        training_control_default,
        mapper,
        default_pr_full=default_pr,
        ankle_pr_scale=1.0,
        control_lower=control_lower,
        control_upper=control_upper,
    )
    reset_robot(data, observe_joints, control_joints, cfg, stand_pr, control_default)

    imu = SensorIMU(model)
    policy_to_mujoco = make_mapping(cfg["robot"]["policy_joint_names"], cfg["robot"]["mujoco_joint_names"])
    model_to_policy = make_mapping(cfg["robot"]["mujoco_joint_names"], cfg["robot"]["policy_joint_names"])
    policy_action_sign = policy_signs(cfg["robot"]["policy_joint_names"], args)
    action_scale_policy = np.asarray(cfg["robot"]["action_scale"], dtype=np.float64)
    if len(action_scale_policy) != len(cfg["robot"]["policy_joint_names"]):
        raise ValueError("robot.action_scale must be in policy_joint_names order")
    command = np.asarray(args.cmd_vel or cfg["simulation"]["default_cmd_vel"], dtype=np.float32)
    duration = float(args.sim_duration or cfg["simulation"]["duration"])
    print_every = int(args.print_every or cfg["simulation"].get("print_every", 50))
    decimation = int(cfg["simulation"]["decimation"])
    stand_warmup = float(args.stand_warmup if args.stand_warmup is not None else cfg["simulation"].get("stand_warmup", 0.5))
    command_ramp = float(args.command_ramp if args.command_ramp is not None else cfg["simulation"].get("command_ramp", 1.0))
    ankle_action_scale_mult = float(
        args.ankle_action_scale_mult
        if args.ankle_action_scale_mult is not None
        else cfg["robot"].get("pr_to_ul_action_scale_mult", 1.0)
    )
    cfg.setdefault("ankle_command_safety", {})
    if args.ankle_command_rel_clip is not None:
        cfg["ankle_command_safety"]["rel_clip"] = None if args.ankle_command_rel_clip < 0.0 else args.ankle_command_rel_clip
    if args.ankle_command_rate_limit is not None:
        cfg["ankle_command_safety"]["rate_limit"] = (
            None if args.ankle_command_rate_limit < 0.0 else args.ankle_command_rate_limit
        )
    ankle_feedback_cfg = cfg.setdefault("ankle_pr_feedback", {})
    ankle_pr_track_gain = float(
        args.ankle_pr_track_gain
        if args.ankle_pr_track_gain is not None
        else ankle_feedback_cfg.get("gain", 0.0)
    )
    ankle_pr_track_clip = (
        None
        if args.ankle_pr_track_clip is not None and args.ankle_pr_track_clip < 0.0
        else (
            float(args.ankle_pr_track_clip)
            if args.ankle_pr_track_clip is not None
            else ankle_feedback_cfg.get("clip", None)
        )
    )
    if ankle_pr_track_clip is not None:
        ankle_pr_track_clip = float(ankle_pr_track_clip)

    policy = OnnxPolicy(policy_path)
    renderer, camera, writer, viewer = init_render(args, model, data, cfg)

    print(f"[INFO] policy: {policy_path}")
    print(f"[INFO] policy backend: {policy.backend}")
    print(f"[INFO] XML: {xml_path}")
    print("[INFO] policy IO: pitch/roll")
    print("[INFO] plant: upper/lower physical motors")
    print(f"[INFO] pr2ul left: {left_model}")
    print(f"[INFO] pr2ul right: {right_model}")
    print(
        "[INFO] physical stand ankle upper/lower: "
        f"({control_default[control_index['left_ankle_upper_joint']]:+.3f}, "
        f"{control_default[control_index['left_ankle_lower_joint']]:+.3f}, "
        f"{control_default[control_index['right_ankle_upper_joint']]:+.3f}, "
        f"{control_default[control_index['right_ankle_lower_joint']]:+.3f})"
    )
    print("[INFO] obs: gyro=angular-velocity sensor, gravity=orientation sensor quaternion")
    print("[INFO] control: explicit torque PD")
    print("[INFO] default_pos: training semantics converted to MuJoCo joint axes")
    print(
        "[INFO] ankle command safety: "
        f"rel_clip={cfg['ankle_command_safety'].get('rel_clip', 0.45)}, "
        f"rate_limit={cfg['ankle_command_safety'].get('rate_limit', 4.0)} rad/s"
    )
    if np.any(policy_action_sign < 0.0):
        flipped = [name for name, sign in zip(cfg["robot"]["policy_joint_names"], policy_action_sign) if sign < 0.0]
        print(f"[INFO] policy sign flips: {flipped}")
    print(f"[INFO] cmd: ({command[0]:.2f}, {command[1]:.2f}, {command[2]:.2f})")
    print(f"[INFO] startup: stand_warmup={stand_warmup:.2f}s, command_ramp={command_ramp:.2f}s")
    print(f"[INFO] ankle PR target scale multiplier before PR->UL conversion: {ankle_action_scale_mult:.3f}")
    print(f"[INFO] ankle PR mapping feedback: gain={ankle_pr_track_gain:.3f}, clip={ankle_pr_track_clip}")

    last_action = np.zeros(int(cfg["robot"]["num_actions"]), dtype=np.float32)
    target_pr = stand_pr.copy()
    control_target = control_default.copy()
    ul_control_ids = np.array([control_index[name] for name in UPPER_LOWER], dtype=np.int32)
    default_ul_for_safety = control_default[ul_control_ids].copy()
    previous_ul_target = default_ul_for_safety.copy()
    ankle_pr_ids = np.array([observe_index[name] for name in ANKLE_PR], dtype=np.int32)
    ankle_pr_map = stand_pr[ankle_pr_ids].copy()
    ankle_pr_err = np.zeros(4, dtype=np.float64)
    min_root_z = float("inf")
    max_abs_yaw_rate = 0.0
    max_abs_vy = 0.0
    steps = int(duration / model.opt.timestep)
    warmup_steps = int(round(stand_warmup / model.opt.timestep))
    ramp_steps = int(round(command_ramp / model.opt.timestep))
    start = time.time()

    for step in tqdm(range(steps), desc="Simulating"):
        if step % decimation == 0:
            policy_enabled = (not args.hold_default) and step >= warmup_steps
            if policy_enabled and ramp_steps > 0:
                command_alpha = float(np.clip((step - warmup_steps) / ramp_steps, 0.0, 1.0))
            else:
                command_alpha = 0.0 if not policy_enabled else 1.0
            obs_command = command * command_alpha
            obs, _ = build_observation(
                data,
                observe_joints,
                imu,
                policy_to_mujoco,
                policy_action_sign,
                default_pr,
                last_action,
                obs_command,
            )
            if not policy_enabled:
                raw_action = np.zeros_like(last_action)
            else:
                raw_action = policy(obs).astype(np.float32)
            last_action = raw_action
            if not policy_enabled:
                target_pr = stand_pr.copy()
                control_target = control_default.copy()
                ankle_pr_map = stand_pr[ankle_pr_ids].copy()
                ankle_pr_err[:] = 0.0
            else:
                target_delta_policy = raw_action.astype(np.float64) * action_scale_policy * policy_action_sign
                target_pr = default_pr.copy()
                target_pr[policy_to_mujoco] += target_delta_policy
                np.clip(target_pr, observe_lower, observe_upper, out=target_pr)
                actual_pr_full = observe_joints.qpos(data)
                ankle_pr_map, ankle_pr_err = compensated_ankle_pr_for_mapping(
                    target_pr,
                    actual_pr_full,
                    default_pr,
                    observe_lower,
                    observe_upper,
                    observe_index,
                    ankle_pr_track_gain,
                    ankle_pr_track_clip,
                )
                control_target = target_pr_to_control(
                    target_pr,
                    cfg,
                    control_default,
                    mapper,
                    default_pr_full=default_pr,
                    ankle_pr_scale=ankle_action_scale_mult,
                    ankle_pr_for_mapping=ankle_pr_map,
                    control_lower=control_lower,
                    control_upper=control_upper,
                )
            desired_ul = control_target[ul_control_ids]
            limited_ul = limit_upper_lower_command(desired_ul, previous_ul_target, default_ul_for_safety, cfg)
            control_target[ul_control_ids] = np.clip(limited_ul, control_lower[ul_control_ids], control_upper[ul_control_ids])
            previous_ul_target = control_target[ul_control_ids].copy()

            policy_step = step // decimation
            if policy_step % max(1, print_every) == 0:
                quat = imu.quat(data)
                rot = quat_to_rotmat_wxyz(quat)
                lin_vel_b = rot.T @ data.qvel[:3]
                gyro = imu.gyro_body(data)
                ul = control_target[ul_control_ids]
                ul0 = control_default[ul_control_ids]
                pr_tgt = target_pr[ankle_pr_ids]
                pr_act = observe_joints.qpos(data)[ankle_pr_ids]
                print(
                    f"t={data.time:5.2f}s "
                    f"cmd=({obs_command[0]:+.2f},{obs_command[1]:+.2f},{obs_command[2]:+.2f}) "
                    f"vel=({lin_vel_b[0]:+.2f},{lin_vel_b[1]:+.2f},{gyro[2]:+.2f}) "
                    f"z={data.qpos[2]:+.3f} "
                    "ankle_pr_tgt="
                    f"({pr_tgt[0]:+.3f},{pr_tgt[1]:+.3f},{pr_tgt[2]:+.3f},{pr_tgt[3]:+.3f}) "
                    "ankle_pr_act="
                    f"({pr_act[0]:+.3f},{pr_act[1]:+.3f},{pr_act[2]:+.3f},{pr_act[3]:+.3f}) "
                    "ankle_pr_map="
                    f"({ankle_pr_map[0]:+.3f},{ankle_pr_map[1]:+.3f},{ankle_pr_map[2]:+.3f},{ankle_pr_map[3]:+.3f}) "
                    "ankle_pr_err="
                    f"({ankle_pr_err[0]:+.3f},{ankle_pr_err[1]:+.3f},{ankle_pr_err[2]:+.3f},{ankle_pr_err[3]:+.3f}) "
                    "ankle_ul_rel="
                    f"({ul[0]-ul0[0]:+.3f},{ul[1]-ul0[1]:+.3f},{ul[2]-ul0[2]:+.3f},{ul[3]-ul0[3]:+.3f})"
                )

            if not render(args, data, renderer, camera, writer, viewer):
                break

        torque = pd_torque(control_joints.qpos(data), control_joints.qvel(data), control_target, cfg)
        data.ctrl[control_joints.actuator_ids] = torque
        mujoco.mj_step(model, data)
        min_root_z = min(min_root_z, float(data.qpos[2]))
        max_abs_yaw_rate = max(max_abs_yaw_rate, abs(float(imu.gyro_body(data)[2])))
        quat = imu.quat(data)
        max_abs_vy = max(max_abs_vy, abs(float((quat_to_rotmat_wxyz(quat).T @ data.qvel[:3])[1])))
        target_time = (step + 1) * model.opt.timestep
        elapsed = time.time() - start
        if not args.headless and elapsed < target_time:
            time.sleep(target_time - elapsed)

    print(
        "[SUMMARY] "
        f"min_root_z={min_root_z:.3f}, "
        f"max_abs_yaw_rate={max_abs_yaw_rate:.3f}, "
        f"max_abs_vy={max_abs_vy:.3f}"
    )
    if args.headless:
        writer.release()
        print(f"[INFO] saved video: {args.output}")
    elif viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
