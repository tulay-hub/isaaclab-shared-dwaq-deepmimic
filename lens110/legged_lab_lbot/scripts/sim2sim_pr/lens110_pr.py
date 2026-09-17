# SPDX-License-Identifier: BSD-3-Clause
"""MuJoCo sim2sim runner for the Lens110 AMP policy.

This is a compact version of the previous script.  The behavior is kept the
same: keyboard commands, warmup standing, IsaacLab policy joint order, MuJoCo
training-style MuJoCo position-PD mode, optional raw torque/position modes, headless video, and debug
plots are all still supported.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass

if "--headless" in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

import matplotlib.pyplot as plt
import mujoco
try:
    import cv2
except ImportError:  # Video output is optional for headless replay.
    cv2 = None
try:
    import mujoco_viewer
except ImportError:  # Window rendering is optional for headless replay.
    mujoco_viewer = None
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

try:
    from pynput import keyboard
except ImportError:
    keyboard = None


DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "pr.json")
DEFAULT_FRAME_STACK = 1


def repo_root() -> str:
    current = os.path.abspath(os.path.dirname(__file__))
    while True:
        if os.path.isdir(os.path.join(current, "source", "legged_lab")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        current = parent


def resolve_path(path: str | None, *base_dirs: str) -> str | None:
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    candidates = [os.path.abspath(path)]
    candidates.extend(os.path.join(base_dir, path) for base_dir in base_dirs)
    candidates.append(os.path.join(repo_root(), path))
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(os.path.join(repo_root(), path))


@dataclass
class RunParams:
    params_dir: str
    default_pos: np.ndarray | None = None
    action_scale: np.ndarray | None = None
    init_height: float | None = None
    action_clip: float | None = None
    static_friction: float | None = None
    dynamic_friction: float | None = None
    restitution: float | None = None


@dataclass
class Sim2SimConfig:
    torchscript_policy: str
    onnx_policy: str
    policy_backend: str
    model_path: str
    source_params_dir: str | None
    dt: float
    decimation: int
    num_actions: int
    num_obs: int
    frame_stack: int
    mujoco_joint_names: list[str]
    policy_joint_names: list[str]
    urdf_joint_names: list[str]
    default_pos: np.ndarray
    action_scale: np.ndarray
    init_height: float | None
    fallback_init_height: float
    action_clip: float | None
    kp: np.ndarray
    kd: np.ndarray
    tau_limit: np.ndarray
    armature: np.ndarray
    foot_contact_geoms: list[str]
    static_friction: float | None
    dynamic_friction: float | None
    restitution: float | None


def _as_float_array(config: dict, key: str, expected_len: int) -> np.ndarray:
    values = np.array(config[key], dtype=np.float64)
    if values.shape != (expected_len,):
        raise ValueError(f"Config key '{key}' must contain {expected_len} values, got shape {values.shape}")
    return values


def load_sim2sim_config(path: str) -> Sim2SimConfig:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    config_dir = os.path.dirname(os.path.abspath(path))
    robot = raw["robot"]
    num_actions = int(robot["num_actions"])
    mujoco_joint_names = list(robot["mujoco_joint_names"])
    policy_joint_names = list(robot["policy_joint_names"])
    urdf_joint_names = list(robot.get("urdf_joint_names", mujoco_joint_names))
    if len(mujoco_joint_names) != num_actions:
        raise ValueError("mujoco_joint_names length must match robot.num_actions")
    if len(policy_joint_names) != num_actions:
        raise ValueError("policy_joint_names length must match robot.num_actions")
    if len(urdf_joint_names) != num_actions:
        raise ValueError("urdf_joint_names length must match robot.num_actions")

    simulation = raw["simulation"]
    policy = raw["policy"]
    model = raw["model"]
    physics = raw.get("physics", {})
    source = raw.get("source", {})
    frame_stack = int(robot.get("frame_stack", DEFAULT_FRAME_STACK))
    return Sim2SimConfig(
        torchscript_policy=resolve_path(policy["torchscript"], config_dir),
        onnx_policy=resolve_path(policy["onnx"], config_dir),
        policy_backend=policy.get("backend", "torchscript"),
        model_path=resolve_path(model["path"], config_dir),
        source_params_dir=resolve_path(source.get("params_dir"), config_dir),
        dt=float(simulation["dt"]),
        decimation=int(simulation["decimation"]),
        num_actions=num_actions,
        num_obs=9 + 3 * num_actions,
        frame_stack=frame_stack,
        mujoco_joint_names=mujoco_joint_names,
        policy_joint_names=policy_joint_names,
        urdf_joint_names=urdf_joint_names,
        default_pos=_as_float_array(robot, "default_pos", num_actions),
        action_scale=_as_float_array(robot, "action_scale", num_actions),
        init_height=None if robot.get("init_height") is None else float(robot["init_height"]),
        fallback_init_height=float(robot.get("fallback_init_height", 0.68)),
        action_clip=None if robot.get("action_clip") is None else float(robot["action_clip"]),
        kp=_as_float_array(robot, "kp", num_actions),
        kd=_as_float_array(robot, "kd", num_actions),
        tau_limit=_as_float_array(robot, "tau_limit", num_actions),
        armature=_as_float_array(robot, "armature", num_actions),
        foot_contact_geoms=list(robot["foot_contact_geoms"]),
        static_friction=None if physics.get("static_friction") is None else float(physics["static_friction"]),
        dynamic_friction=None if physics.get("dynamic_friction") is None else float(physics["dynamic_friction"]),
        restitution=None if physics.get("restitution") is None else float(physics["restitution"]),
    )


class TorchScriptPolicy:
    def __init__(self, path: str):
        self.policy = torch.jit.load(path, map_location="cpu")
        self.policy.eval()

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return self.policy(torch.from_numpy(obs.astype(np.float32, copy=False)))[0].detach().cpu().numpy()


class OnnxPolicy:
    def __init__(self, path: str):
        try:
            import onnxruntime as ort

            self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.backend = "onnxruntime"
            self._run = self._run_onnxruntime
        except ImportError:
            import onnx
            from onnx.reference import ReferenceEvaluator

            model = onnx.load(path)
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


class CommandState:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0
    vx_increment = 0.1
    vy_increment = 0.1
    dyaw_increment = 0.1
    min_vx = -0.8
    max_vx = 2.0
    min_vy = -0.8
    max_vy = 0.8
    min_dyaw = -1.5
    max_dyaw = 1.5
    camera_follow = True
    reset_requested = False

    @classmethod
    def set_initial(cls, values: tuple[float, float, float]) -> None:
        cls.vx, cls.vy, cls.dyaw = values

    @classmethod
    def update_vx(cls, delta: float) -> None:
        cls.vx = float(np.clip(cls.vx + delta, cls.min_vx, cls.max_vx))
        cls.print()

    @classmethod
    def update_vy(cls, delta: float) -> None:
        cls.vy = float(np.clip(cls.vy + delta, cls.min_vy, cls.max_vy))
        cls.print()

    @classmethod
    def update_yaw(cls, delta: float) -> None:
        cls.dyaw = float(np.clip(cls.dyaw + delta, cls.min_dyaw, cls.max_dyaw))
        cls.print()

    @classmethod
    def reset(cls) -> None:
        cls.vx = cls.vy = cls.dyaw = 0.0
        print(f"速度已重置: vx: {cls.vx:.2f}, vy: {cls.vy:.2f}, dyaw: {cls.dyaw:.2f}")

    @classmethod
    def print(cls) -> None:
        print(f"vx: {cls.vx:.2f}, vy: {cls.vy:.2f}, dyaw: {cls.dyaw:.2f}")


def on_press(key) -> None:
    try:
        char = key.char.lower() if hasattr(key, "char") and key.char is not None else None
        if char == "8":
            CommandState.update_vx(CommandState.vx_increment)
        elif char == "2":
            CommandState.update_vx(-CommandState.vx_increment)
        elif char == "4":
            CommandState.update_vy(CommandState.vy_increment)
        elif char == "6":
            CommandState.update_vy(-CommandState.vy_increment)
        elif char == "7":
            CommandState.update_yaw(CommandState.dyaw_increment)
        elif char == "9":
            CommandState.update_yaw(-CommandState.dyaw_increment)
        elif char == "f":
            CommandState.camera_follow = not CommandState.camera_follow
            print(f"Camera follow: {CommandState.camera_follow}")
        elif char == "0":
            CommandState.reset_requested = True
            print("Reset requested (0 key pressed)")
    except AttributeError:
        if key == keyboard.Key.up:
            CommandState.update_vx(CommandState.vx_increment)
        elif key == keyboard.Key.down:
            CommandState.update_vx(-CommandState.vx_increment)
        elif key == keyboard.Key.left:
            CommandState.update_yaw(CommandState.dyaw_increment)
        elif key == keyboard.Key.right:
            CommandState.update_yaw(-CommandState.dyaw_increment)


def start_keyboard_listener():
    if keyboard is None:
        print("键盘监听不可用，跳过键盘控制。")
        return None
    listener = keyboard.Listener(on_press=on_press, on_release=lambda _: None)
    listener.start()
    return listener


def make_index_mapping(source_names: list[str], target_names: list[str]) -> list[int]:
    target_index = {name: i for i, name in enumerate(target_names)}
    return [target_index[name] for name in source_names]


def name_to_id(model: mujoco.MjModel, obj_type, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise ValueError(f"Missing {obj_type.name}: {name}")
    return obj_id


def build_joint_data(model: mujoco.MjModel, joint_names: list[str]):
    joint_ids = np.array([name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names], dtype=np.int32)
    qpos_ids = model.jnt_qposadr[joint_ids].astype(np.int32)
    qvel_ids = model.jnt_dofadr[joint_ids].astype(np.int32)

    actuator_ids = np.full(len(joint_ids), -1, dtype=np.int32)
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        matches = np.flatnonzero(joint_ids == joint_id)
        if len(matches):
            actuator_ids[matches[0]] = actuator_id
    if np.any(actuator_ids < 0):
        missing = [name for name, actuator_id in zip(joint_names, actuator_ids) if actuator_id < 0]
        raise ValueError(f"Missing actuators for joints: {missing}")

    target_min = np.full(len(joint_ids), -np.inf, dtype=np.float64)
    target_max = np.full(len(joint_ids), np.inf, dtype=np.float64)
    for i, joint_id in enumerate(joint_ids):
        if model.jnt_limited[joint_id]:
            target_min[i], target_max[i] = model.jnt_range[joint_id]

    return joint_ids, qpos_ids, qvel_ids, actuator_ids, target_min, target_max


def policy_signs(policy_joint_names: list[str], args: argparse.Namespace, num_actions: int) -> np.ndarray:
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

    signs = np.ones(num_actions, dtype=np.float64)
    for i, name in enumerate(policy_joint_names):
        if name in flip_names:
            signs[i] = -1.0
    return signs


def _parse_scalar(value: str):
    value = value.strip()
    if value == "null":
        return None
    try:
        return float(value)
    except ValueError:
        return value


def _section_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _extract_mapping(lines: list[str], start_idx: int, indent: int) -> tuple[dict[str, object], int]:
    values: dict[str, object] = {}
    idx = start_idx
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            idx += 1
            continue
        current_indent = _section_indent(line)
        if current_indent <= indent:
            break
        if current_indent == indent + 2 and ":" in stripped and not stripped.startswith("-"):
            key, value = stripped.split(":", 1)
            values[key.strip()] = _parse_scalar(value)
        idx += 1
    return values, idx


def _extract_tuple(lines: list[str], start_idx: int, indent: int) -> tuple[list[float], int]:
    values: list[float] = []
    idx = start_idx
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        if not stripped:
            idx += 1
            continue
        current_indent = _section_indent(line)
        if current_indent <= indent:
            break
        if stripped.startswith("-"):
            values.append(float(stripped[1:].strip()))
        idx += 1
    return values, idx


def _extract_physics_material(lines: list[str]) -> dict[str, float | None]:
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        indent = _section_indent(line)
        if indent == 2 and stripped == "physics_material:":
            candidate, next_idx = _extract_mapping(lines, idx + 1, indent)
            if "static_friction" in candidate and "dynamic_friction" in candidate:
                return {
                    "static_friction": None
                    if candidate.get("static_friction") is None
                    else float(candidate["static_friction"]),
                    "dynamic_friction": None
                    if candidate.get("dynamic_friction") is None
                    else float(candidate["dynamic_friction"]),
                    "restitution": None if candidate.get("restitution") is None else float(candidate["restitution"]),
                }
            idx = next_idx
            continue
        idx += 1
    return {}


def _extract_env_params(
    env_yaml_path: str, cfg: Sim2SimConfig, policy_joint_names: list[str]
) -> tuple[np.ndarray | None, np.ndarray | None, float | None, dict[str, float | None]]:
    with open(env_yaml_path, encoding="utf-8") as f:
        lines = f.readlines()

    joint_pos: dict[str, object] | None = None
    scale_map: dict[str, object] | None = None
    root_pos: list[float] | None = None
    in_scene_robot = False
    in_actions_joint_pos = False

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        indent = _section_indent(line)

        if indent == 2 and stripped == "robot:":
            in_scene_robot = True
            in_actions_joint_pos = False
            idx += 1
            continue
        if indent == 0 and stripped == "actions:":
            in_scene_robot = False
            idx += 1
            continue
        if indent == 2 and stripped == "joint_pos:" and not in_scene_robot:
            in_actions_joint_pos = True
            idx += 1
            continue

        if in_scene_robot and indent == 6 and stripped.startswith("pos:"):
            root_pos, idx = _extract_tuple(lines, idx + 1, indent)
            continue
        if in_scene_robot and indent == 6 and stripped == "joint_pos:":
            joint_pos, idx = _extract_mapping(lines, idx + 1, indent)
            continue
        if in_actions_joint_pos and indent == 4 and stripped == "scale:":
            scale_map, idx = _extract_mapping(lines, idx + 1, indent)
            continue
        if indent <= 2 and in_actions_joint_pos and stripped and not stripped.startswith("joint_pos:"):
            in_actions_joint_pos = False

        idx += 1

    default_pos = None
    if joint_pos:
        default_pos = np.array([float(joint_pos[name]) for name in cfg.mujoco_joint_names], dtype=np.float64)

    action_scale = None
    if scale_map:
        action_scale = np.array([_match_scale_from_map(name, scale_map) for name in policy_joint_names], dtype=np.float64)

    init_height = float(root_pos[2]) if root_pos and len(root_pos) >= 3 else None
    return default_pos, action_scale, init_height, _extract_physics_material(lines)


def _match_scale_from_map(joint_name: str, scale_map: dict[str, object]) -> float:
    for pattern, value in scale_map.items():
        if re.fullmatch(pattern, joint_name):
            return float(value)
    raise ValueError(f"No action scale in params matches joint: {joint_name}")


def _extract_agent_clip(agent_yaml_path: str) -> float | None:
    with open(agent_yaml_path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("clip_actions:"):
                _, value = stripped.split(":", 1)
                parsed = _parse_scalar(value)
                return None if parsed is None else float(parsed)
    return None


def load_run_params(params_dir: str | None, cfg: Sim2SimConfig, policy_joint_names: list[str]) -> RunParams | None:
    if not params_dir:
        return None
    env_yaml_path = os.path.join(params_dir, "env.yaml")
    agent_yaml_path = os.path.join(params_dir, "agent.yaml")
    if not os.path.isfile(env_yaml_path) or not os.path.isfile(agent_yaml_path):
        raise FileNotFoundError(f"Missing env.yaml or agent.yaml under params dir: {params_dir}")

    default_pos, action_scale, init_height, physics_material = _extract_env_params(env_yaml_path, cfg, policy_joint_names)
    action_clip = _extract_agent_clip(agent_yaml_path)
    return RunParams(
        params_dir=params_dir,
        default_pos=default_pos,
        action_scale=action_scale,
        init_height=init_height,
        action_clip=action_clip,
        static_friction=physics_material.get("static_friction"),
        dynamic_friction=physics_material.get("dynamic_friction"),
        restitution=physics_material.get("restitution"),
    )


def has_run_params(params_dir: str | None) -> bool:
    if not params_dir:
        return False
    return os.path.isfile(os.path.join(params_dir, "env.yaml")) and os.path.isfile(os.path.join(params_dir, "agent.yaml"))


def select_policy_joint_names(args: argparse.Namespace, cfg: Sim2SimConfig) -> list[str]:
    if args.mujoco_joint_order:
        return cfg.mujoco_joint_names
    if args.isaac_urdf_joint_order:
        return cfg.urdf_joint_names
    return cfg.policy_joint_names


def geom_min_z(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float:
    if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
        sx, sy, sz = model.geom_size[geom_id]
        corners = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)], dtype=np.float64)
        points = data.geom_xpos[geom_id] + corners @ data.geom_xmat[geom_id].reshape(3, 3).T
        return float(points[:, 2].min())
    return float(data.geom_xpos[geom_id, 2] - model.geom_rbound[geom_id])


def grounded_base_height(model: mujoco.MjModel, data: mujoco.MjData, cfg: "RobotCfg", clearance: float = 0.002) -> float:
    saved_qpos, saved_qvel, saved_ctrl = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()
    data.qpos[:3] = cfg.base_pos
    data.qpos[3:7] = cfg.base_quat
    data.qpos[cfg.qpos_ids] = cfg.default_pos
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    min_z = []
    for name in cfg.foot_contact_geoms:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id >= 0:
            min_z.append(geom_min_z(model, data, geom_id))

    data.qpos[:], data.qvel[:], data.ctrl[:] = saved_qpos, saved_qvel, saved_ctrl
    mujoco.mj_forward(model, data)
    return float(cfg.base_pos[2] if not min_z else cfg.base_pos[2] - min(min_z) + clearance)


def get_obs(data: mujoco.MjData, use_root_ang_vel: bool, use_sensor_orientation: bool):
    q = data.qpos.astype(np.float64)
    dq = data.qvel.astype(np.float64)
    quat_wxyz = data.sensor("orientation").data.astype(np.float64) if use_sensor_orientation else q[3:7]
    rot = R.from_quat(quat_wxyz[[1, 2, 3, 0]])
    lin_vel_b = rot.apply(data.qvel[:3], inverse=True).astype(np.float64)
    ang_vel_b = rot.apply(data.qvel[3:6], inverse=True).astype(np.float64) if use_root_ang_vel else data.sensor("angular-velocity").data.astype(np.float64)
    gravity_b = rot.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.float64)
    return q, dq, lin_vel_b, ang_vel_b, gravity_b


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


@dataclass
class SimCfg:
    model_path: str
    duration: float
    viewer_width: int
    viewer_height: int
    dt: float
    decimation: int
    num_actions: int
    num_obs: int
    frame_stack: int


@dataclass
class RobotCfg:
    policy_joint_names: list[str]
    policy_to_mujoco: list[int]
    policy_action_sign: np.ndarray
    action_scale: np.ndarray
    base_pos: np.ndarray
    base_quat: np.ndarray
    control_mode: str
    auto_base_height: bool
    use_mujoco_passive_damping: bool
    action_clip: float | None
    debug_actions: bool
    debug_contacts: bool
    debug_action_frames: int
    print_every: int
    base_ang_vel_from_qvel: bool
    use_sensor_orientation: bool
    stand_warmup: float
    command_ramp: float
    start_policy_immediately: bool
    policy_command_threshold: float
    leg_kp_scale: float = 1.0
    leg_kd_scale: float = 1.0
    ankle_kp_scale: float = 1.0
    ankle_kd_scale: float = 1.0
    static_friction: float | None = None
    dynamic_friction: float | None = None
    restitution: float | None = None
    mujoco_spin_friction: float | None = None
    mujoco_roll_friction: float | None = None
    patch_contacts: bool = True
    clip_target_to_joint_limits: bool = True
    mujoco_joint_names: list[str] | None = None
    foot_contact_geoms: list[str] | None = None
    default_pos: np.ndarray | None = None
    kp: np.ndarray | None = None
    kd: np.ndarray | None = None
    tau_limit: np.ndarray | None = None
    armature: np.ndarray | None = None
    qpos_ids: np.ndarray | None = None
    qvel_ids: np.ndarray | None = None
    actuator_ids: np.ndarray | None = None
    target_min: np.ndarray | None = None
    target_max: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.mujoco_joint_names is None:
            raise ValueError("mujoco_joint_names must be provided by config")
        if self.foot_contact_geoms is None:
            self.foot_contact_geoms = []
        if self.default_pos is None:
            raise ValueError("default_pos must be provided by config")
        self.kp = self.kp.copy()
        self.kd = self.kd.copy()
        self.kp[:12] *= self.leg_kp_scale
        self.kd[:12] *= self.leg_kd_scale
        ankle_ids = [4, 5, 10, 11]
        self.kp[ankle_ids] *= self.ankle_kp_scale
        self.kd[ankle_ids] *= self.ankle_kd_scale
        self.tau_limit = self.tau_limit.copy()
        self.armature = self.armature.copy()


@dataclass
class RunCfg:
    sim: SimCfg
    robot: RobotCfg


def make_cfg(args: argparse.Namespace, base_cfg: Sim2SimConfig) -> RunCfg:
    policy_joint_names = select_policy_joint_names(args, base_cfg)
    run_params = load_run_params(args.params_dir, base_cfg, policy_joint_names) if args.params_dir else None
    action_clip = args.action_clip if args.action_clip is not None else (
        run_params.action_clip if run_params else base_cfg.action_clip
    )
    init_height = (
        run_params.init_height
        if run_params and run_params.init_height is not None
        else args.init_height
    )
    if run_params:
        print(f"[INFO] Loaded sim2sim params from: {run_params.params_dir}")
        if run_params.action_scale is not None:
            print(f"[INFO] action_scale from params: {np.array2string(run_params.action_scale, precision=3)}")
        if run_params.default_pos is not None:
            print("[INFO] default joint positions loaded from params/env.yaml")
        if run_params.init_height is not None:
            print(f"[INFO] init height from params: {run_params.init_height:.3f}")
        print(f"[INFO] action_clip from params/CLI: {action_clip}")
        if run_params.static_friction is not None or run_params.dynamic_friction is not None:
            print(
                "[INFO] physics material from params: "
                f"static_friction={run_params.static_friction}, "
                f"dynamic_friction={run_params.dynamic_friction}, "
                f"restitution={run_params.restitution}"
            )
    return RunCfg(
        sim=SimCfg(
            model_path=args.model_path,
            duration=args.sim_duration,
            viewer_width=args.viewer_width,
            viewer_height=args.viewer_height,
            dt=base_cfg.dt,
            decimation=base_cfg.decimation,
            num_actions=base_cfg.num_actions,
            num_obs=base_cfg.num_obs,
            frame_stack=base_cfg.frame_stack,
        ),
        robot=RobotCfg(
            policy_joint_names=policy_joint_names,
            policy_to_mujoco=make_index_mapping(policy_joint_names, base_cfg.mujoco_joint_names),
            policy_action_sign=policy_signs(policy_joint_names, args, base_cfg.num_actions),
            action_scale=run_params.action_scale if run_params and run_params.action_scale is not None else base_cfg.action_scale.copy(),
            base_pos=np.array([0.0, 0.0, init_height], dtype=np.float64),
            base_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            control_mode=args.control_mode,
            auto_base_height=not args.no_auto_base_height,
            use_mujoco_passive_damping=args.use_mujoco_passive_damping,
            action_clip=action_clip,
            debug_actions=args.debug_actions,
            debug_contacts=args.debug_contacts,
            debug_action_frames=5,
            print_every=args.print_every,
            base_ang_vel_from_qvel=args.base_ang_vel_from_qvel,
            use_sensor_orientation=args.sensor_orientation,
            stand_warmup=args.stand_warmup,
            command_ramp=args.command_ramp,
            start_policy_immediately=not args.wait_for_command,
            policy_command_threshold=0.05,
            leg_kp_scale=args.leg_kp_scale,
            leg_kd_scale=args.leg_kd_scale,
            ankle_kp_scale=args.ankle_kp_scale,
            ankle_kd_scale=args.ankle_kd_scale,
            static_friction=run_params.static_friction if run_params else base_cfg.static_friction,
            dynamic_friction=run_params.dynamic_friction if run_params else base_cfg.dynamic_friction,
            restitution=run_params.restitution if run_params else base_cfg.restitution,
            mujoco_spin_friction=args.mujoco_spin_friction,
            mujoco_roll_friction=args.mujoco_roll_friction,
            patch_contacts=not args.keep_mujoco_contacts,
            mujoco_joint_names=base_cfg.mujoco_joint_names,
            foot_contact_geoms=base_cfg.foot_contact_geoms,
            default_pos=run_params.default_pos if run_params and run_params.default_pos is not None else base_cfg.default_pos.copy(),
            kp=base_cfg.kp,
            kd=base_cfg.kd,
            tau_limit=base_cfg.tau_limit,
            armature=base_cfg.armature,
        ),
    )


def print_help() -> None:
    print("=" * 60)
    print("键盘控制说明：")
    print("  ↑/↓ 或 8/2: 前进/后退 vx")
    print("  4/6: 左/右平移 vy")
    print("  ←/→ 或 7/9: 左/右转 dyaw")
    print("  0: 重置速度和机器人状态")
    print("  F: 切换相机跟随")
    print("=" * 60)


def patch_model(model: mujoco.MjModel, cfg: RobotCfg) -> None:
    if cfg.patch_contacts:
        foot_geom_ids = {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in cfg.foot_contact_geoms
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0
        }
        for geom_id in range(model.ngeom):
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name == "floor":
                model.geom_contype[geom_id] = 1
                model.geom_conaffinity[geom_id] = 1
            elif geom_id in foot_geom_ids:
                model.geom_contype[geom_id] = 1
                model.geom_conaffinity[geom_id] = 1
            else:
                model.geom_contype[geom_id] = 0
                model.geom_conaffinity[geom_id] = 0
        print(f"[INFO] active contact geoms: floor + {len(foot_geom_ids)} foot geoms")

    if cfg.static_friction is not None:
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_id >= 0:
            original = model.geom_friction[floor_id].copy()
            slide = cfg.dynamic_friction if cfg.dynamic_friction is not None else cfg.static_friction
            spin = cfg.mujoco_spin_friction if cfg.mujoco_spin_friction is not None else original[1]
            roll = cfg.mujoco_roll_friction if cfg.mujoco_roll_friction is not None else original[2]
            model.geom_friction[floor_id] = [slide, spin, roll]
            print(
                "[INFO] patched floor friction "
                f"from {np.array2string(original, precision=3)} "
                f"to {np.array2string(model.geom_friction[floor_id], precision=3)} "
                "(MuJoCo order: slide, spin, roll)"
            )
    model.dof_armature[cfg.qvel_ids] = cfg.armature
    if cfg.control_mode == "position_pd":
        model.dof_damping[:] = 0.0
        for mujoco_id, actuator_id in enumerate(cfg.actuator_ids):
            kp = float(cfg.kp[mujoco_id])
            kd = float(cfg.kd[mujoco_id])
            effort = float(cfg.tau_limit[mujoco_id])
            model.actuator_gaintype[actuator_id] = mujoco.mjtGain.mjGAIN_FIXED
            model.actuator_biastype[actuator_id] = mujoco.mjtBias.mjBIAS_AFFINE
            model.actuator_gainprm[actuator_id, :] = 0.0
            model.actuator_gainprm[actuator_id, 0] = kp
            model.actuator_biasprm[actuator_id, :] = 0.0
            model.actuator_biasprm[actuator_id, 1] = -kp
            model.actuator_biasprm[actuator_id, 2] = -kd
            model.actuator_ctrllimited[actuator_id] = 0
            model.actuator_forcelimited[actuator_id] = 1
            model.actuator_forcerange[actuator_id] = [-effort, effort]
        print("[INFO] patched MuJoCo position actuators to training PD gains and torque limits.")
        return

    if not cfg.use_mujoco_passive_damping:
        model.dof_damping[:] = 0.0
        print("[INFO] zeroed MuJoCo passive joint damping to match the training actuator model.")
    else:
        print("[INFO] keeping MuJoCo passive joint damping for debugging.")
    if cfg.control_mode == "position":
        print("[WARN] raw MuJoCo position mode uses actuator gains from the XML; prefer position_pd for trained policies.")
        return
    if cfg.control_mode != "torque_pd":
        return
    for mujoco_id, actuator_id in enumerate(cfg.actuator_ids):
        effort = float(cfg.tau_limit[mujoco_id])
        model.actuator_gaintype[actuator_id] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_biastype[actuator_id] = mujoco.mjtBias.mjBIAS_NONE
        model.actuator_gainprm[actuator_id, :] = 0.0
        model.actuator_gainprm[actuator_id, 0] = 1.0
        model.actuator_biasprm[actuator_id, :] = 0.0
        model.actuator_ctrllimited[actuator_id] = 0
        model.actuator_forcelimited[actuator_id] = 1
        model.actuator_forcerange[actuator_id] = [-effort, effort]


def hold_default_stand(model: mujoco.MjModel, data: mujoco.MjData, cfg: RobotCfg) -> None:
    data.qpos[:3] = cfg.base_pos
    data.qpos[3:7] = cfg.base_quat
    data.qpos[cfg.qpos_ids] = cfg.default_pos
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    if cfg.control_mode in ("position", "position_pd"):
        data.ctrl[cfg.actuator_ids] = cfg.default_pos
    mujoco.mj_forward(model, data)


def reset_state(model: mujoco.MjModel, data: mujoco.MjData, initial_qpos, initial_qvel, target_pos, action, hist_obs, cfg: RobotCfg) -> bool:
    if not CommandState.reset_requested:
        return False
    print("Performing reset: restoring qpos/qvel and zeroing commands")
    data.qpos[:] = initial_qpos
    data.qvel[:] = initial_qvel
    CommandState.reset()
    target_pos[:] = cfg.default_pos
    action[:] = 0.0
    hist_obs.fill(0.0)
    data.ctrl[:] = 0.0
    if cfg.control_mode in ("position", "position_pd"):
        data.ctrl[cfg.actuator_ids] = cfg.default_pos
    mujoco.mj_forward(model, data)
    CommandState.reset_requested = False
    return True


def make_observation(q, dq, omega, gravity, action, cfg: RobotCfg) -> np.ndarray:
    joint_pos_rel = q - cfg.default_pos
    num_actions = len(cfg.policy_joint_names)
    num_obs = 9 + 3 * num_actions
    q_obs = np.zeros(num_actions, dtype=np.float64)
    dq_obs = np.zeros(num_actions, dtype=np.float64)
    for policy_id, mujoco_id in enumerate(cfg.policy_to_mujoco):
        sign = cfg.policy_action_sign[policy_id]
        q_obs[policy_id] = sign * joint_pos_rel[mujoco_id]
        dq_obs[policy_id] = sign * dq[mujoco_id]

    obs = np.zeros((1, num_obs), dtype=np.float32)
    obs[0, 0:3] = omega
    obs[0, 3:6] = gravity
    obs[0, 6:9] = [CommandState.vx, CommandState.vy, CommandState.dyaw]
    obs[0, 9 : 9 + num_actions] = q_obs
    obs[0, 9 + num_actions : 9 + 2 * num_actions] = dq_obs
    obs[0, 9 + 2 * num_actions : 9 + 3 * num_actions] = action
    return obs


def policy_target(action: np.ndarray, cfg: RobotCfg) -> tuple[np.ndarray, np.ndarray]:
    target_q_policy = action * cfg.action_scale * cfg.policy_action_sign
    target_pos = cfg.default_pos.copy()
    for policy_id, mujoco_id in enumerate(cfg.policy_to_mujoco):
        target_pos[mujoco_id] += target_q_policy[policy_id]
    if cfg.clip_target_to_joint_limits:
        np.clip(target_pos, cfg.target_min, cfg.target_max, out=target_pos)
    return target_q_policy, target_pos


def print_action_debug(action, target_q, target_pos, cfg: RobotCfg) -> None:
    print("Policy action -> MuJoCo target:")
    for policy_id, mujoco_id in enumerate(cfg.policy_to_mujoco):
        print(
            f"  p{policy_id:02d} {cfg.policy_joint_names[policy_id]:28s} "
            f"a={action[policy_id]: .3f} sign={cfg.policy_action_sign[policy_id]: .0f} "
            f"dq={target_q[policy_id]: .3f} "
            f"-> m{mujoco_id:02d} {cfg.mujoco_joint_names[mujoco_id]:28s} "
            f"target={target_pos[mujoco_id]: .3f}"
        )


def init_render(model: mujoco.MjModel, data: mujoco.MjData, cfg: SimCfg, headless: bool):
    if headless:
        if cv2 is None:
            raise ImportError("Video rendering requires OpenCV; install opencv-python in the active local environment.")
        model.vis.global_.offwidth = cfg.viewer_width
        model.vis.global_.offheight = cfg.viewer_height
        renderer = mujoco.Renderer(model, width=cfg.viewer_width, height=cfg.viewer_height)
        cam = mujoco.MjvCamera()
        cam.distance, cam.azimuth, cam.elevation, cam.lookat = 4.0, 45.0, -20.0, [0, 0, 1]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 1.0 / cfg.dt / cfg.decimation
        out = cv2.VideoWriter("simulation.mp4", fourcc, fps, (cfg.viewer_width, cfg.viewer_height))
        return renderer, cam, out, None

    if mujoco_viewer is None:
        raise ImportError("Window rendering requires mujoco-python-viewer in the active local environment.")
    viewer = mujoco_viewer.MujocoViewer(
        model,
        data,
        mode="window",
        width=cfg.viewer_width,
        height=cfg.viewer_height,
        hide_menus=True,
    )
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.lookat = 4.0, 45.0, -20.0, [0, 0, 1]
    return None, None, None, viewer


def render_frame(data, renderer, cam, out, viewer, headless: bool) -> bool:
    if CommandState.camera_follow:
        base_pos = [float(x) for x in data.qpos[0:3]]
        if headless:
            cam.lookat = base_pos
        else:
            viewer.cam.lookat = base_pos
    if headless:
        renderer.update_scene(data, camera=cam)
        out.write(renderer.render())
    else:
        try:
            viewer.render()
        except Exception as exc:
            if "GLFW window does not exist" in str(exc):
                print("Viewer window closed; exiting simulation.")
                return False
            raise
    return True


def print_mapping(cfg: RobotCfg) -> None:
    print("Policy/训练关节顺序 -> MuJoCo关节顺序:")
    for policy_id, mujoco_id in enumerate(cfg.policy_to_mujoco):
        print(
            f"  {policy_id:02d} {cfg.policy_joint_names[policy_id]} "
            f"-> {mujoco_id:02d} {cfg.mujoco_joint_names[mujoco_id]} "
            f"sign={cfg.policy_action_sign[policy_id]:+.0f}"
        )


def summarize_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, int]:
    counts: dict[str, int] = {}
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or f"geom_{contact.geom1}"
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or f"geom_{contact.geom2}"
        pair = " / ".join(sorted((name1, name2)))
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def append_plot_data(store, step, target_pos, q, tau, lin_vel, ang_vel_z, dt: float):
    store["time"].append(step * dt)
    store["target"].append(target_pos.copy())
    store["actual"].append(q.copy())
    store["tau"].append(tau.copy())
    store["cmd_vx"].append(CommandState.vx)
    store["cmd_vy"].append(CommandState.vy)
    store["cmd_yaw"].append(CommandState.dyaw)
    store["lin_vel"].append(lin_vel[:2].copy())
    store["ang_vel"].append(float(ang_vel_z))


def save_plots(store, num_actions: int) -> None:
    if not store["time"]:
        return
    time_data = np.array(store["time"])
    target = np.array(store["target"])
    actual = np.array(store["actual"])
    lin_vel = np.array(store["lin_vel"])
    ang_vel = np.array(store["ang_vel"])

    rows, cols = (num_actions + 3) // 4, 4
    fig1, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), sharex=True)
    for i, ax in enumerate(axes.flatten()):
        if i >= num_actions:
            fig1.delaxes(ax)
            continue
        ax.plot(time_data, target[:, i], label="Commanded", linestyle="--")
        ax.plot(time_data, actual[:, i], label="Actual")
        ax.set_title(f"Joint {i + 1}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Position [rad]")
        ax.legend()
        ax.grid(True)
    fig1.suptitle("Commanded vs Actual Joint Positions", fontsize=16)
    plt.tight_layout()
    fig1.savefig("joint_positions.png")

    fig2, axes2 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    axes2[0].plot(time_data, store["cmd_vx"], label="Commanded Vx", linestyle="--")
    axes2[0].plot(time_data, lin_vel[:, 0], label="Actual Vx")
    axes2[1].plot(time_data, store["cmd_vy"], label="Commanded Vy", linestyle="--")
    axes2[1].plot(time_data, lin_vel[:, 1], label="Actual Vy")
    axes2[2].plot(time_data, store["cmd_yaw"], label="Commanded Dyaw", linestyle="--")
    axes2[2].plot(time_data, ang_vel, label="Actual Dyaw")
    for ax, title, ylabel in zip(
        axes2,
        ["Base Linear Velocity X", "Base Linear Velocity Y", "Base Angular Velocity Z (Dyaw)"],
        ["Velocity [m/s]", "Velocity [m/s]", "Angular Velocity [rad/s]"],
    ):
        ax.set_title(title)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)
    fig2.suptitle("Commanded vs Actual Base Velocities", fontsize=16)
    plt.tight_layout()
    fig2.savefig("base_velocities.png")
    print("Plots finished.")


def run_mujoco(policy, cfg: RunCfg, headless: bool) -> None:
    print_help()
    listener = start_keyboard_listener()

    model = mujoco.MjModel.from_xml_path(cfg.sim.model_path)
    model.opt.timestep = cfg.sim.dt
    robot = cfg.robot
    robot.policy_to_mujoco = make_index_mapping(robot.policy_joint_names, robot.mujoco_joint_names)
    (
        _joint_ids,
        robot.qpos_ids,
        robot.qvel_ids,
        robot.actuator_ids,
        robot.target_min,
        robot.target_max,
    ) = build_joint_data(model, robot.mujoco_joint_names)
    print_mapping(robot)
    patch_model(model, robot)

    data = mujoco.MjData(model)
    if robot.auto_base_height:
        robot.base_pos[2] = grounded_base_height(model, data, robot)
        print(f"[INFO] auto_base_height={robot.base_pos[2]:.3f}")
    hold_default_stand(model, data, robot)
    initial_qpos, initial_qvel = data.qpos.copy(), data.qvel.copy()

    renderer, cam, out, viewer = init_render(model, data, cfg.sim, headless)
    target_pos = robot.default_pos.copy()
    action = np.zeros(cfg.sim.num_actions, dtype=np.float64)
    hist_obs = np.zeros((cfg.sim.frame_stack, cfg.sim.num_obs), dtype=np.float64)
    tau = np.zeros(cfg.sim.num_actions, dtype=np.float64)

    warmup_steps = int(round(robot.stand_warmup / cfg.sim.dt))
    command_ramp_steps = int(round(robot.command_ramp / cfg.sim.dt))
    policy_active = robot.start_policy_immediately
    policy_start_step = 0 if policy_active else None
    first_obs = True
    start_time = time.time()
    store = {k: [] for k in ["time", "target", "actual", "tau", "cmd_vx", "cmd_vy", "cmd_yaw", "lin_vel", "ang_vel"]}
    min_root_z = float("inf")
    max_abs_yaw_rate = 0.0
    max_abs_vy = 0.0
    last_contacts: dict[str, int] = {}

    for step in tqdm(range(int(cfg.sim.duration / cfg.sim.dt)), desc="Simulating..."):
        if reset_state(model, data, initial_qpos, initial_qvel, target_pos, action, hist_obs, robot):
            policy_active, policy_start_step, first_obs = robot.start_policy_immediately, 0, True

        if not policy_active:
            hold_default_stand(model, data, robot)

        qpos, qvel, lin_vel, omega, gravity = get_obs(data, robot.base_ang_vel_from_qvel, robot.use_sensor_orientation)
        q = qpos[robot.qpos_ids]
        dq = qvel[robot.qvel_ids]
        min_root_z = min(min_root_z, float(qpos[2]))
        max_abs_yaw_rate = max(max_abs_yaw_rate, abs(float(omega[2])))
        max_abs_vy = max(max_abs_vy, abs(float(lin_vel[1])))

        if step % cfg.sim.decimation == 0:
            obs = make_observation(q, dq, omega, gravity, action, robot)
            print_every = max(1, robot.print_every)
            if (step // cfg.sim.decimation) % print_every == 0:
                print(f"当前命令: 线速度x={CommandState.vx:.2f}, 线速度y={CommandState.vy:.2f}, 角速度z={CommandState.dyaw:.2f}")
                print(f"当前速度: 线速度x={lin_vel[0]:.2f}, 线速度y={lin_vel[1]:.2f}, 角速度z={omega[2]:.2f}")
                if robot.debug_contacts:
                    last_contacts = summarize_contacts(model, data)
                    print(f"当前接触: {last_contacts}")

            if first_obs:
                hist_obs = np.tile(obs, (cfg.sim.frame_stack, 1))
                first_obs = False
            else:
                hist_obs = np.concatenate((hist_obs[1:], obs.reshape(1, -1)), axis=0)

            command_norm = np.linalg.norm([CommandState.vx, CommandState.vy, CommandState.dyaw])
            if not policy_active and step >= warmup_steps and command_norm > robot.policy_command_threshold:
                policy_active = True
                policy_start_step = step
                action[:] = 0.0
                print("Policy enabled by non-zero command.")

            if policy_active and step >= warmup_steps:
                if policy_start_step is not None and command_ramp_steps > 0:
                    alpha = np.clip((step - policy_start_step) / command_ramp_steps, 0.0, 1.0)
                    obs[0, 6:9] = np.array([CommandState.vx, CommandState.vy, CommandState.dyaw], dtype=np.float32) * alpha
                    hist_obs[-1, :] = obs
                action[:] = policy(hist_obs.reshape(1, -1))
                if robot.action_clip is not None:
                    np.clip(action, -robot.action_clip, robot.action_clip, out=action)
                if robot.debug_actions and step < warmup_steps + cfg.sim.decimation * robot.debug_action_frames:
                    print("raw_action:", np.array2string(action, precision=3, suppress_small=True))
            else:
                action[:] = 0.0

            target_q, target_pos = policy_target(action, robot)
            if not policy_active:
                target_pos[:] = robot.default_pos
                hold_default_stand(model, data, robot)
            if robot.debug_actions and step < warmup_steps + cfg.sim.decimation * robot.debug_action_frames:
                print_action_debug(action, target_q, target_pos, robot)

            append_plot_data(store, step, target_pos, q, tau, lin_vel, omega[2], cfg.sim.dt)
            if not render_frame(data, renderer, cam, out, viewer, headless):
                break

        if not policy_active:
            hold_default_stand(model, data, robot)
        elif robot.control_mode in ("position", "position_pd"):
            data.ctrl[robot.actuator_ids] = target_pos
            mujoco.mj_step(model, data)
        else:
            target_vel = np.zeros(cfg.sim.num_actions, dtype=np.float64)
            tau = pd_control(target_pos, q, robot.kp, target_vel, dq, robot.kd)
            tau = np.clip(tau, -robot.tau_limit, robot.tau_limit)
            data.ctrl[robot.actuator_ids] = tau
            mujoco.mj_step(model, data)

        elapsed = time.time() - start_time
        target_time = (step + 1) * cfg.sim.dt
        if elapsed < target_time:
            time.sleep(target_time - elapsed)

    last_contacts = summarize_contacts(model, data)
    print(
        "[SUMMARY] "
        f"min_root_z={min_root_z:.3f}, "
        f"max_abs_yaw_rate={max_abs_yaw_rate:.3f}, "
        f"max_abs_vy={max_abs_vy:.3f}, "
        f"final_contacts={last_contacts}"
    )

    if headless:
        out.release()
    else:
        viewer.close()
    if listener is not None:
        listener.stop()

    print("Simulation finished. Generating plots...")
    save_plots(store, cfg.sim.num_actions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lens110 MuJoCo sim2sim runner.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Sim2sim JSON config path.")
    parser.add_argument("--load_model", default=None, help="Override TorchScript policy path from config.")
    parser.add_argument("--load_onnx", default=None, help="Override ONNX policy path from config.")
    parser.add_argument(
        "--policy_backend",
        choices=("torchscript", "onnx"),
        default=None,
        help="Override policy inference backend from config.",
    )
    parser.add_argument(
        "--params_dir",
        default=None,
        help="Optional training params directory containing env.yaml and agent.yaml. Overrides config values when set.",
    )
    parser.add_argument("--terrain", action="store_true", default=False, help="Kept for CLI compatibility.")
    parser.add_argument("--headless", action="store_true", help="Run without GUI and save simulation.mp4.")
    parser.add_argument("--sim_duration", type=float, default=10000.0, help="Simulation duration in seconds.")
    parser.add_argument("--viewer_width", type=int, default=1280, help="MuJoCo viewer/video width.")
    parser.add_argument("--viewer_height", type=int, default=720, help="MuJoCo viewer/video height.")
    parser.add_argument("--model_path", default=None, help="Override MuJoCo XML model path from config.")
    parser.add_argument("--cmd_vel", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="Initial command: vx vy yaw_rate.")
    parser.add_argument("--action_clip", type=float, default=None, help="Override policy action clip. Default follows params.")
    parser.add_argument("--debug_actions", action="store_true", help="Print first policy action mappings.")
    parser.add_argument("--debug_contacts", action="store_true", help="Print active contact pairs every --print_every policy steps.")
    parser.add_argument("--control_mode", choices=["position_pd", "torque_pd", "position"], default="position_pd")
    parser.add_argument("--leg_kp_scale", type=float, default=1.0, help="Scale hip/knee/ankle PD stiffness.")
    parser.add_argument("--leg_kd_scale", type=float, default=1.0, help="Scale hip/knee/ankle PD damping.")
    parser.add_argument("--ankle_kp_scale", type=float, default=1.0, help="Additional ankle stiffness scale.")
    parser.add_argument("--ankle_kd_scale", type=float, default=1.0, help="Additional ankle damping scale.")
    parser.add_argument("--print_every", type=int, default=10, help="Print every N policy steps.")
    parser.add_argument("--init_height", type=float, default=None, help="Override initial pelvis height from config.")
    parser.add_argument("--no_auto_base_height", action="store_true", help="Use --init_height directly.")
    parser.add_argument("--stand_warmup", type=float, default=0.0, help="Hold stand pose before policy starts.")
    parser.add_argument("--command_ramp", type=float, default=0.0, help="Ramp commanded velocity after policy starts.")
    parser.add_argument("--wait_for_command", action="store_true", help="Hold stand pose until a non-zero command is given.")
    parser.add_argument("--use_mujoco_passive_damping", action="store_true", help="Keep MJCF joint damping.")
    parser.add_argument("--keep_mujoco_contacts", action="store_true", help="Keep contact geom masks from the MJCF file.")
    parser.add_argument("--mujoco_spin_friction", type=float, default=None, help="Override MuJoCo torsional friction for the floor.")
    parser.add_argument("--mujoco_roll_friction", type=float, default=None, help="Override MuJoCo rolling friction for the floor.")
    parser.add_argument("--mujoco_joint_order", action="store_true", help="Use MuJoCo joint order for debugging.")
    parser.add_argument("--isaac_urdf_joint_order", action="store_true", help="Use raw URDF parser order for debugging.")
    parser.add_argument("--flip_leg_pitch", action="store_true")
    parser.add_argument("--flip_hip_pitch", action="store_true")
    parser.add_argument("--flip_knee", action="store_true")
    parser.add_argument("--flip_ankle_pitch", action="store_true")
    parser.add_argument("--flip_roll", action="store_true")
    parser.add_argument("--base_ang_vel_from_qvel", action="store_true", help="Debug: derive base angular velocity from qvel.")
    parser.add_argument("--base_ang_vel_from_sensor", action="store_true", help="Deprecated; sensor gyro is now the default.")
    parser.add_argument("--root_orientation_from_qpos", action="store_true", default=True)
    parser.add_argument("--sensor_orientation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = load_sim2sim_config(args.config)
    cli_params_dir = args.params_dir
    args.load_model = resolve_path(args.load_model) if args.load_model else base_cfg.torchscript_policy
    args.load_onnx = resolve_path(args.load_onnx) if args.load_onnx else base_cfg.onnx_policy
    args.policy_backend = args.policy_backend or base_cfg.policy_backend
    args.params_dir = (
        resolve_path(cli_params_dir, os.path.dirname(os.path.abspath(args.config)))
        if cli_params_dir
        else base_cfg.source_params_dir
    )
    if args.params_dir and not has_run_params(args.params_dir):
        if cli_params_dir:
            raise FileNotFoundError(f"Missing env.yaml or agent.yaml under params dir: {args.params_dir}")
        print(f"[WARN] Config params_dir is missing env.yaml/agent.yaml, using JSON values: {args.params_dir}")
        args.params_dir = None
    args.model_path = resolve_path(args.model_path) if args.model_path else base_cfg.model_path
    args.init_height = (
        args.init_height
        if args.init_height is not None
        else (base_cfg.init_height or base_cfg.fallback_init_height)
    )

    CommandState.set_initial(tuple(args.cmd_vel))
    if args.policy_backend == "onnx":
        policy = OnnxPolicy(args.load_onnx)
        print(f"[INFO] loaded ONNX policy with {policy.backend}: {args.load_onnx}")
    else:
        policy = TorchScriptPolicy(args.load_model)
        print(f"[INFO] loaded TorchScript policy: {args.load_model}")
    run_mujoco(policy, make_cfg(args, base_cfg), args.headless)


if __name__ == "__main__":
    main()
