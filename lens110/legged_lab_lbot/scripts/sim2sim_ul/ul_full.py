# SPDX-License-Identifier: BSD-3-Clause
"""Compact MuJoCo sim2sim runner for Lens110 upper/lower ankle commands.

Two policy IO modes are supported:

* ``pitch_roll`` keeps the legacy policy observation/action semantics and maps
  commanded ankle pitch/roll targets to physical upper/lower ankle motors.
* ``upper_lower`` feeds the policy real upper/lower ankle motor states and
  applies policy ankle outputs directly to upper/lower motors.
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
from scipy.optimize import least_squares
from tqdm import tqdm

try:
    from pynput import keyboard
except ImportError:
    keyboard = None


DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "ul.json")
UPPER_LOWER_JOINT_NAMES = (
    "left_ankle_upper_joint",
    "left_ankle_lower_joint",
    "right_ankle_upper_joint",
    "right_ankle_lower_joint",
)
PITCH_ROLL_JOINT_NAMES = (
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
POLICY_ANKLE_TO_CONTROL_JOINT = {
    "left_ankle_pitch_joint": "left_ankle_upper_joint",
    "left_ankle_roll_joint": "left_ankle_lower_joint",
    "right_ankle_pitch_joint": "right_ankle_upper_joint",
    "right_ankle_roll_joint": "right_ankle_lower_joint",
    "left_ankle_upper_joint": "left_ankle_upper_joint",
    "left_ankle_lower_joint": "left_ankle_lower_joint",
    "right_ankle_upper_joint": "right_ankle_upper_joint",
    "right_ankle_lower_joint": "right_ankle_lower_joint",
}
UPPER_LOWER_TO_PITCH_ROLL_JOINT = {
    "left_ankle_upper_joint": "left_ankle_pitch_joint",
    "left_ankle_lower_joint": "left_ankle_roll_joint",
    "right_ankle_upper_joint": "right_ankle_pitch_joint",
    "right_ankle_lower_joint": "right_ankle_roll_joint",
}
GROUND_GEOM_NAMES = {"floor", "ground"}


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


def default_ankle_model_path() -> str:
    return os.path.join(repo_root(), "pitchRoll2UpperLower", "ankle_model", "models")


def default_policy_ankle_model_path() -> str:
    return os.path.join(repo_root(), "pitchRoll2UpperLower", "ankle_model", "models")


@dataclass
class RunParams:
    params_dir: str
    default_pos: np.ndarray | None = None
    action_scale: np.ndarray | None = None
    init_height: float | None = None
    action_clip: float | None = None
    policy_ankle_model_path: str | None = None
    ankle_pd_space: str | None = None
    static_friction: float | None = None
    dynamic_friction: float | None = None
    restitution: float | None = None


@dataclass
class Sim2SimConfig:
    torchscript_policy: str | None
    onnx_policy: str | None
    policy_backend: str
    policy_io: str
    upper_lower_action_mode: str
    upper_lower_observation_source: str
    model_path: str
    source_params_dir: str | None
    ankle_model_path: str
    policy_ankle_model_path: str
    dt: float
    decimation: int
    sim_duration: float
    default_cmd_vel: tuple[float, float, float]
    print_every: int
    num_actions: int
    num_obs: int
    mujoco_joint_names: list[str]
    control_joint_names: list[str]
    policy_joint_names: list[str]
    policy_joint_signs: np.ndarray
    default_pos: np.ndarray
    stand_default_pos: np.ndarray | None
    passive_joint_default_pos: dict[str, float]
    action_scale: np.ndarray
    init_height: float | None
    fallback_init_height: float
    action_clip: float | None
    clip_policy_targets_to_joint_limits: bool
    clip_control_targets_to_joint_limits: bool
    kp: np.ndarray
    kd: np.ndarray
    tau_limit: np.ndarray
    armature: np.ndarray
    upper_lower_action_clip_min: np.ndarray | None
    upper_lower_action_clip_max: np.ndarray | None
    upper_lower_control_clip_min: np.ndarray | None
    upper_lower_control_clip_max: np.ndarray | None
    foot_contact_geoms: list[str]
    static_friction: float | None
    dynamic_friction: float | None
    tendon_stiffness: float | None
    tendon_damping: float | None
    tendon_frictionloss: float | None
    ankle_motor_kp_scale: float
    ankle_motor_kd_scale: float
    ankle_motor_tau_scale: float
    ankle_pd_space: str
    ankle_command_rel_clip: float | None
    ankle_command_rate_limit: float | None
    use_training_foot_collision_boxes: bool
    use_training_passive_ankle_limits: bool


@dataclass
class RobotRuntime:
    policy_to_mujoco: list[int]
    control_to_mujoco: list[int]
    default_pos: np.ndarray
    control_default_pos: np.ndarray
    action_scale: np.ndarray
    policy_joint_signs: np.ndarray
    base_pos: np.ndarray
    action_clip: float | None
    static_friction: float | None
    dynamic_friction: float | None
    qpos_ids: np.ndarray | None = None
    qvel_ids: np.ndarray | None = None
    control_qpos_ids: np.ndarray | None = None
    control_qvel_ids: np.ndarray | None = None
    actuator_ids: np.ndarray | None = None
    observe_target_min: np.ndarray | None = None
    observe_target_max: np.ndarray | None = None
    control_target_min: np.ndarray | None = None
    control_target_max: np.ndarray | None = None
    passive_default_qpos_ids: np.ndarray | None = None
    passive_default_values: np.ndarray | None = None


def as_float_array(config: dict, key: str, expected_len: int) -> np.ndarray:
    values = np.array(config[key], dtype=np.float64)
    if values.shape != (expected_len,):
        raise ValueError(f"Config key '{key}' must contain {expected_len} values, got shape {values.shape}")
    return values


def parse_named_clip(
    clip_config: dict[str, list[float]] | None,
    joint_names: list[str],
    *,
    ignore_unknown: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if clip_config is None:
        return None, None
    clip_min = np.full(len(joint_names), -np.inf, dtype=np.float64)
    clip_max = np.full(len(joint_names), np.inf, dtype=np.float64)
    joint_index = {name: index for index, name in enumerate(joint_names)}
    for joint_name, bounds in clip_config.items():
        if joint_name not in joint_index:
            if ignore_unknown:
                continue
            raise ValueError(f"upper_lower_action_clip contains unknown joint '{joint_name}'")
        if len(bounds) != 2:
            raise ValueError(f"upper_lower_action_clip['{joint_name}'] must contain [min, max]")
        clip_min[joint_index[joint_name]] = float(bounds[0])
        clip_max[joint_index[joint_name]] = float(bounds[1])
    return clip_min, clip_max


def parse_policy_joint_signs(sign_config: list[float] | dict[str, float] | None, policy_joint_names: list[str]) -> np.ndarray:
    signs = np.ones(len(policy_joint_names), dtype=np.float64)
    if sign_config is None:
        return signs
    if isinstance(sign_config, list):
        values = np.asarray(sign_config, dtype=np.float64)
        if values.shape != signs.shape:
            raise ValueError(
                f"policy_joint_signs list length must match policy_joint_names: {len(values)} != {len(signs)}"
            )
        return values
    if isinstance(sign_config, dict):
        policy_index = {name: index for index, name in enumerate(policy_joint_names)}
        for joint_name, sign in sign_config.items():
            if joint_name not in policy_index:
                raise ValueError(f"policy_joint_signs contains unknown policy joint '{joint_name}'")
            signs[policy_index[joint_name]] = float(sign)
        return signs
    raise ValueError("robot.policy_joint_signs must be a list, dict, or null")


def load_config(path: str) -> Sim2SimConfig:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    config_dir = os.path.dirname(os.path.abspath(path))
    robot = raw["robot"]
    num_actions = int(robot["num_actions"])
    mujoco_joint_names = list(robot["mujoco_joint_names"])
    control_joint_names = list(robot.get("control_joint_names", mujoco_joint_names))
    policy_joint_names = list(robot["policy_joint_names"])
    if len(mujoco_joint_names) != num_actions:
        raise ValueError("mujoco_joint_names length must match robot.num_actions")
    if len(control_joint_names) != num_actions:
        raise ValueError("control_joint_names length must match robot.num_actions")
    if len(policy_joint_names) != num_actions:
        raise ValueError("policy_joint_names length must match robot.num_actions")

    simulation = raw["simulation"]
    physics = raw.get("physics", {})
    source = raw.get("source", {})
    ankle_motor = raw.get("ankle_motor", {})
    command_safety = raw.get("ankle_command_safety", {})
    default_cmd_vel = tuple(float(v) for v in simulation.get("default_cmd_vel", (0.0, 0.0, 0.0)))
    if len(default_cmd_vel) != 3:
        raise ValueError("simulation.default_cmd_vel must contain three values: vx, vy, yaw_rate")
    policy_io = raw["policy"].get("io", "pitch_roll")
    if policy_io not in {"pitch_roll", "upper_lower"}:
        raise ValueError("policy.io must be either 'pitch_roll' or 'upper_lower'")
    upper_lower_action_mode = raw["policy"].get("upper_lower_action_mode", "through_pitch_roll")
    if upper_lower_action_mode not in {"direct", "through_pitch_roll"}:
        raise ValueError("policy.upper_lower_action_mode must be either 'direct' or 'through_pitch_roll'")
    upper_lower_observation_source = raw["policy"].get("upper_lower_observation_source", "converted_pitch_roll")
    if upper_lower_observation_source not in {"control", "converted_pitch_roll"}:
        raise ValueError("policy.upper_lower_observation_source must be either 'control' or 'converted_pitch_roll'")
    ankle_pd_space = raw["policy"].get("ankle_pd_space", "upper_lower")
    if ankle_pd_space not in {"pitch_roll", "upper_lower", "mlp_torque_roundtrip"}:
        raise ValueError("policy.ankle_pd_space must be pitch_roll, upper_lower, or mlp_torque_roundtrip")
    upper_lower_action_clip_min, upper_lower_action_clip_max = parse_named_clip(
        robot.get("upper_lower_action_clip"),
        policy_joint_names,
    )
    upper_lower_control_clip_min, upper_lower_control_clip_max = parse_named_clip(
        robot.get("upper_lower_action_clip"),
        control_joint_names,
        ignore_unknown=True,
    )

    return Sim2SimConfig(
        torchscript_policy=resolve_path(raw["policy"]["torchscript"], config_dir),
        onnx_policy=resolve_path(raw["policy"]["onnx"], config_dir),
        policy_backend=raw["policy"].get("backend", "torchscript"),
        policy_io=policy_io,
        upper_lower_action_mode=upper_lower_action_mode,
        upper_lower_observation_source=upper_lower_observation_source,
        model_path=resolve_path(raw["model"]["path"], config_dir),
        source_params_dir=resolve_path(source.get("params_dir"), config_dir) if source.get("params_dir") else None,
        ankle_model_path=resolve_path(raw["model"].get("ankle_model_path", default_ankle_model_path()), config_dir),
        policy_ankle_model_path=resolve_path(
            raw["model"].get("policy_ankle_model_path", default_policy_ankle_model_path()), config_dir
        ),
        dt=float(simulation["dt"]),
        decimation=int(simulation["decimation"]),
        sim_duration=float(simulation.get("duration", 10000.0)),
        default_cmd_vel=default_cmd_vel,
        print_every=int(simulation.get("print_every", 10)),
        num_actions=num_actions,
        num_obs=9 + 3 * num_actions,
        mujoco_joint_names=mujoco_joint_names,
        control_joint_names=control_joint_names,
        policy_joint_names=policy_joint_names,
        policy_joint_signs=parse_policy_joint_signs(robot.get("policy_joint_signs"), policy_joint_names),
        default_pos=as_float_array(robot, "default_pos", num_actions),
        stand_default_pos=as_float_array(robot, "stand_default_pos", num_actions)
        if "stand_default_pos" in robot
        else None,
        passive_joint_default_pos={name: float(value) for name, value in robot.get("passive_joint_default_pos", {}).items()},
        action_scale=as_float_array(robot, "action_scale", num_actions),
        init_height=None if robot.get("init_height") is None else float(robot["init_height"]),
        fallback_init_height=float(robot.get("fallback_init_height", 0.68)),
        action_clip=None if robot.get("action_clip") is None else float(robot["action_clip"]),
        clip_policy_targets_to_joint_limits=bool(robot.get("clip_policy_targets_to_joint_limits", True)),
        clip_control_targets_to_joint_limits=bool(robot.get("clip_control_targets_to_joint_limits", True)),
        kp=as_float_array(robot, "kp", num_actions),
        kd=as_float_array(robot, "kd", num_actions),
        tau_limit=as_float_array(robot, "tau_limit", num_actions),
        armature=as_float_array(robot, "armature", num_actions),
        upper_lower_action_clip_min=upper_lower_action_clip_min,
        upper_lower_action_clip_max=upper_lower_action_clip_max,
        upper_lower_control_clip_min=upper_lower_control_clip_min,
        upper_lower_control_clip_max=upper_lower_control_clip_max,
        foot_contact_geoms=list(robot["foot_contact_geoms"]),
        static_friction=None if physics.get("static_friction") is None else float(physics["static_friction"]),
        dynamic_friction=None if physics.get("dynamic_friction") is None else float(physics["dynamic_friction"]),
        tendon_stiffness=None if physics.get("tendon_stiffness") is None else float(physics["tendon_stiffness"]),
        tendon_damping=None if physics.get("tendon_damping") is None else float(physics["tendon_damping"]),
        tendon_frictionloss=None
        if physics.get("tendon_frictionloss") is None
        else float(physics["tendon_frictionloss"]),
        ankle_motor_kp_scale=float(ankle_motor.get("kp_scale", 1.0)),
        ankle_motor_kd_scale=float(ankle_motor.get("kd_scale", 1.0)),
        ankle_motor_tau_scale=float(ankle_motor.get("tau_scale", 1.0)),
        ankle_pd_space=ankle_pd_space,
        ankle_command_rel_clip=None
        if command_safety.get("rel_clip") is None
        else float(command_safety["rel_clip"]),
        ankle_command_rate_limit=None
        if command_safety.get("rate_limit") is None
        else float(command_safety["rate_limit"]),
        use_training_foot_collision_boxes=bool(physics.get("use_training_foot_collision_boxes", True)),
        use_training_passive_ankle_limits=bool(physics.get("use_training_passive_ankle_limits", True)),
    )


def validate_policy_joint_order(cfg: Sim2SimConfig) -> None:
    if cfg.policy_io != "upper_lower":
        return
    if len(set(cfg.policy_joint_names)) != len(cfg.policy_joint_names):
        raise ValueError("upper_lower policy_joint_names contains duplicate joints")
    required = set(UPPER_LOWER_JOINT_NAMES)
    missing = sorted(required - set(cfg.policy_joint_names))
    if missing:
        raise ValueError(f"upper_lower policy_joint_names must expose upper/lower ankle actions; missing: {missing}")
    forbidden = sorted(set(PITCH_ROLL_JOINT_NAMES) & set(cfg.policy_joint_names))
    if forbidden:
        raise ValueError(f"upper_lower policy_joint_names must not expose pitch/roll ankle names: {forbidden}")
    try:
        make_index_mapping(cfg.policy_joint_names, cfg.mujoco_joint_names)
    except KeyError as exc:
        raise ValueError(f"upper_lower policy_joint_names contains a joint that cannot map to MuJoCo: {exc}") from exc


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


def resolve_ankle_model_path(path: str) -> str:
    candidate = resolve_path(path)
    if os.path.isdir(candidate):
        weighted_mlp_path = os.path.join(candidate, "pr_to_ul_state", "scripted.pth")
        if os.path.exists(weighted_mlp_path):
            return candidate
        mlp_path = os.path.join(candidate, "pos_pr2ul_mlp_scripted.pth")
        if os.path.exists(mlp_path):
            return mlp_path
        npz_path = os.path.join(candidate, "ankle_poly_models.npz")
        if os.path.exists(npz_path):
            return npz_path
        pkl_path = os.path.join(candidate, "pos_pr2ul.pkl")
        if os.path.exists(pkl_path):
            return candidate
    return candidate


class PolynomialRidgeModel:
    def __init__(self, powers: np.ndarray, coef: np.ndarray, intercept: np.ndarray):
        self.powers = np.asarray(powers, dtype=np.int64)
        self.coef = np.asarray(coef, dtype=np.float64)
        self.intercept = np.asarray(intercept, dtype=np.float64)

    def __call__(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        values_shape = values.shape
        values = values.reshape(-1, 4)
        features = np.prod(np.power(values[:, None, :], self.powers[None, :, :]), axis=-1)
        output = features @ self.coef.T + self.intercept
        return output.reshape(*values_shape[:-1], 4)


class AnkleCommandModel:
    """Evaluate the fitted pitch/roll -> upper/lower position model."""

    def __init__(self, model_path: str):
        self.path = resolve_ankle_model_path(model_path)
        self._model_kind = "legacy"
        self._weighted_models = None
        if os.path.isdir(self.path) and os.path.exists(os.path.join(self.path, "pr_to_ul_state", "scripted.pth")):
            self._model_kind = "weighted_mlp"
            self._weighted_models = {
                "pr_to_ul_state": torch.jit.load(
                    os.path.join(self.path, "pr_to_ul_state", "scripted.pth"), map_location="cpu"
                ).eval(),
                "ul_to_pr_state": torch.jit.load(
                    os.path.join(self.path, "ul_to_pr_state", "scripted.pth"), map_location="cpu"
                ).eval(),
                "pr_to_ul_torque": torch.jit.load(
                    os.path.join(self.path, "pr_to_ul_torque", "scripted.pth"), map_location="cpu"
                ).eval(),
                "ul_to_pr_torque": torch.jit.load(
                    os.path.join(self.path, "ul_to_pr_torque", "scripted.pth"), map_location="cpu"
                ).eval(),
            }
            self._pos_pr2ul_model = None
            self._pos_ul2pr_model = None
            self._vel_pr2ul_model = None
        elif os.path.isdir(self.path):
            self._pos_pr2ul_model = self._load_pkl(os.path.join(self.path, "pos_pr2ul.pkl"))
            ul2pr_path = os.path.join(self.path, "pos_ul2pr.pkl")
            self._pos_ul2pr_model = self._load_pkl(ul2pr_path) if os.path.exists(ul2pr_path) else None
            vel_pr2ul_path = os.path.join(self.path, "vel_pr2ul.pkl")
            self._vel_pr2ul_model = self._load_pkl(vel_pr2ul_path) if os.path.exists(vel_pr2ul_path) else None
        elif self.path.endswith(".pth"):
            self._pos_pr2ul_model = torch.jit.load(self.path, map_location="cpu")
            self._pos_pr2ul_model.eval()
            self._pos_ul2pr_model = None
            self._vel_pr2ul_model = None
        elif self.path.endswith(".npz"):
            model_data = np.load(self.path)
            self._pos_pr2ul_model = PolynomialRidgeModel(
                model_data["pos_pr2ul_powers"],
                model_data["pos_pr2ul_coef"],
                model_data["pos_pr2ul_intercept"],
            )
            self._pos_ul2pr_model = PolynomialRidgeModel(
                model_data["pos_ul2pr_powers"],
                model_data["pos_ul2pr_coef"],
                model_data["pos_ul2pr_intercept"],
            )
            self._vel_pr2ul_model = PolynomialRidgeModel(
                model_data["vel_pr2ul_powers"],
                model_data["vel_pr2ul_coef"],
                model_data["vel_pr2ul_intercept"],
            )
        elif self.path.endswith(".pkl"):
            self._pos_pr2ul_model = self._load_pkl(self.path)
            self._pos_ul2pr_model = None
            self._vel_pr2ul_model = None
        else:
            raise ValueError(
                "ankle_model_path must be weighted_mlp directory, pos_pr2ul_mlp_scripted.pth, "
                "ankle_poly_models.npz, pos_pr2ul.pkl, or a directory containing one of them: "
                f"{model_path}"
            )

    @staticmethod
    def _load_pkl(path: str):
        try:
            import joblib
        except ImportError as exc:
            raise ImportError(f"Loading {path} requires joblib. Use ankle_poly_models.npz to avoid this dependency.") from exc
        return joblib.load(path)

    def _predict(self, model, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64).reshape(-1, 4)
        if isinstance(model, PolynomialRidgeModel):
            return model(values)
        if hasattr(model, "predict"):
            return model.predict(values)
        values_f32 = values.astype(np.float32)
        with torch.inference_mode():
            return model(torch.from_numpy(values_f32)).detach().cpu().numpy().astype(np.float64)

    def _predict_weighted(self, model_name: str, values: np.ndarray, output_dim: int) -> np.ndarray:
        if self._weighted_models is None:
            raise ValueError(f"{self.path} is not a weighted MLP ankle model directory")
        values = np.asarray(values, dtype=np.float64)
        values_shape = values.shape
        values = values.reshape(-1, values_shape[-1]).astype(np.float32)
        with torch.inference_mode():
            output = self._weighted_models[model_name](torch.from_numpy(values)).detach().cpu().numpy()
        return output.astype(np.float64).reshape(*values_shape[:-1], output_dim)

    def _pr2ul_sidewise(self, values: np.ndarray, model) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64).reshape(-1, 4)
        left_inputs = np.column_stack((values[:, 0], values[:, 1], values[:, 0], -values[:, 1]))
        right_inputs = np.column_stack((values[:, 2], -values[:, 3], values[:, 2], values[:, 3]))
        left_outputs = self._predict(model, left_inputs)
        right_outputs = self._predict(model, right_inputs)
        outputs = np.empty((values.shape[0], 4), dtype=np.float64)
        outputs[:, 0:2] = left_outputs[:, 0:2]
        outputs[:, 2:4] = right_outputs[:, 2:4]
        return outputs

    def pos_pr2ul(self, values: np.ndarray) -> np.ndarray:
        if self._model_kind == "weighted_mlp":
            values = np.asarray(values, dtype=np.float64)
            zeros = np.zeros_like(values)
            pos, _ = self.pr_to_ul_state(values, zeros)
            return pos
        return self._pr2ul_sidewise(values, self._pos_pr2ul_model)

    def vel_pr2ul(self, values: np.ndarray) -> np.ndarray:
        if self._model_kind == "weighted_mlp":
            raise ValueError(f"{self.path} needs pr_to_ul_state(pr_pos, pr_vel) for velocity conversion")
        if self._vel_pr2ul_model is None:
            raise ValueError(f"{self.path} does not contain a velocity pitch/roll -> upper/lower model")
        return self._pr2ul_sidewise(values, self._vel_pr2ul_model)

    def pos_ul2pr(self, values: np.ndarray) -> np.ndarray:
        if self._model_kind == "weighted_mlp":
            values = np.asarray(values, dtype=np.float64)
            zeros = np.zeros_like(values)
            pos, _ = self.ul_to_pr_state(values, zeros)
            return pos
        if self._pos_ul2pr_model is None:
            raise ValueError(f"{self.path} does not contain a position upper/lower -> pitch/roll model")
        values = np.asarray(values, dtype=np.float64).reshape(-1, 4)
        left_inputs = np.column_stack((values[:, 0], values[:, 1], values[:, 0], values[:, 1]))
        right_inputs = np.column_stack((values[:, 2], values[:, 3], values[:, 2], values[:, 3]))
        left_outputs = self._predict(self._pos_ul2pr_model, left_inputs)
        right_outputs = self._predict(self._pos_ul2pr_model, right_inputs)
        outputs = np.empty((values.shape[0], 4), dtype=np.float64)
        outputs[:, 0:2] = left_outputs[:, 0:2]
        outputs[:, 2:4] = right_outputs[:, 2:4]
        return outputs

    def pr_to_ul_state(self, pr_pos: np.ndarray, pr_vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._model_kind != "weighted_mlp":
            return self.pos_pr2ul(pr_pos), self.vel_pr2ul(pr_vel)
        pr_pos = np.asarray(pr_pos, dtype=np.float64)
        pr_vel = np.asarray(pr_vel, dtype=np.float64)
        values = np.concatenate((pr_pos.reshape(-1, 4), pr_vel.reshape(-1, 4)), axis=-1)
        output = self._predict_weighted("pr_to_ul_state", values, 8).reshape(*pr_pos.shape[:-1], 8)
        return output[..., 0:4], output[..., 4:8]

    def ul_to_pr_state(self, ul_pos: np.ndarray, ul_vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._model_kind != "weighted_mlp":
            return self.pos_ul2pr(ul_pos), np.asarray(ul_vel, dtype=np.float64)
        ul_pos = np.asarray(ul_pos, dtype=np.float64)
        ul_vel = np.asarray(ul_vel, dtype=np.float64)
        values = np.concatenate((ul_pos.reshape(-1, 4), ul_vel.reshape(-1, 4)), axis=-1)
        output = self._predict_weighted("ul_to_pr_state", values, 8).reshape(*ul_pos.shape[:-1], 8)
        return output[..., 0:4], output[..., 4:8]

    def pr_to_ul_torque(self, pr_pos: np.ndarray, pr_vel: np.ndarray, pr_tau: np.ndarray) -> np.ndarray:
        values = np.concatenate(
            (
                np.asarray(pr_pos, dtype=np.float64).reshape(-1, 4),
                np.asarray(pr_vel, dtype=np.float64).reshape(-1, 4),
                np.asarray(pr_tau, dtype=np.float64).reshape(-1, 4),
            ),
            axis=-1,
        )
        return self._predict_weighted("pr_to_ul_torque", values, 4).reshape(np.asarray(pr_tau).shape)

    def ul_to_pr_torque(self, ul_pos: np.ndarray, ul_vel: np.ndarray, ul_tau: np.ndarray) -> np.ndarray:
        values = np.concatenate(
            (
                np.asarray(ul_pos, dtype=np.float64).reshape(-1, 4),
                np.asarray(ul_vel, dtype=np.float64).reshape(-1, 4),
                np.asarray(ul_tau, dtype=np.float64).reshape(-1, 4),
            ),
            axis=-1,
        )
        return self._predict_weighted("ul_to_pr_torque", values, 4).reshape(np.asarray(ul_tau).shape)


def pitch_roll_target_to_upper_lower_command(
    target_pos: np.ndarray,
    cfg: Sim2SimConfig,
    ankle_model: AnkleCommandModel,
) -> np.ndarray:
    observe_index = {name: i for i, name in enumerate(cfg.mujoco_joint_names)}
    ankle_pr = np.array(
        [
            target_pos[observe_index["left_ankle_pitch_joint"]],
            target_pos[observe_index["left_ankle_roll_joint"]],
            target_pos[observe_index["right_ankle_pitch_joint"]],
            target_pos[observe_index["right_ankle_roll_joint"]],
        ],
        dtype=np.float64,
    )
    return ankle_model.pos_pr2ul(ankle_pr.reshape(1, 4))[0]


def limit_real_ankle_command(
    desired_abs: np.ndarray,
    previous_abs: np.ndarray,
    default_abs: np.ndarray,
    cfg: Sim2SimConfig,
) -> np.ndarray:
    limited = desired_abs.copy()
    if cfg.ankle_command_rel_clip is not None:
        rel = np.clip(limited - default_abs, -cfg.ankle_command_rel_clip, cfg.ankle_command_rel_clip)
        limited = default_abs + rel
    if cfg.ankle_command_rate_limit is not None:
        max_delta = cfg.ankle_command_rate_limit * cfg.dt * cfg.decimation
        limited = previous_abs + np.clip(limited - previous_abs, -max_delta, max_delta)
    if cfg.upper_lower_control_clip_min is not None and cfg.upper_lower_control_clip_max is not None:
        ul_ids = upper_lower_control_ids(cfg)
        limited = np.clip(
            limited,
            cfg.upper_lower_control_clip_min[ul_ids],
            cfg.upper_lower_control_clip_max[ul_ids],
        )
    return limited


def apply_upper_lower_command(control_target: np.ndarray, ankle_ul: np.ndarray, cfg: Sim2SimConfig) -> None:
    control_index = {name: i for i, name in enumerate(cfg.control_joint_names)}
    if not all(name in control_index for name in UPPER_LOWER_JOINT_NAMES):
        return
    control_target[
        [
            control_index["left_ankle_upper_joint"],
            control_index["left_ankle_lower_joint"],
            control_index["right_ankle_upper_joint"],
            control_index["right_ankle_lower_joint"],
        ]
    ] = ankle_ul


def clip_policy_target(target_pos: np.ndarray, robot: RobotRuntime, cfg: Sim2SimConfig) -> np.ndarray:
    if cfg.clip_policy_targets_to_joint_limits:
        np.clip(target_pos, robot.observe_target_min, robot.observe_target_max, out=target_pos)
    return target_pos


def clip_control_target(control_target: np.ndarray, robot: RobotRuntime, cfg: Sim2SimConfig) -> np.ndarray:
    if cfg.clip_control_targets_to_joint_limits:
        np.clip(control_target, robot.control_target_min, robot.control_target_max, out=control_target)
    return control_target


def upper_lower_control_ids(cfg: Sim2SimConfig) -> np.ndarray:
    control_index = {name: i for i, name in enumerate(cfg.control_joint_names)}
    if not all(name in control_index for name in UPPER_LOWER_JOINT_NAMES):
        return np.array([], dtype=np.int32)
    return np.array([control_index[name] for name in UPPER_LOWER_JOINT_NAMES], dtype=np.int32)


def policy_ankle_action_ids(cfg: Sim2SimConfig) -> np.ndarray:
    return np.array(
        [
            index
            for index, name in enumerate(cfg.policy_joint_names)
            if POLICY_ANKLE_TO_CONTROL_JOINT.get(name, name) in UPPER_LOWER_JOINT_NAMES
        ],
        dtype=np.int32,
    )


class CommandState:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0
    camera_follow = True
    reset_requested = False

    @classmethod
    def set_initial(cls, command: tuple[float, float, float]) -> None:
        cls.vx, cls.vy, cls.dyaw = command

    @classmethod
    def update(cls, dvx: float = 0.0, dvy: float = 0.0, ddyaw: float = 0.0) -> None:
        cls.vx = float(np.clip(cls.vx + dvx, -0.8, 2.0))
        cls.vy = float(np.clip(cls.vy + dvy, -0.8, 0.8))
        cls.dyaw = float(np.clip(cls.dyaw + ddyaw, -1.5, 1.5))
        print(f"cmd vx={cls.vx:.2f}, vy={cls.vy:.2f}, dyaw={cls.dyaw:.2f}")

    @classmethod
    def zero(cls) -> None:
        cls.vx = cls.vy = cls.dyaw = 0.0


def on_press(key) -> None:
    try:
        char = key.char.lower() if hasattr(key, "char") and key.char is not None else None
        if char == "8":
            CommandState.update(dvx=0.1)
        elif char == "2":
            CommandState.update(dvx=-0.1)
        elif char == "4":
            CommandState.update(dvy=0.1)
        elif char == "6":
            CommandState.update(dvy=-0.1)
        elif char == "7":
            CommandState.update(ddyaw=0.1)
        elif char == "9":
            CommandState.update(ddyaw=-0.1)
        elif char == "f":
            CommandState.camera_follow = not CommandState.camera_follow
            print(f"camera_follow={CommandState.camera_follow}")
        elif char == "0":
            CommandState.reset_requested = True
    except AttributeError:
        if key == keyboard.Key.up:
            CommandState.update(dvx=0.1)
        elif key == keyboard.Key.down:
            CommandState.update(dvx=-0.1)
        elif key == keyboard.Key.left:
            CommandState.update(ddyaw=0.1)
        elif key == keyboard.Key.right:
            CommandState.update(ddyaw=-0.1)


def start_keyboard_listener():
    if keyboard is None:
        return None
    listener = keyboard.Listener(on_press=on_press, on_release=lambda _: None)
    listener.start()
    return listener


def make_index_mapping(source_names: list[str], target_names: list[str]) -> list[int]:
    target_index = {name: i for i, name in enumerate(target_names)}
    mapping = []
    for name in source_names:
        mapped_name = name if name in target_index else UPPER_LOWER_TO_PITCH_ROLL_JOINT.get(name, name)
        mapping.append(target_index[mapped_name])
    return mapping


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


def _match_scale_from_map(joint_name: str, scale_map: dict[str, object]) -> float:
    action_name = POLICY_ANKLE_TO_CONTROL_JOINT.get(joint_name, joint_name)
    for pattern, value in scale_map.items():
        if re.fullmatch(pattern, action_name):
            return float(value)
    for pattern, value in scale_map.items():
        if re.fullmatch(pattern, joint_name):
            return float(value)
    raise ValueError(f"No action scale in params matches joint/action: {joint_name}")


def _training_default_to_mujoco(name: str, value: float, cfg: Sim2SimConfig) -> float:
    # lens110.xml has the right elbow axis flipped relative to the URDF policy/training semantics.
    policy_index = {joint_name: index for index, joint_name in enumerate(cfg.policy_joint_names)}
    if name in policy_index and cfg.policy_joint_signs[policy_index[name]] < 0.0:
        return -value
    return value


def _extract_env_params(
    env_yaml_path: str,
    cfg: Sim2SimConfig,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    float | None,
    str | None,
    str | None,
    dict[str, float | None],
]:
    with open(env_yaml_path, encoding="utf-8") as f:
        lines = f.readlines()

    joint_pos: dict[str, object] | None = None
    scale_map: dict[str, object] | None = None
    action_model_dir: str | None = None
    ankle_pd_space: str | None = None
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
        if in_actions_joint_pos and indent == 4 and stripped.startswith("model_dir:"):
            _, value = stripped.split(":", 1)
            parsed = _parse_scalar(value)
            action_model_dir = None if parsed is None else str(parsed)
        if in_actions_joint_pos and indent == 4 and stripped.startswith("ankle_pd_space:"):
            _, value = stripped.split(":", 1)
            parsed = _parse_scalar(value)
            ankle_pd_space = None if parsed is None else str(parsed)
        if indent <= 2 and in_actions_joint_pos and stripped and not stripped.startswith("joint_pos:"):
            in_actions_joint_pos = False

        idx += 1

    default_pos = None
    if joint_pos:
        default_pos = np.array(
            [
                _training_default_to_mujoco(name, float(joint_pos[name]), cfg)
                for name in cfg.mujoco_joint_names
            ],
            dtype=np.float64,
        )

    action_scale = None
    if scale_map:
        action_scale = np.array([_match_scale_from_map(name, scale_map) for name in cfg.policy_joint_names], dtype=np.float64)

    init_height = float(root_pos[2]) if root_pos and len(root_pos) >= 3 else None
    return default_pos, action_scale, init_height, action_model_dir, ankle_pd_space, _extract_physics_material(lines)


def _extract_agent_clip(agent_yaml_path: str) -> float | None:
    with open(agent_yaml_path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("clip_actions:"):
                _, value = stripped.split(":", 1)
                parsed = _parse_scalar(value)
                return None if parsed is None else float(parsed)
    return None


def load_run_params(params_dir: str | None, cfg: Sim2SimConfig) -> RunParams | None:
    if not params_dir:
        return None
    env_yaml_path = os.path.join(params_dir, "env.yaml")
    agent_yaml_path = os.path.join(params_dir, "agent.yaml")
    if not os.path.isfile(env_yaml_path) or not os.path.isfile(agent_yaml_path):
        raise FileNotFoundError(f"Missing env.yaml or agent.yaml under params dir: {params_dir}")

    default_pos, action_scale, init_height, action_model_dir, ankle_pd_space, physics_material = _extract_env_params(
        env_yaml_path, cfg
    )
    action_clip = _extract_agent_clip(agent_yaml_path)
    return RunParams(
        params_dir=params_dir,
        default_pos=default_pos,
        action_scale=action_scale,
        init_height=init_height,
        action_clip=action_clip,
        policy_ankle_model_path=action_model_dir,
        ankle_pd_space=ankle_pd_space,
        static_friction=physics_material.get("static_friction"),
        dynamic_friction=physics_material.get("dynamic_friction"),
        restitution=physics_material.get("restitution"),
    )


def name_to_id(model: mujoco.MjModel, obj_type, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise ValueError(f"Missing {obj_type.name}: {name}")
    return obj_id


FOOT_GEOM_BODY_FALLBACKS = {
    "left_ankle_pitch_collision": "left_ankle_pitch_link",
    "left_ankle_roll_collision": "left_ankle_roll_link",
    "right_ankle_pitch_collision": "right_ankle_pitch_link",
    "right_ankle_roll_collision": "right_ankle_roll_link",
}


TRAINING_FOOT_BOXES = {
    "left_ankle_roll_collision": ((0.019, 0.009, -0.030), (0.0975, 0.0475, 0.010)),
    "right_ankle_roll_collision": ((0.019, -0.009, -0.030), (0.0975, 0.0475, 0.010)),
    "left_ankle_pitch_collision": ((0.019, 0.009, -0.030), (0.0975, 0.0475, 0.010)),
    "right_ankle_pitch_collision": ((0.019, -0.009, -0.030), (0.0975, 0.0475, 0.010)),
}


def fallback_foot_geom_id(model: mujoco.MjModel, collision_name: str) -> int:
    body_name = FOOT_GEOM_BODY_FALLBACKS.get(collision_name)
    if body_name is None:
        return -1
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return -1

    body_geom_ids = [int(i) for i in np.flatnonzero(model.geom_bodyid == body_id)]
    if not body_geom_ids:
        return -1

    contact_geom_ids = [
        geom_id
        for geom_id in body_geom_ids
        if model.geom_contype[geom_id] != 0 or model.geom_conaffinity[geom_id] != 0
    ]
    if contact_geom_ids:
        return contact_geom_ids[-1]

    non_visual_geom_ids = [geom_id for geom_id in body_geom_ids if model.geom_group[geom_id] != 1]
    if non_visual_geom_ids:
        return non_visual_geom_ids[-1]
    return body_geom_ids[-1]


def foot_collision_geom_id(model: mujoco.MjModel, collision_name: str) -> int:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, collision_name)
    if geom_id >= 0:
        return geom_id
    return fallback_foot_geom_id(model, collision_name)


def collect_foot_geom_ids(model: mujoco.MjModel, foot_contact_geoms: list[str]) -> set[int]:
    return {
        geom_id
        for geom_id in (foot_collision_geom_id(model, name) for name in foot_contact_geoms)
        if geom_id >= 0
    }


def build_joint_data(model: mujoco.MjModel, cfg: Sim2SimConfig):
    observe_joint_ids = np.array(
        [name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in cfg.mujoco_joint_names],
        dtype=np.int32,
    )
    control_joint_ids = np.array(
        [name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in cfg.control_joint_names],
        dtype=np.int32,
    )
    qpos_ids = model.jnt_qposadr[observe_joint_ids].astype(np.int32)
    qvel_ids = model.jnt_dofadr[observe_joint_ids].astype(np.int32)
    control_qpos_ids = model.jnt_qposadr[control_joint_ids].astype(np.int32)
    control_qvel_ids = model.jnt_dofadr[control_joint_ids].astype(np.int32)

    actuator_ids = np.full(len(control_joint_ids), -1, dtype=np.int32)
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        matches = np.flatnonzero(control_joint_ids == joint_id)
        if len(matches):
            actuator_ids[matches[0]] = actuator_id
    if np.any(actuator_ids < 0):
        missing = [name for name, actuator_id in zip(cfg.control_joint_names, actuator_ids) if actuator_id < 0]
        raise ValueError(f"Missing actuators for joints: {missing}")

    observe_target_min = np.full(len(observe_joint_ids), -np.inf, dtype=np.float64)
    observe_target_max = np.full(len(observe_joint_ids), np.inf, dtype=np.float64)
    for i, joint_id in enumerate(observe_joint_ids):
        if model.jnt_limited[joint_id]:
            observe_target_min[i], observe_target_max[i] = model.jnt_range[joint_id]

    control_target_min = np.full(len(control_joint_ids), -np.inf, dtype=np.float64)
    control_target_max = np.full(len(control_joint_ids), np.inf, dtype=np.float64)
    for i, joint_id in enumerate(control_joint_ids):
        if model.jnt_limited[joint_id]:
            control_target_min[i], control_target_max[i] = model.jnt_range[joint_id]

    return (
        qpos_ids,
        qvel_ids,
        control_qpos_ids,
        control_qvel_ids,
        actuator_ids,
        observe_target_min,
        observe_target_max,
        control_target_min,
        control_target_max,
    )


def align_default_pos_to_model_ranges(model: mujoco.MjModel, cfg: Sim2SimConfig, *, verbose: bool = True) -> None:
    """Fix per-XML axis flips when a training default only fits after sign inversion."""
    policy_index = {name: i for i, name in enumerate(cfg.policy_joint_names)}
    adjusted: list[str] = []
    eps = 1e-6
    for mujoco_id, joint_name in enumerate(cfg.mujoco_joint_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0 or not model.jnt_limited[joint_id]:
            continue
        lower, upper = model.jnt_range[joint_id]
        value = float(cfg.default_pos[mujoco_id])
        if lower - eps <= value <= upper + eps:
            continue
        flipped = -value
        if lower - eps <= flipped <= upper + eps:
            cfg.default_pos[mujoco_id] = flipped
            if joint_name in policy_index:
                cfg.policy_joint_signs[policy_index[joint_name]] *= -1.0
            adjusted.append(joint_name)
    if adjusted and verbose:
        print("[INFO] default/sign auto-aligned to XML joint ranges: " + ", ".join(adjusted))


def build_passive_joint_defaults(model: mujoco.MjModel, cfg: Sim2SimConfig) -> tuple[np.ndarray, np.ndarray]:
    qpos_ids = []
    values = []
    for joint_name, default_value in cfg.passive_joint_default_pos.items():
        joint_id = name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qpos_ids.append(int(model.jnt_qposadr[joint_id]))
        values.append(float(default_value))
    return np.asarray(qpos_ids, dtype=np.int32), np.asarray(values, dtype=np.float64)


def write_passive_default_qpos(data: mujoco.MjData, robot: RobotRuntime) -> None:
    if robot.passive_default_qpos_ids is None or robot.passive_default_values is None:
        return
    if len(robot.passive_default_qpos_ids) == 0:
        return
    data.qpos[robot.passive_default_qpos_ids] = robot.passive_default_values


def geom_min_z(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float:
    if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
        sx, sy, sz = model.geom_size[geom_id]
        corners = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)], dtype=np.float64)
        points = data.geom_xpos[geom_id] + corners @ data.geom_xmat[geom_id].reshape(3, 3).T
        return float(points[:, 2].min())
    return float(data.geom_xpos[geom_id, 2] - model.geom_rbound[geom_id])


def grounded_base_height(
    model: mujoco.MjModel, data: mujoco.MjData, robot: RobotRuntime, cfg: Sim2SimConfig, clearance: float = 0.002
) -> float:
    saved_qpos = data.qpos.copy()
    saved_qvel = data.qvel.copy()
    saved_ctrl = data.ctrl.copy()

    data.qpos[:3] = robot.base_pos
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[robot.qpos_ids] = getattr(robot, "stand_pos", robot.default_pos)
    write_passive_default_qpos(data, robot)
    write_control_only_qpos(data, robot, cfg)
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.ctrl[robot.actuator_ids] = robot.control_default_pos
    mujoco.mj_forward(model, data)

    min_z = [geom_min_z(model, data, geom_id) for geom_id in collect_foot_geom_ids(model, cfg.foot_contact_geoms)]

    data.qpos[:], data.qvel[:], data.ctrl[:] = saved_qpos, saved_qvel, saved_ctrl
    mujoco.mj_forward(model, data)
    return float(robot.base_pos[2] if not min_z else robot.base_pos[2] - min(min_z) + clearance)


def patch_training_foot_collision_boxes(model: mujoco.MjModel) -> None:
    patched = 0
    for geom_name, (geom_pos, geom_size) in TRAINING_FOOT_BOXES.items():
        geom_id = foot_collision_geom_id(model, geom_name)
        if geom_id < 0:
            continue
        model.geom_type[geom_id] = mujoco.mjtGeom.mjGEOM_BOX
        model.geom_pos[geom_id] = geom_pos
        model.geom_size[geom_id] = geom_size
        model.geom_rbound[geom_id] = np.linalg.norm(geom_size)
        model.geom_condim[geom_id] = 4
        model.geom_friction[geom_id] = [1.0, 0.2, 0.2]
        model.geom_contype[geom_id] = 1
        model.geom_conaffinity[geom_id] = 1
        patched += 1
    if patched:
        print(f"[INFO] patched {patched} foot collision geoms to training box shapes")


def patch_training_passive_ankle_limits(model: mujoco.MjModel) -> None:
    # The passive pitch/roll ankle joints in lens110.xml already use the URDF
    # training limits. Do not widen them here: upper/lower policies can output
    # large motor actions, and widening these passive limits lets the converted
    # pitch/roll targets leave the range seen by the training plant.
    limits = {}
    patched = 0
    for joint_name, joint_range in limits.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            continue
        model.jnt_limited[joint_id] = 1
        model.jnt_range[joint_id] = joint_range
        patched += 1
    if patched:
        print(f"[INFO] patched {patched} joint limits to training ranges")


def patch_model(
    model: mujoco.MjModel,
    robot: RobotRuntime,
    cfg: Sim2SimConfig,
    torque_control_ids: set[int] | None = None,
) -> None:
    torque_control_ids = torque_control_ids or set()
    if cfg.use_training_foot_collision_boxes:
        patch_training_foot_collision_boxes(model)
    foot_geom_ids = collect_foot_geom_ids(model, cfg.foot_contact_geoms)
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if geom_name in GROUND_GEOM_NAMES or geom_id in foot_geom_ids:
            model.geom_contype[geom_id] = 1
            model.geom_conaffinity[geom_id] = 1
        else:
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0
    active_ground_names = [
        name for name in sorted(GROUND_GEOM_NAMES) if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0
    ]
    print(f"[INFO] active contacts: {active_ground_names or ['<missing-ground>']} + {len(foot_geom_ids)} foot geoms")

    if cfg.tendon_stiffness is not None or cfg.tendon_damping is not None or cfg.tendon_frictionloss is not None:
        original_stiffness = model.tendon_stiffness.copy()
        original_damping = model.tendon_damping.copy()
        original_frictionloss = model.tendon_frictionloss.copy()
        if cfg.tendon_stiffness is not None:
            model.tendon_stiffness[:] = cfg.tendon_stiffness
        if cfg.tendon_damping is not None:
            model.tendon_damping[:] = cfg.tendon_damping
        if cfg.tendon_frictionloss is not None:
            model.tendon_frictionloss[:] = cfg.tendon_frictionloss
        print(
            "[INFO] tendon params "
            f"stiffness {np.array2string(original_stiffness, precision=3)} -> "
            f"{np.array2string(model.tendon_stiffness, precision=3)}, "
            f"damping {np.array2string(original_damping, precision=3)} -> "
            f"{np.array2string(model.tendon_damping, precision=3)}, "
            f"frictionloss {np.array2string(original_frictionloss, precision=3)} -> "
            f"{np.array2string(model.tendon_frictionloss, precision=3)}"
        )

    if robot.static_friction is not None:
        ground_name = next(
            (name for name in sorted(GROUND_GEOM_NAMES) if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0),
            None,
        )
        if ground_name is not None:
            ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, ground_name)
            original = model.geom_friction[ground_id].copy()
            slide = robot.dynamic_friction if robot.dynamic_friction is not None else robot.static_friction
            model.geom_condim[ground_id] = 4
            model.geom_margin[ground_id] = 0.001
            model.geom_friction[ground_id] = [slide, 0.2, 0.2]
            print(
                f"[INFO] {ground_name} friction "
                f"{np.array2string(original, precision=3)} -> "
                f"{np.array2string(model.geom_friction[ground_id], precision=3)}"
            )

    model.dof_damping[:] = 0.0
    model.dof_armature[robot.control_qvel_ids] = cfg.armature
    scaled_ankle_motor_ids = []
    for mujoco_id, actuator_id in enumerate(robot.actuator_ids):
        kp = float(cfg.kp[mujoco_id])
        kd = float(cfg.kd[mujoco_id])
        effort = float(cfg.tau_limit[mujoco_id])
        if cfg.control_joint_names[mujoco_id] in UPPER_LOWER_JOINT_NAMES:
            kp *= cfg.ankle_motor_kp_scale
            kd *= cfg.ankle_motor_kd_scale
            effort *= cfg.ankle_motor_tau_scale
            scaled_ankle_motor_ids.append(mujoco_id)
        if mujoco_id in torque_control_ids:
            model.actuator_gaintype[actuator_id] = mujoco.mjtGain.mjGAIN_FIXED
            model.actuator_biastype[actuator_id] = mujoco.mjtBias.mjBIAS_NONE
            model.actuator_gainprm[actuator_id, :] = 0.0
            model.actuator_gainprm[actuator_id, 0] = 1.0
            model.actuator_biasprm[actuator_id, :] = 0.0
            model.actuator_ctrllimited[actuator_id] = 1
            model.actuator_ctrlrange[actuator_id] = [-effort, effort]
            model.actuator_forcelimited[actuator_id] = 1
            model.actuator_forcerange[actuator_id] = [-effort, effort]
            continue
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
    if torque_control_ids:
        if len(torque_control_ids) >= len(robot.actuator_ids):
            print("[INFO] motor XML actuators kept as torque motors for explicit PD")
        else:
            print("[INFO] non-ankle motors patched to position-PD; ankle upper/lower kept as torque motors")
    else:
        print("[INFO] motor XML actuators patched to MuJoCo internal position-PD")
    if scaled_ankle_motor_ids:
        print(
            "[INFO] ankle motor scales: "
            f"kp={cfg.ankle_motor_kp_scale:g}, "
            f"kd={cfg.ankle_motor_kd_scale:g}, "
            f"tau={cfg.ankle_motor_tau_scale:g}"
        )


def write_control_only_qpos(data: mujoco.MjData, robot: RobotRuntime, cfg: Sim2SimConfig) -> None:
    control_index = {name: i for i, name in enumerate(cfg.control_joint_names)}
    control_default_pos = getattr(robot, "stand_control_pos", robot.control_default_pos)
    for name in UPPER_LOWER_JOINT_NAMES:
        if name not in control_index:
            continue
        control_id = control_index[name]
        data.qpos[robot.control_qpos_ids[control_id]] = control_default_pos[control_id]


def hold_default_stand(data: mujoco.MjData, robot: RobotRuntime, cfg: Sim2SimConfig) -> None:
    stand_pos = getattr(robot, "stand_pos", robot.default_pos)
    stand_control_pos = getattr(robot, "stand_control_pos", robot.control_default_pos)
    data.qpos[:3] = robot.base_pos
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[robot.qpos_ids] = stand_pos
    write_passive_default_qpos(data, robot)
    write_control_only_qpos(data, robot, cfg)
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.ctrl[robot.actuator_ids] = stand_control_pos


def get_base_obs(data: mujoco.MjData, use_root_ang_vel: bool = False, use_sensor_orientation: bool = False):
    qpos = data.qpos.astype(np.float64)
    qvel = data.qvel.astype(np.float64)
    quat_wxyz = data.sensor("orientation").data.astype(np.float64) if use_sensor_orientation else qpos[3:7]
    rot = R.from_quat(quat_wxyz[[1, 2, 3, 0]])
    lin_vel_b = rot.apply(qvel[:3], inverse=True).astype(np.float64)
    ang_vel_b = (
        rot.apply(qvel[3:6], inverse=True).astype(np.float64)
        if use_root_ang_vel
        else data.sensor("angular-velocity").data.astype(np.float64)
    )
    gravity_b = rot.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.float64)
    return qpos, qvel, lin_vel_b, ang_vel_b, gravity_b


def make_observation(
    q: np.ndarray,
    dq: np.ndarray,
    control_q: np.ndarray,
    control_dq: np.ndarray,
    omega: np.ndarray,
    gravity: np.ndarray,
    action: np.ndarray,
    robot: RobotRuntime,
    cfg: Sim2SimConfig,
    policy_ankle_model: AnkleCommandModel | None = None,
):
    q_obs = np.zeros(cfg.num_actions, dtype=np.float64)
    dq_obs = np.zeros(cfg.num_actions, dtype=np.float64)
    if cfg.policy_io == "upper_lower":
        control_index = {name: i for i, name in enumerate(cfg.control_joint_names)}
        mujoco_index = {name: i for i, name in enumerate(cfg.mujoco_joint_names)}
        if cfg.upper_lower_observation_source == "converted_pitch_roll":
            if policy_ankle_model is None:
                raise ValueError("upper_lower converted_pitch_roll observations require a policy ankle model")
            pr_names = (
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
            )
            pr_ids = np.array([mujoco_index[name] for name in pr_names], dtype=np.int32)
            ankle_ul_pos, ankle_ul_vel_abs = policy_ankle_model.pr_to_ul_state(
                q[pr_ids].reshape(1, 4),
                dq[pr_ids].reshape(1, 4),
            )
            default_ul_pos, default_ul_vel = policy_ankle_model.pr_to_ul_state(
                robot.default_pos[pr_ids].reshape(1, 4),
                np.zeros((1, 4), dtype=np.float64),
            )
            ankle_ul_rel = ankle_ul_pos[0] - default_ul_pos[0]
            ankle_ul_vel = ankle_ul_vel_abs[0] - default_ul_vel[0]
        for policy_id, policy_name in enumerate(cfg.policy_joint_names):
            control_name = POLICY_ANKLE_TO_CONTROL_JOINT.get(policy_name)
            if control_name is not None and control_name in UPPER_LOWER_JOINT_NAMES:
                if cfg.upper_lower_observation_source == "converted_pitch_roll":
                    ankle_id = UPPER_LOWER_JOINT_NAMES.index(control_name)
                    q_obs[policy_id] = ankle_ul_rel[ankle_id]
                    dq_obs[policy_id] = ankle_ul_vel[ankle_id]
                elif control_name in control_index:
                    control_id = control_index[control_name]
                    q_obs[policy_id] = control_q[control_id] - robot.control_default_pos[control_id]
                    dq_obs[policy_id] = control_dq[control_id]
                else:
                    raise ValueError(
                        "upper_lower_observation_source='control' requires upper/lower control joints "
                        f"but '{control_name}' is not present in control_joint_names."
                    )
            else:
                mujoco_id = mujoco_index[policy_name]
                sign = robot.policy_joint_signs[policy_id]
                q_obs[policy_id] = sign * (q[mujoco_id] - robot.default_pos[mujoco_id])
                dq_obs[policy_id] = sign * dq[mujoco_id]
    else:
        joint_pos_rel = q - robot.default_pos
        for policy_id, mujoco_id in enumerate(robot.policy_to_mujoco):
            sign = robot.policy_joint_signs[policy_id]
            q_obs[policy_id] = sign * joint_pos_rel[mujoco_id]
            dq_obs[policy_id] = sign * dq[mujoco_id]

    obs = np.zeros((1, cfg.num_obs), dtype=np.float32)
    obs[0, 0:3] = omega
    obs[0, 3:6] = gravity
    obs[0, 6:9] = [CommandState.vx, CommandState.vy, CommandState.dyaw]
    obs[0, 9 : 9 + cfg.num_actions] = q_obs
    obs[0, 9 + cfg.num_actions : 9 + 2 * cfg.num_actions] = dq_obs
    obs[0, 9 + 2 * cfg.num_actions : 9 + 3 * cfg.num_actions] = action
    return obs


def ankle_pitch_roll_to_upper_lower(side: str, pitch: float, roll: float) -> tuple[float, float]:
    """Local inverse map from trained ankle pitch/roll target to linkage motors."""
    if side == "left":
        upper = -0.000143 - 1.01599 * pitch + 0.83328 * roll
        lower = 0.000118 - 1.02322 * pitch - 0.83019 * roll
    elif side == "right":
        upper = -0.000143 - 1.01599 * pitch - 0.83328 * roll
        lower = 0.000118 - 1.02322 * pitch + 0.83019 * roll
    else:
        raise ValueError(side)
    return upper, lower


def observe_to_control_target(observe_target: np.ndarray, robot: RobotRuntime, cfg: Sim2SimConfig) -> np.ndarray:
    control_target = np.zeros(cfg.num_actions, dtype=np.float64)
    observe_index = {name: i for i, name in enumerate(cfg.mujoco_joint_names)}
    control_index = {name: i for i, name in enumerate(cfg.control_joint_names)}
    for control_id, observe_id in enumerate(robot.control_to_mujoco):
        if observe_id >= 0:
            control_target[control_id] = observe_target[observe_id]

    if not all(name in control_index for name in UPPER_LOWER_JOINT_NAMES):
        return clip_control_target(control_target, robot, cfg)
    if all(name in observe_index for name in UPPER_LOWER_JOINT_NAMES):
        return clip_control_target(control_target, robot, cfg)

    pr_names = (
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    )
    if not all(name in observe_index for name in pr_names):
        raise ValueError(
            "Cannot map observe targets to upper/lower controls: observation joints contain neither "
            "upper/lower ankles nor pitch/roll ankles."
        )

    left_upper, left_lower = ankle_pitch_roll_to_upper_lower(
        "left",
        float(observe_target[observe_index["left_ankle_pitch_joint"]]),
        float(observe_target[observe_index["left_ankle_roll_joint"]]),
    )
    right_upper, right_lower = ankle_pitch_roll_to_upper_lower(
        "right",
        float(observe_target[observe_index["right_ankle_pitch_joint"]]),
        float(observe_target[observe_index["right_ankle_roll_joint"]]),
    )

    control_target[control_index["left_ankle_upper_joint"]] = left_upper
    control_target[control_index["left_ankle_lower_joint"]] = left_lower
    control_target[control_index["right_ankle_upper_joint"]] = right_upper
    control_target[control_index["right_ankle_lower_joint"]] = right_lower
    return clip_control_target(control_target, robot, cfg)


def observe_to_control_default(observe_default: np.ndarray, cfg: Sim2SimConfig) -> np.ndarray:
    control_default = np.zeros(cfg.num_actions, dtype=np.float64)
    observe_index = {name: i for i, name in enumerate(cfg.mujoco_joint_names)}
    control_index = {name: i for i, name in enumerate(cfg.control_joint_names)}
    for control_id, name in enumerate(cfg.control_joint_names):
        if name in observe_index:
            control_default[control_id] = observe_default[observe_index[name]]

    if not all(name in control_index for name in UPPER_LOWER_JOINT_NAMES):
        return control_default
    if all(name in observe_index for name in UPPER_LOWER_JOINT_NAMES):
        return control_default

    pr_names = (
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    )
    if not all(name in observe_index for name in pr_names):
        raise ValueError(
            "Cannot map observe defaults to upper/lower controls: observation joints contain neither "
            "upper/lower ankles nor pitch/roll ankles."
        )

    left_upper, left_lower = ankle_pitch_roll_to_upper_lower(
        "left",
        float(observe_default[observe_index["left_ankle_pitch_joint"]]),
        float(observe_default[observe_index["left_ankle_roll_joint"]]),
    )
    right_upper, right_lower = ankle_pitch_roll_to_upper_lower(
        "right",
        float(observe_default[observe_index["right_ankle_pitch_joint"]]),
        float(observe_default[observe_index["right_ankle_roll_joint"]]),
    )
    control_default[control_index["left_ankle_upper_joint"]] = left_upper
    control_default[control_index["left_ankle_lower_joint"]] = left_lower
    control_default[control_index["right_ankle_upper_joint"]] = right_upper
    control_default[control_index["right_ankle_lower_joint"]] = right_lower
    return control_default


class PolynomialAnkleMapper:
    def __init__(self, model_path: str, robot: RobotRuntime, cfg: Sim2SimConfig):
        self.model = AnkleCommandModel(model_path)
        self.robot = robot
        self.cfg = cfg
        self.observe_index = {name: i for i, name in enumerate(cfg.mujoco_joint_names)}
        self.control_index = {name: i for i, name in enumerate(cfg.control_joint_names)}

    def observe_to_control_target(self, observe_target: np.ndarray) -> np.ndarray:
        control_target = np.zeros(self.cfg.num_actions, dtype=np.float64)
        for control_id, observe_id in enumerate(self.robot.control_to_mujoco):
            if observe_id >= 0:
                control_target[control_id] = observe_target[observe_id]

        ankle_pr = np.array(
            [
                observe_target[self.observe_index["left_ankle_pitch_joint"]],
                observe_target[self.observe_index["left_ankle_roll_joint"]],
                observe_target[self.observe_index["right_ankle_pitch_joint"]],
                observe_target[self.observe_index["right_ankle_roll_joint"]],
            ],
            dtype=np.float64,
        )
        ankle_ul = self.model.pos_pr2ul(ankle_pr.reshape(1, 4))[0]
        control_target[
            [
                self.control_index["left_ankle_upper_joint"],
                self.control_index["left_ankle_lower_joint"],
                self.control_index["right_ankle_upper_joint"],
                self.control_index["right_ankle_lower_joint"],
            ]
        ] = ankle_ul
        return clip_control_target(control_target, self.robot, self.cfg)


class IdentityControlMapper:
    def __init__(self, robot: RobotRuntime, cfg: Sim2SimConfig):
        self.robot = robot
        self.cfg = cfg

    def observe_to_control_target(self, observe_target: np.ndarray) -> np.ndarray:
        control_target = observe_to_control_target(observe_target, self.robot, self.cfg)
        return clip_control_target(control_target, self.robot, self.cfg)


class ExactAnkleMapper:
    def __init__(self, model: mujoco.MjModel, robot: RobotRuntime, cfg: Sim2SimConfig):
        if model.ntendon < 4:
            raise ValueError("upper/lower ankle mapping expects the four Lens110 spatial tendons")
        self.model = model
        self.data = mujoco.MjData(model)
        self.robot = robot
        self.cfg = cfg
        self.observe_index = {name: i for i, name in enumerate(cfg.mujoco_joint_names)}
        self.control_index = {name: i for i, name in enumerate(cfg.control_joint_names)}
        self.qpos_template = np.zeros(model.nq, dtype=np.float64)
        self.qpos_template[:3] = [0.0, 0.0, 0.68]
        self.qpos_template[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.qpos_template[robot.qpos_ids] = robot.default_pos
        for name in UPPER_LOWER_JOINT_NAMES:
            if name in self.control_index:
                control_id = self.control_index[name]
                self.qpos_template[robot.control_qpos_ids[control_id]] = robot.control_default_pos[control_id]
        self.tendon_ids = {"left": (0, 1), "right": (2, 3)}
        self.target_lengths = model.tendon_range[:, 0].copy()
        self.previous = {
            "left": robot.control_default_pos[
                [self.control_index["left_ankle_upper_joint"], self.control_index["left_ankle_lower_joint"]]
            ].copy(),
            "right": robot.control_default_pos[
                [self.control_index["right_ankle_upper_joint"], self.control_index["right_ankle_lower_joint"]]
            ].copy(),
        }

    def solve_side(self, side: str, pitch: float, roll: float) -> np.ndarray:
        if side == "left":
            pitch_name, roll_name = "left_ankle_pitch_joint", "left_ankle_roll_joint"
            upper_name, lower_name = "left_ankle_upper_joint", "left_ankle_lower_joint"
        else:
            pitch_name, roll_name = "right_ankle_pitch_joint", "right_ankle_roll_joint"
            upper_name, lower_name = "right_ankle_upper_joint", "right_ankle_lower_joint"

        pitch_qpos = self.robot.qpos_ids[self.observe_index[pitch_name]]
        roll_qpos = self.robot.qpos_ids[self.observe_index[roll_name]]
        upper_qpos = self.robot.control_qpos_ids[self.control_index[upper_name]]
        lower_qpos = self.robot.control_qpos_ids[self.control_index[lower_name]]
        tendon_ids = self.tendon_ids[side]
        target_lengths = self.target_lengths[list(tendon_ids)]

        def residual(x: np.ndarray) -> np.ndarray:
            self.data.qpos[:] = self.qpos_template
            self.data.qpos[pitch_qpos] = pitch
            self.data.qpos[roll_qpos] = roll
            self.data.qpos[upper_qpos] = x[0]
            self.data.qpos[lower_qpos] = x[1]
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
            return self.data.ten_length[list(tendon_ids)] - target_lengths

        x0 = self.previous[side]
        if np.linalg.norm(residual(x0), ord=np.inf) > 1e-4:
            x0 = np.array(ankle_pitch_roll_to_upper_lower(side, pitch, roll), dtype=np.float64)

        upper_id = self.control_index[upper_name]
        lower_id = self.control_index[lower_name]
        lower_bound = self.robot.control_target_min[[upper_id, lower_id]]
        upper_bound = self.robot.control_target_max[[upper_id, lower_id]]
        x0 = np.clip(x0, lower_bound, upper_bound)
        result = least_squares(
            residual,
            x0,
            bounds=(lower_bound, upper_bound),
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            max_nfev=20,
        )
        if result.cost < 1e-12:
            self.previous[side] = result.x
        else:
            self.previous[side] = x0
        return self.previous[side].copy()

    def jacobian(self, side: str, pitch: float, roll: float, eps: float = 1e-5) -> np.ndarray:
        saved_previous = self.previous[side].copy()
        base = self.solve_side(side, pitch, roll)
        self.previous[side] = base
        upper_pitch = self.solve_side(side, pitch + eps, roll)
        self.previous[side] = base
        lower_pitch = self.solve_side(side, pitch - eps, roll)
        self.previous[side] = base
        upper_roll = self.solve_side(side, pitch, roll + eps)
        self.previous[side] = base
        lower_roll = self.solve_side(side, pitch, roll - eps)
        self.previous[side] = saved_previous
        return np.column_stack(((upper_pitch - lower_pitch) / (2.0 * eps), (upper_roll - lower_roll) / (2.0 * eps)))

    def observe_to_control_target(self, observe_target: np.ndarray) -> np.ndarray:
        control_target = observe_to_control_target(observe_target, self.robot, self.cfg)
        left = self.solve_side(
            "left",
            float(observe_target[self.observe_index["left_ankle_pitch_joint"]]),
            float(observe_target[self.observe_index["left_ankle_roll_joint"]]),
        )
        right = self.solve_side(
            "right",
            float(observe_target[self.observe_index["right_ankle_pitch_joint"]]),
            float(observe_target[self.observe_index["right_ankle_roll_joint"]]),
        )
        control_target[
            [self.control_index["left_ankle_upper_joint"], self.control_index["left_ankle_lower_joint"]]
        ] = left
        control_target[
            [self.control_index["right_ankle_upper_joint"], self.control_index["right_ankle_lower_joint"]]
        ] = right
        return clip_control_target(control_target, self.robot, self.cfg)


def action_to_target(action: np.ndarray, robot: RobotRuntime, cfg: Sim2SimConfig) -> np.ndarray:
    target_pos = robot.default_pos.copy()
    target_delta_policy = action * robot.action_scale * robot.policy_joint_signs
    for policy_id, mujoco_id in enumerate(robot.policy_to_mujoco):
        target_pos[mujoco_id] += target_delta_policy[policy_id]
    return clip_policy_target(target_pos, robot, cfg)


def policy_default_upper_lower(
    robot: RobotRuntime,
    cfg: Sim2SimConfig,
    policy_ankle_model: AnkleCommandModel,
) -> np.ndarray:
    observe_index = {name: i for i, name in enumerate(cfg.mujoco_joint_names)}
    default_pr = robot.default_pos[
        [
            observe_index["left_ankle_pitch_joint"],
            observe_index["left_ankle_roll_joint"],
            observe_index["right_ankle_pitch_joint"],
            observe_index["right_ankle_roll_joint"],
        ]
    ]
    return policy_ankle_model.pos_pr2ul(default_pr.reshape(1, 4))[0]


def upper_lower_action_to_control_target(
    action: np.ndarray,
    robot: RobotRuntime,
    cfg: Sim2SimConfig,
    policy_ankle_model: AnkleCommandModel | None = None,
) -> np.ndarray:
    control_index = {name: i for i, name in enumerate(cfg.control_joint_names)}
    control_target = robot.control_default_pos.copy()
    policy_default_ul = (
        policy_default_upper_lower(robot, cfg, policy_ankle_model) if policy_ankle_model is not None else None
    )
    for policy_id, policy_name in enumerate(cfg.policy_joint_names):
        control_name = POLICY_ANKLE_TO_CONTROL_JOINT.get(policy_name, policy_name)
        if control_name not in control_index:
            continue
        control_id = control_index[control_name]
        default = robot.control_default_pos[control_id]
        if policy_default_ul is not None and control_name in UPPER_LOWER_JOINT_NAMES:
            default = policy_default_ul[UPPER_LOWER_JOINT_NAMES.index(control_name)]
        value = default + action[policy_id] * robot.action_scale[policy_id] * robot.policy_joint_signs[policy_id]
        if cfg.upper_lower_action_clip_min is not None and cfg.upper_lower_action_clip_max is not None:
            value = float(np.clip(value, cfg.upper_lower_action_clip_min[policy_id], cfg.upper_lower_action_clip_max[policy_id]))
        control_target[control_id] = value
    return clip_control_target(control_target, robot, cfg)


def upper_lower_policy_action_to_pitch_roll_target(
    action: np.ndarray,
    robot: RobotRuntime,
    cfg: Sim2SimConfig,
    policy_ankle_model: AnkleCommandModel,
) -> np.ndarray:
    target_pos = action_to_target(action, robot, cfg)
    control_index = {name: i for i, name in enumerate(cfg.control_joint_names)}
    observe_index = {name: i for i, name in enumerate(cfg.mujoco_joint_names)}
    default_ul = policy_default_upper_lower(robot, cfg, policy_ankle_model)
    ankle_ul = default_ul.copy()
    for policy_id, policy_name in enumerate(cfg.policy_joint_names):
        control_name = POLICY_ANKLE_TO_CONTROL_JOINT.get(policy_name)
        if control_name not in UPPER_LOWER_JOINT_NAMES:
            continue
        ankle_id = UPPER_LOWER_JOINT_NAMES.index(control_name)
        value = default_ul[ankle_id] + action[policy_id] * robot.action_scale[policy_id] * robot.policy_joint_signs[policy_id]
        if cfg.upper_lower_action_clip_min is not None and cfg.upper_lower_action_clip_max is not None:
            value = float(np.clip(value, cfg.upper_lower_action_clip_min[policy_id], cfg.upper_lower_action_clip_max[policy_id]))
        ankle_ul[ankle_id] = value
    ankle_pr = policy_ankle_model.pos_ul2pr(ankle_ul.reshape(1, 4))[0]
    default_pr = robot.default_pos[
        [
            observe_index["left_ankle_pitch_joint"],
            observe_index["left_ankle_roll_joint"],
            observe_index["right_ankle_pitch_joint"],
            observe_index["right_ankle_roll_joint"],
        ]
    ]
    ankle_pr += default_pr - policy_ankle_model.pos_ul2pr(default_ul.reshape(1, 4))[0]
    target_pos[
        [
            observe_index["left_ankle_pitch_joint"],
            observe_index["left_ankle_roll_joint"],
            observe_index["right_ankle_pitch_joint"],
            observe_index["right_ankle_roll_joint"],
        ]
    ] = ankle_pr
    return clip_policy_target(target_pos, robot, cfg)


def ankle_torque_control(
    side: str,
    q: np.ndarray,
    dq: np.ndarray,
    target_pos: np.ndarray,
    robot: RobotRuntime,
    cfg: Sim2SimConfig,
    ankle_mapper: ExactAnkleMapper,
) -> np.ndarray:
    observe_index = {name: i for i, name in enumerate(cfg.mujoco_joint_names)}
    control_index = {name: i for i, name in enumerate(cfg.control_joint_names)}
    if side == "left":
        pitch_name, roll_name = "left_ankle_pitch_joint", "left_ankle_roll_joint"
        upper_name, lower_name = "left_ankle_upper_joint", "left_ankle_lower_joint"
    else:
        pitch_name, roll_name = "right_ankle_pitch_joint", "right_ankle_roll_joint"
        upper_name, lower_name = "right_ankle_upper_joint", "right_ankle_lower_joint"

    task_ids = np.array([observe_index[pitch_name], observe_index[roll_name]], dtype=np.int32)
    motor_ids = np.array([control_index[upper_name], control_index[lower_name]], dtype=np.int32)
    tau_task = cfg.kp[task_ids] * (target_pos[task_ids] - q[task_ids]) - cfg.kd[task_ids] * dq[task_ids]
    tau_task = np.clip(tau_task, -cfg.tau_limit[task_ids], cfg.tau_limit[task_ids])
    jacobian = ankle_mapper.jacobian(side, float(q[task_ids[0]]), float(q[task_ids[1]]))
    try:
        tau_motor = np.linalg.solve(jacobian.T, tau_task)
    except np.linalg.LinAlgError:
        tau_motor = np.linalg.pinv(jacobian.T) @ tau_task
    return np.clip(tau_motor, -cfg.tau_limit[motor_ids], cfg.tau_limit[motor_ids])


def ankle_torque_control_from_tendon_jacobian(
    side: str,
    data: mujoco.MjData,
    q: np.ndarray,
    dq: np.ndarray,
    target_pos: np.ndarray,
    robot: RobotRuntime,
    cfg: Sim2SimConfig,
) -> np.ndarray:
    observe_index = {name: i for i, name in enumerate(cfg.mujoco_joint_names)}
    control_index = {name: i for i, name in enumerate(cfg.control_joint_names)}
    if side == "left":
        pitch_name, roll_name = "left_ankle_pitch_joint", "left_ankle_roll_joint"
        upper_name, lower_name = "left_ankle_upper_joint", "left_ankle_lower_joint"
        tendon_rows = np.array([0, 1], dtype=np.int32)
    else:
        pitch_name, roll_name = "right_ankle_pitch_joint", "right_ankle_roll_joint"
        upper_name, lower_name = "right_ankle_upper_joint", "right_ankle_lower_joint"
        tendon_rows = np.array([2, 3], dtype=np.int32)

    task_ids = np.array([observe_index[pitch_name], observe_index[roll_name]], dtype=np.int32)
    motor_ids = np.array([control_index[upper_name], control_index[lower_name]], dtype=np.int32)
    tau_task = cfg.kp[task_ids] * (target_pos[task_ids] - q[task_ids]) - cfg.kd[task_ids] * dq[task_ids]
    tau_task = np.clip(tau_task, -cfg.tau_limit[task_ids], cfg.tau_limit[task_ids])

    task_dofs = robot.qvel_ids[task_ids]
    motor_dofs = robot.control_qvel_ids[motor_ids]
    tendon_task = data.ten_J[np.ix_(tendon_rows, task_dofs)]
    tendon_motor = data.ten_J[np.ix_(tendon_rows, motor_dofs)]
    try:
        velocity_jacobian = -np.linalg.solve(tendon_motor, tendon_task)
        tau_motor = np.linalg.solve(velocity_jacobian.T, tau_task)
    except np.linalg.LinAlgError:
        velocity_jacobian = -np.linalg.pinv(tendon_motor) @ tendon_task
        tau_motor = np.linalg.pinv(velocity_jacobian.T) @ tau_task
    return np.clip(tau_motor, -cfg.tau_limit[motor_ids], cfg.tau_limit[motor_ids])


def init_render(model: mujoco.MjModel, data: mujoco.MjData, args: argparse.Namespace, cfg: Sim2SimConfig):
    if args.headless:
        if cv2 is None:
            raise ImportError("Video rendering requires OpenCV; install opencv-python in the active local environment.")
        model.vis.global_.offwidth = args.viewer_width
        model.vis.global_.offheight = args.viewer_height
        renderer = mujoco.Renderer(model, width=args.viewer_width, height=args.viewer_height)
        camera = mujoco.MjvCamera()
        camera.distance = 4.0
        camera.azimuth = 45.0
        camera.elevation = -20.0
        camera.lookat = [0.0, 0.0, 1.0]
        writer = cv2.VideoWriter(
            args.output,
            cv2.VideoWriter_fourcc(*"mp4v"),
            1.0 / cfg.dt / cfg.decimation,
            (args.viewer_width, args.viewer_height),
        )
        return renderer, camera, writer, None

    if mujoco_viewer is None:
        raise ImportError("Window rendering requires mujoco-python-viewer in the active local environment.")
    viewer = mujoco_viewer.MujocoViewer(
        model,
        data,
        mode="window",
        width=args.viewer_width,
        height=args.viewer_height,
        hide_menus=not args.show_menus,
    )
    viewer.cam.distance = 4.0
    viewer.cam.azimuth = 45.0
    viewer.cam.elevation = -20.0
    viewer.cam.lookat = [0.0, 0.0, 1.0]
    return None, None, None, viewer


def render(data: mujoco.MjData, renderer, camera, writer, viewer, headless: bool) -> bool:
    if CommandState.camera_follow:
        base_pos = [float(x) for x in data.qpos[:3]]
        if headless:
            camera.lookat = base_pos
        else:
            viewer.cam.lookat = base_pos
    if headless:
        renderer.update_scene(data, camera=camera)
        writer.write(renderer.render())
        return True
    try:
        viewer.render()
    except Exception as exc:
        if "GLFW window does not exist" in str(exc):
            return False
        raise
    return True


def summarize_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, int]:
    counts: dict[str, int] = {}
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or f"geom_{contact.geom1}"
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or f"geom_{contact.geom2}"
        pair = " / ".join(sorted((name1, name2)))
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def apply_run_params(cfg: Sim2SimConfig, params_dir: str | None) -> None:
    run_params = load_run_params(params_dir, cfg)
    if run_params is None:
        return
    print(f"[INFO] loaded training params: {run_params.params_dir}")
    if run_params.default_pos is not None:
        cfg.default_pos = run_params.default_pos
        print("[INFO] default_pos from params/env.yaml (with MuJoCo right-elbow axis conversion)")
    if run_params.action_scale is not None:
        cfg.action_scale = run_params.action_scale
        print(f"[INFO] action_scale from params: {np.array2string(cfg.action_scale, precision=3)}")
    if run_params.init_height is not None:
        cfg.init_height = run_params.init_height
        print(f"[INFO] init_height from params: {cfg.init_height:.3f}")
    if run_params.policy_ankle_model_path is not None:
        cfg.policy_ankle_model_path = resolve_path(run_params.policy_ankle_model_path)
        print(f"[INFO] policy ankle model from params: {cfg.policy_ankle_model_path}")
    if run_params.ankle_pd_space is not None:
        cfg.ankle_pd_space = run_params.ankle_pd_space
        print(f"[INFO] ankle_pd_space from params: {cfg.ankle_pd_space}")
    cfg.action_clip = run_params.action_clip
    print(f"[INFO] action_clip from params/agent.yaml: {cfg.action_clip}")
    if run_params.static_friction is not None:
        cfg.static_friction = run_params.static_friction
    if run_params.dynamic_friction is not None:
        cfg.dynamic_friction = run_params.dynamic_friction
    if run_params.static_friction is not None or run_params.dynamic_friction is not None:
        print(
            "[INFO] physics material from params: "
            f"static_friction={cfg.static_friction}, dynamic_friction={cfg.dynamic_friction}, "
            f"restitution={run_params.restitution}"
        )


def print_policy_mapping(robot: RobotRuntime, cfg: Sim2SimConfig) -> None:
    print("[INFO] policy/action order -> observe/control order:")
    for policy_id, policy_name in enumerate(cfg.policy_joint_names):
        observe_id = robot.policy_to_mujoco[policy_id]
        control_name = POLICY_ANKLE_TO_CONTROL_JOINT.get(policy_name, policy_name)
        control_id = cfg.control_joint_names.index(control_name) if control_name in cfg.control_joint_names else -1
        print(
            f"  p{policy_id:02d} {policy_name:28s} "
            f"sign={robot.policy_joint_signs[policy_id]:+.0f} scale={robot.action_scale[policy_id]:.3f} "
            f"obs=m{observe_id:02d}:{cfg.mujoco_joint_names[observe_id]:28s} "
            f"ctrl=c{control_id:02d}:{control_name}"
        )


def print_first_obs_debug(
    obs: np.ndarray,
    q: np.ndarray,
    dq: np.ndarray,
    control_q: np.ndarray,
    action: np.ndarray,
    robot: RobotRuntime,
    cfg: Sim2SimConfig,
) -> None:
    print("[DEBUG] first policy observation:")
    print(f"  omega={np.array2string(obs[0, 0:3], precision=4, suppress_small=True)}")
    print(f"  gravity={np.array2string(obs[0, 3:6], precision=4, suppress_small=True)}")
    print(f"  command={np.array2string(obs[0, 6:9], precision=4, suppress_small=True)}")
    q_obs = obs[0, 9 : 9 + cfg.num_actions]
    dq_obs = obs[0, 9 + cfg.num_actions : 9 + 2 * cfg.num_actions]
    for policy_id, policy_name in enumerate(cfg.policy_joint_names):
        observe_id = robot.policy_to_mujoco[policy_id]
        control_name = POLICY_ANKLE_TO_CONTROL_JOINT.get(policy_name, policy_name)
        print(
            f"  p{policy_id:02d} {policy_name:28s} "
            f"q_obs={q_obs[policy_id]:+.4f} dq_obs={dq_obs[policy_id]:+.4f} "
            f"q={q[observe_id]:+.4f} dq={dq[observe_id]:+.4f} "
            f"default={robot.default_pos[observe_id]:+.4f} action_prev={action[policy_id]:+.4f} "
            f"control={control_name}"
        )


def print_first_action_debug(
    action: np.ndarray,
    target_pos: np.ndarray,
    control_target_pos: np.ndarray,
    robot: RobotRuntime,
    cfg: Sim2SimConfig,
) -> None:
    print("[DEBUG] first policy action:")
    print(f"  raw_action={np.array2string(action, precision=4, suppress_small=True)}")
    observe_index = {name: i for i, name in enumerate(cfg.mujoco_joint_names)}
    control_index = {name: i for i, name in enumerate(cfg.control_joint_names)}
    for policy_id, policy_name in enumerate(cfg.policy_joint_names):
        observe_name = UPPER_LOWER_TO_PITCH_ROLL_JOINT.get(policy_name, policy_name)
        observe_id = observe_index[observe_name]
        control_name = POLICY_ANKLE_TO_CONTROL_JOINT.get(policy_name, policy_name)
        control_id = control_index.get(control_name, -1)
        scaled_delta = action[policy_id] * robot.action_scale[policy_id] * robot.policy_joint_signs[policy_id]
        control_text = (
            f"ctrl={control_target_pos[control_id]:+.4f}"
            if control_id >= 0
            else "ctrl=<none>"
        )
        print(
            f"  p{policy_id:02d} {policy_name:28s} "
            f"a={action[policy_id]:+.4f} scale={robot.action_scale[policy_id]:.3f} "
            f"sign={robot.policy_joint_signs[policy_id]:+.0f} delta={scaled_delta:+.4f} "
            f"obs_target={target_pos[observe_id]:+.4f} {control_text}"
        )


def make_robot(args: argparse.Namespace, cfg: Sim2SimConfig) -> RobotRuntime:
    observe_index = {name: i for i, name in enumerate(cfg.mujoco_joint_names)}
    control_to_mujoco = [observe_index.get(name, -1) for name in cfg.control_joint_names]
    allowed_mapped = {
        "left_ankle_upper_joint",
        "left_ankle_lower_joint",
        "right_ankle_upper_joint",
        "right_ankle_lower_joint",
    }
    unmapped = [name for name, observe_id in zip(cfg.control_joint_names, control_to_mujoco) if observe_id < 0]
    unexpected = sorted(set(unmapped) - allowed_mapped)
    if unexpected:
        raise ValueError(f"control_joint_names contains unmapped non-ankle joints: {unexpected}")

    default_control = observe_to_control_default(cfg.default_pos, cfg)
    return RobotRuntime(
        policy_to_mujoco=make_index_mapping(cfg.policy_joint_names, cfg.mujoco_joint_names),
        control_to_mujoco=control_to_mujoco,
        default_pos=cfg.default_pos.copy(),
        control_default_pos=default_control,
        action_scale=cfg.action_scale.copy(),
        policy_joint_signs=cfg.policy_joint_signs.copy(),
        base_pos=np.array([0.0, 0.0, args.init_height], dtype=np.float64),
        action_clip=cfg.action_clip,
        static_friction=cfg.static_friction,
        dynamic_friction=cfg.dynamic_friction,
    )


def run(policy, args: argparse.Namespace, cfg: Sim2SimConfig) -> None:
    print("Keyboard: 8/2 vx, 4/6 vy, 7/9 yaw, arrows vx/yaw, 0 reset, F camera follow")
    listener = start_keyboard_listener()
    print(f"[INFO] policy IO mode: {cfg.policy_io}")
    print(
        "[INFO] target clipping: "
        f"policy_to_joint_limits={cfg.clip_policy_targets_to_joint_limits}, "
        f"control_to_joint_limits={cfg.clip_control_targets_to_joint_limits}, "
        f"upper_lower_action_clip={'on' if cfg.upper_lower_action_clip_min is not None else 'off'}"
    )
    if cfg.policy_io == "upper_lower":
        print(f"[INFO] upper/lower action mode: {cfg.upper_lower_action_mode}")
        print(f"[INFO] upper/lower observation source: {cfg.upper_lower_observation_source}")

    model = mujoco.MjModel.from_xml_path(args.model_path)
    model.opt.timestep = cfg.dt
    data = mujoco.MjData(model)
    print(f"[INFO] MuJoCo XML: {args.model_path}")
    align_default_pos_to_model_ranges(model, cfg)

    robot = make_robot(args, cfg)
    control_index = {name: i for i, name in enumerate(cfg.control_joint_names)}
    has_upper_lower_control = all(name in control_index for name in UPPER_LOWER_JOINT_NAMES)
    print(
        "[INFO] MuJoCo ankle plant: "
        f"{'upper/lower physical motors' if has_upper_lower_control else 'pitch/roll training motors'}"
    )
    ankle_control_ids = {control_index[name] for name in UPPER_LOWER_JOINT_NAMES} if has_upper_lower_control else set()
    torque_control_ids = ankle_control_ids if args.ankle_control in {"torque_jacobian", "torque_tendon_jacobian"} else set()
    if cfg.use_training_passive_ankle_limits:
        patch_training_passive_ankle_limits(model)
    (
        robot.qpos_ids,
        robot.qvel_ids,
        robot.control_qpos_ids,
        robot.control_qvel_ids,
        robot.actuator_ids,
        robot.observe_target_min,
        robot.observe_target_max,
        robot.control_target_min,
        robot.control_target_max,
    ) = build_joint_data(model, cfg)
    if args.debug_actions or args.debug_obs:
        print_policy_mapping(robot, cfg)
    robot.passive_default_qpos_ids, robot.passive_default_values = build_passive_joint_defaults(model, cfg)
    if len(robot.passive_default_qpos_ids):
        print(f"[INFO] passive joint defaults: {cfg.passive_joint_default_pos}")
    if (
        cfg.policy_io == "upper_lower"
        and cfg.upper_lower_action_mode == "direct"
        and args.ankle_control != "position_model"
    ):
        raise ValueError(
            "policy.io='upper_lower' with upper_lower_action_mode='direct' expects "
            "--ankle_control position_model."
        )
    if not has_upper_lower_control:
        ankle_mapper = IdentityControlMapper(robot, cfg)
        policy_ankle_model = AnkleCommandModel(args.policy_ankle_model_path) if cfg.policy_io == "upper_lower" else None
        print("[INFO] simulation plant: pitch/roll joints; upper/lower policy IO is converted like training/play")
        if policy_ankle_model is not None:
            print(f"[INFO] policy ankle IO model: {policy_ankle_model.path}")
    elif cfg.policy_io == "upper_lower" and cfg.upper_lower_action_mode == "direct":
        policy_ankle_model = (
            AnkleCommandModel(args.policy_ankle_model_path)
            if cfg.upper_lower_observation_source == "converted_pitch_roll"
            else None
        )
        ankle_mapper = IdentityControlMapper(robot, cfg)
        print("[INFO] upper/lower policy actions are applied directly to upper/lower motors")
        if policy_ankle_model is not None:
            print(f"[INFO] policy ankle IO model: {policy_ankle_model.path}")
        print("[INFO] physical ankle command model: direct control-space targets")
    elif cfg.policy_io == "upper_lower":
        policy_ankle_model = AnkleCommandModel(args.policy_ankle_model_path)
        if args.ankle_control == "position_model":
            ankle_mapper = PolynomialAnkleMapper(args.ankle_model_path, robot, cfg)
            physical_model_description = ankle_mapper.model.path
        else:
            ankle_mapper = ExactAnkleMapper(model, robot, cfg)
            physical_model_description = "MuJoCo tendon inverse"
        print(f"[INFO] policy ankle IO model: {policy_ankle_model.path}")
        print(f"[INFO] physical ankle command model: {physical_model_description}")
        if cfg.upper_lower_action_mode != "direct":
            print("[INFO] upper/lower policy actions are converted through pitch/roll before motor commands")
    elif args.ankle_control == "position_model":
        ankle_mapper = PolynomialAnkleMapper(args.ankle_model_path, robot, cfg)
        policy_ankle_model = None
        print(f"[INFO] ankle command model: {ankle_mapper.model.path}")
    else:
        ankle_mapper = ExactAnkleMapper(model, robot, cfg)
        policy_ankle_model = None
        print("[INFO] ankle command model: MuJoCo tendon inverse")
    robot.control_default_pos = ankle_mapper.observe_to_control_target(robot.default_pos)
    if cfg.policy_io == "upper_lower" and cfg.upper_lower_action_mode == "direct" and policy_ankle_model is not None:
        apply_upper_lower_command(
            robot.control_default_pos,
            policy_default_upper_lower(robot, cfg, policy_ankle_model),
            cfg,
        )
    if isinstance(ankle_mapper, ExactAnkleMapper):
        for name in UPPER_LOWER_JOINT_NAMES:
            if name in control_index:
                control_id = control_index[name]
                ankle_mapper.qpos_template[robot.control_qpos_ids[control_id]] = robot.control_default_pos[control_id]
    patch_model(model, robot, cfg, torque_control_ids=torque_control_ids)

    if not args.no_auto_base_height:
        robot.base_pos[2] = grounded_base_height(model, data, robot, cfg)
        print(f"[INFO] auto base height: {robot.base_pos[2]:.3f}")

    hold_default_stand(data, robot, cfg)
    if args.ankle_control in {"torque_jacobian", "torque_tendon_jacobian"}:
        data.ctrl[robot.actuator_ids[list(ankle_control_ids)]] = 0.0
    mujoco.mj_forward(model, data)
    initial_qpos = data.qpos.copy()
    initial_qvel = data.qvel.copy()

    renderer, camera, writer, viewer = init_render(model, data, args, cfg)
    action = np.zeros(cfg.num_actions, dtype=np.float64)
    target_pos = robot.default_pos.copy()
    ul_ids = upper_lower_control_ids(cfg)
    ankle_action_ids = policy_ankle_action_ids(cfg)
    real_ankle_default = robot.control_default_pos[ul_ids].copy() if len(ul_ids) else np.zeros(4, dtype=np.float64)
    real_ankle_target = real_ankle_default.copy()
    data.ctrl[robot.actuator_ids] = robot.control_default_pos
    control_target_pos = robot.control_default_pos.copy()
    mujoco_index = {name: i for i, name in enumerate(cfg.mujoco_joint_names)}
    ankle_pr_names = (
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    )
    ankle_pr_ids = (
        np.array([mujoco_index[name] for name in ankle_pr_names], dtype=np.int32)
        if all(name in mujoco_index for name in ankle_pr_names)
        else np.array([], dtype=np.int32)
    )
    print(
        "[INFO] ankle upper/lower default abs="
        f"({real_ankle_default[0]:+.3f}, {real_ankle_default[1]:+.3f}, "
        f"{real_ankle_default[2]:+.3f}, {real_ankle_default[3]:+.3f}); "
        "rel output below is target - default"
    )
    if cfg.ankle_command_rel_clip is not None or cfg.ankle_command_rate_limit is not None:
        print(
            "[INFO] ankle command safety: "
            f"rel_clip={cfg.ankle_command_rel_clip}, rate_limit={cfg.ankle_command_rate_limit} rad/s"
        )
    ctrl_value = robot.control_default_pos.copy()
    if args.ankle_control in {"torque_jacobian", "torque_tendon_jacobian"}:
        ctrl_value[list(ankle_control_ids)] = 0.0
    warmup_steps = int(round(max(0.0, args.stand_warmup) / cfg.dt))
    command_ramp_steps = int(round(max(0.0, args.command_ramp) / cfg.dt))
    min_root_z = float("inf")
    max_abs_yaw_rate = 0.0
    max_abs_vy = 0.0
    start_time = time.time()
    printed_first_obs = False
    printed_first_action = False
    non_ankle_ids = np.array(
        [index for index in range(cfg.num_actions) if index not in set(ankle_action_ids.tolist())],
        dtype=np.int32,
    )
    non_ankle_observe_ids = np.array(
        [index for index in range(cfg.num_actions) if index not in set(ankle_pr_ids.tolist())],
        dtype=np.int32,
    )

    total_steps = int(args.sim_duration / cfg.dt)
    for step in tqdm(range(total_steps), desc="Simulating"):
        if CommandState.reset_requested:
            data.qpos[:] = initial_qpos
            data.qvel[:] = initial_qvel
            data.ctrl[:] = 0.0
            data.ctrl[robot.actuator_ids] = ctrl_value
            action[:] = 0.0
            target_pos[:] = robot.default_pos
            control_target_pos[:] = robot.control_default_pos
            real_ankle_target = real_ankle_default.copy()
            ctrl_value[:] = robot.control_default_pos
            if args.ankle_control in {"torque_jacobian", "torque_tendon_jacobian"}:
                ctrl_value[list(ankle_control_ids)] = 0.0
            CommandState.zero()
            CommandState.reset_requested = False
            mujoco.mj_forward(model, data)

        qpos, qvel, lin_vel_b, omega_b, gravity_b = get_base_obs(
            data,
            use_root_ang_vel=args.base_ang_vel_from_qvel,
            use_sensor_orientation=args.sensor_orientation,
        )
        q = qpos[robot.qpos_ids]
        dq = qvel[robot.qvel_ids]
        min_root_z = min(min_root_z, float(qpos[2]))
        max_abs_yaw_rate = max(max_abs_yaw_rate, abs(float(omega_b[2])))
        max_abs_vy = max(max_abs_vy, abs(float(lin_vel_b[1])))

        if step % cfg.decimation == 0:
            control_q = qpos[robot.control_qpos_ids]
            control_dq = qvel[robot.control_qvel_ids]
            if args.hold_default:
                action[:] = 0.0
                target_pos = robot.default_pos.copy()
                control_target_pos = robot.control_default_pos.copy()
            elif cfg.policy_io == "upper_lower":
                obs = make_observation(
                    q,
                    dq,
                    control_q,
                    control_dq,
                    omega_b,
                    gravity_b,
                    action,
                    robot,
                    cfg,
                    policy_ankle_model,
                )
                if warmup_steps > 0 and step < warmup_steps:
                    obs[0, 6:9] = 0.0
                elif command_ramp_steps > 0:
                    alpha = np.clip((step - warmup_steps) / command_ramp_steps, 0.0, 1.0)
                    obs[0, 6:9] = np.array(
                        [CommandState.vx, CommandState.vy, CommandState.dyaw],
                        dtype=np.float32,
                    ) * alpha
                if args.debug_obs and not printed_first_obs:
                    print_first_obs_debug(obs, q, dq, control_q, action, robot, cfg)
                    printed_first_obs = True
                if step < warmup_steps:
                    action[:] = 0.0
                else:
                    action[:] = policy(obs)
                if robot.action_clip is not None:
                    np.clip(action, -robot.action_clip, robot.action_clip, out=action)
                if args.ankle_action_scale != 1.0 and len(ankle_action_ids):
                    action[ankle_action_ids] *= args.ankle_action_scale
                if args.action_scale != 1.0:
                    action[:] *= args.action_scale
                if cfg.upper_lower_action_mode == "direct":
                    control_target_pos = upper_lower_action_to_control_target(action, robot, cfg, policy_ankle_model)
                    target_pos = robot.default_pos.copy()
                else:
                    target_pos = upper_lower_policy_action_to_pitch_roll_target(action, robot, cfg, policy_ankle_model)
                    control_target_pos = ankle_mapper.observe_to_control_target(target_pos)
                if args.debug_actions and not printed_first_action and step >= warmup_steps:
                    print_first_action_debug(action, target_pos, control_target_pos, robot, cfg)
                    printed_first_action = True
            else:
                obs = make_observation(
                    q,
                    dq,
                    control_q,
                    control_dq,
                    omega_b,
                    gravity_b,
                    action,
                    robot,
                    cfg,
                    policy_ankle_model,
                )
                if warmup_steps > 0 and step < warmup_steps:
                    obs[0, 6:9] = 0.0
                elif command_ramp_steps > 0:
                    alpha = np.clip((step - warmup_steps) / command_ramp_steps, 0.0, 1.0)
                    obs[0, 6:9] = np.array(
                        [CommandState.vx, CommandState.vy, CommandState.dyaw],
                        dtype=np.float32,
                    ) * alpha
                if args.debug_obs and not printed_first_obs:
                    print_first_obs_debug(obs, q, dq, control_q, action, robot, cfg)
                    printed_first_obs = True
                if step < warmup_steps:
                    action[:] = 0.0
                else:
                    action[:] = policy(obs)
                if robot.action_clip is not None:
                    np.clip(action, -robot.action_clip, robot.action_clip, out=action)
                if args.action_scale != 1.0:
                    action[:] *= args.action_scale
                target_pos = action_to_target(action, robot, cfg)
                control_target_pos = ankle_mapper.observe_to_control_target(target_pos)
                if args.debug_actions and not printed_first_action and step >= warmup_steps:
                    print_first_action_debug(action, target_pos, control_target_pos, robot, cfg)
                    printed_first_action = True
            if step < warmup_steps:
                target_pos[:] = robot.default_pos
                control_target_pos[:] = robot.control_default_pos
            if len(ul_ids):
                desired_real_ankle_target = control_target_pos[ul_ids].copy()
                real_ankle_target = limit_real_ankle_command(
                    desired_real_ankle_target,
                    real_ankle_target,
                    real_ankle_default,
                    cfg,
                )
                control_target_pos[ul_ids] = real_ankle_target
            clip_control_target(control_target_pos, robot, cfg)
            ctrl_value = control_target_pos.copy()
            if has_upper_lower_control and args.ankle_control == "torque_jacobian":
                left_tau = ankle_torque_control("left", q, dq, target_pos, robot, cfg, ankle_mapper)
                right_tau = ankle_torque_control("right", q, dq, target_pos, robot, cfg, ankle_mapper)
            elif has_upper_lower_control and args.ankle_control == "torque_tendon_jacobian":
                left_tau = ankle_torque_control_from_tendon_jacobian("left", data, q, dq, target_pos, robot, cfg)
                right_tau = ankle_torque_control_from_tendon_jacobian("right", data, q, dq, target_pos, robot, cfg)
            else:
                left_tau = right_tau = None
            if left_tau is not None:
                ctrl_value[
                    [control_index["left_ankle_upper_joint"], control_index["left_ankle_lower_joint"]]
                ] = left_tau
                ctrl_value[
                    [control_index["right_ankle_upper_joint"], control_index["right_ankle_lower_joint"]]
                ] = right_tau

            policy_step = step // cfg.decimation
            if policy_step % max(1, args.print_every) == 0:
                real_ankle_rel = real_ankle_target - real_ankle_default
                message = (
                    f"cmd=({CommandState.vx:.2f}, {CommandState.vy:.2f}, {CommandState.dyaw:.2f}) "
                    f"vel=({lin_vel_b[0]:.2f}, {lin_vel_b[1]:.2f}, {omega_b[2]:.2f}) "
                    "ankle_ul_abs="
                    f"({real_ankle_target[0]:+.3f}, {real_ankle_target[1]:+.3f}, "
                    f"{real_ankle_target[2]:+.3f}, {real_ankle_target[3]:+.3f}) "
                    "ankle_ul_rel="
                    f"({real_ankle_rel[0]:+.3f}, {real_ankle_rel[1]:+.3f}, "
                    f"{real_ankle_rel[2]:+.3f}, {real_ankle_rel[3]:+.3f})"
                )
                if args.debug_ankle:
                    measured_ul = qpos[robot.control_qpos_ids][ul_ids] if len(ul_ids) else np.zeros(4)
                    ul_error = real_ankle_target - measured_ul
                    ankle_force = (
                        data.actuator_force[robot.actuator_ids[ul_ids]]
                        if len(ul_ids)
                        else np.zeros(4, dtype=np.float64)
                    )
                    if len(ankle_pr_ids):
                        actual_pr = q[ankle_pr_ids]
                        target_pr = target_pos[ankle_pr_ids]
                        pr_error = target_pr - actual_pr
                        non_ankle_action = float(np.max(np.abs(action[non_ankle_ids]))) if len(non_ankle_ids) else 0.0
                        non_ankle_err = (
                            float(np.max(np.abs(target_pos[non_ankle_observe_ids] - q[non_ankle_observe_ids])))
                            if len(non_ankle_observe_ids)
                            else 0.0
                        )
                        message += (
                            f" max_nonankle_action={non_ankle_action:.3f}"
                            f" max_nonankle_err={non_ankle_err:.3f}"
                            " ankle_pr_tgt="
                            f"({target_pr[0]:+.3f}, {target_pr[1]:+.3f}, {target_pr[2]:+.3f}, {target_pr[3]:+.3f})"
                            " ankle_pr_act="
                            f"({actual_pr[0]:+.3f}, {actual_pr[1]:+.3f}, {actual_pr[2]:+.3f}, {actual_pr[3]:+.3f})"
                            " ankle_pr_err="
                            f"({pr_error[0]:+.3f}, {pr_error[1]:+.3f}, {pr_error[2]:+.3f}, {pr_error[3]:+.3f})"
                        )
                    message += (
                        " ankle_ul_meas="
                        f"({measured_ul[0]:+.3f}, {measured_ul[1]:+.3f}, {measured_ul[2]:+.3f}, {measured_ul[3]:+.3f})"
                        " ankle_ul_err="
                        f"({ul_error[0]:+.3f}, {ul_error[1]:+.3f}, {ul_error[2]:+.3f}, {ul_error[3]:+.3f})"
                        " ankle_tau="
                        f"({ankle_force[0]:+.1f}, {ankle_force[1]:+.1f}, {ankle_force[2]:+.1f}, {ankle_force[3]:+.1f})"
                    )
                print(message)
            if not render(data, renderer, camera, writer, viewer, args.headless):
                break

        data.ctrl[robot.actuator_ids] = ctrl_value
        mujoco.mj_step(model, data)

        elapsed = time.time() - start_time
        target_time = (step + 1) * cfg.dt
        if elapsed < target_time:
            time.sleep(target_time - elapsed)

    print(
        "[SUMMARY] "
        f"min_root_z={min_root_z:.3f}, "
        f"max_abs_yaw_rate={max_abs_yaw_rate:.3f}, "
        f"max_abs_vy={max_abs_vy:.3f}, "
        f"final_contacts={summarize_contacts(model, data)}"
    )

    if args.headless:
        writer.release()
        print(f"[INFO] saved video: {args.output}")
    else:
        viewer.close()
    if listener is not None:
        listener.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lens110 motor sim2sim with pitch/roll to upper/lower ankle mapping.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Sim2sim JSON config path.")
    parser.add_argument("--load_model", default=None, help="Override TorchScript policy path from config.")
    parser.add_argument("--load_onnx", default=None, help="Override ONNX policy path from config.")
    parser.add_argument("--policy_backend", choices=("torchscript", "onnx"), default=None)
    parser.add_argument(
        "--params_dir",
        default=None,
        help="Training params directory containing env.yaml and agent.yaml. Defaults to config source.params_dir.",
    )
    parser.add_argument("--policy_io", choices=("pitch_roll", "upper_lower"), default=None)
    parser.add_argument("--upper_lower_action_mode", choices=("direct", "through_pitch_roll"), default=None)
    parser.add_argument("--upper_lower_observation_source", choices=("control", "converted_pitch_roll"), default=None)
    parser.add_argument(
        "--clip_policy_targets_to_joint_limits",
        dest="clip_policy_targets_to_joint_limits",
        action="store_true",
        default=None,
        help="Clip policy-space position targets to MuJoCo joint limits before conversion/control.",
    )
    parser.add_argument(
        "--no_clip_policy_targets_to_joint_limits",
        dest="clip_policy_targets_to_joint_limits",
        action="store_false",
        help="Do not clip policy-space position targets to MuJoCo joint limits.",
    )
    parser.add_argument(
        "--clip_control_targets_to_joint_limits",
        dest="clip_control_targets_to_joint_limits",
        action="store_true",
        default=None,
        help="Clip final control targets to MuJoCo joint limits.",
    )
    parser.add_argument(
        "--no_clip_control_targets_to_joint_limits",
        dest="clip_control_targets_to_joint_limits",
        action="store_false",
        help="Do not clip final control targets to MuJoCo joint limits.",
    )
    parser.add_argument("--debug_ankle", action="store_true", help="Print ankle pitch/roll tracking diagnostics.")
    parser.add_argument(
        "--ankle_motor_kp_scale",
        type=float,
        default=None,
        help="Scale MuJoCo position-PD kp for the four physical ankle upper/lower motors.",
    )
    parser.add_argument(
        "--ankle_motor_kd_scale",
        type=float,
        default=None,
        help="Scale MuJoCo position-PD kd for the four physical ankle upper/lower motors.",
    )
    parser.add_argument(
        "--ankle_motor_tau_scale",
        type=float,
        default=None,
        help="Scale torque limits for the four physical ankle upper/lower motors.",
    )
    parser.add_argument(
        "--ankle_command_rel_clip",
        type=float,
        default=None,
        help="Override ankle command relative position clip around the default pose. Negative disables it.",
    )
    parser.add_argument(
        "--ankle_command_rate_limit",
        type=float,
        default=None,
        help="Override ankle command rate limit in rad/s. Negative disables it.",
    )
    parser.add_argument("--model_path", default=None, help="Override MuJoCo XML model path from config.")
    parser.add_argument(
        "--base_ang_vel_from_qvel",
        dest="base_ang_vel_from_qvel",
        action="store_true",
        default=False,
        help="Use root qvel rotated into base frame. This matches IsaacLab root-state observations better.",
    )
    parser.add_argument(
        "--base_ang_vel_from_sensor",
        dest="base_ang_vel_from_qvel",
        action="store_false",
        help="Use the MuJoCo gyro sensor for base angular velocity.",
    )
    parser.add_argument(
        "--sensor_orientation",
        dest="sensor_orientation",
        action="store_true",
        default=False,
        help="Use MuJoCo IMU framequat for projected gravity.",
    )
    parser.add_argument(
        "--root_orientation_from_qpos",
        dest="sensor_orientation",
        action="store_false",
        help="Use the free-joint root quaternion for projected gravity.",
    )
    parser.add_argument("--tendon_stiffness", type=float, default=None, help="Override all ankle tendon stiffness values.")
    parser.add_argument("--tendon_damping", type=float, default=None, help="Override all ankle tendon damping values.")
    parser.add_argument("--tendon_frictionloss", type=float, default=None, help="Override all ankle tendon frictionloss values.")
    parser.add_argument(
        "--ankle_action_scale",
        type=float,
        default=1.0,
        help="Post-policy scale for the four ankle action slots only. Useful when the real upper/lower plant is too aggressive.",
    )
    parser.add_argument(
        "--action_scale",
        type=float,
        default=1.0,
        help="Post-policy scale for all action slots.",
    )
    parser.add_argument(
        "--policy_action_clip",
        type=float,
        default=None,
        help="Override raw policy action clipping. Negative disables clipping.",
    )
    parser.add_argument("--debug_actions", action="store_true", help="Print policy/action to MuJoCo mapping.")
    parser.add_argument("--debug_obs", action="store_true", help="Print the first policy observation slices.")
    parser.add_argument(
        "--ankle_control",
        choices=("position_model", "position_inverse", "torque_tendon_jacobian", "torque_jacobian"),
        default="position_model",
        help="How to convert policy ankle pitch/roll control to upper/lower motors.",
    )
    parser.add_argument("--ankle_model_path", default=None, help="Override pitch/roll -> upper/lower model path.")
    parser.add_argument(
        "--policy_ankle_model_path",
        default=None,
        help="Override the pitch/roll <-> upper/lower model used by upper_lower policy IO.",
    )
    parser.add_argument("--cmd_vel", type=float, nargs=3, default=None, help="Initial command: vx vy yaw_rate.")
    parser.add_argument("--sim_duration", type=float, default=None)
    parser.add_argument("--stand_warmup", type=float, default=0.0, help="Hold default pose before policy starts.")
    parser.add_argument("--command_ramp", type=float, default=0.0, help="Ramp command velocity after warmup.")
    parser.add_argument(
        "--hold_default",
        action="store_true",
        help="Skip policy inference and hold the loaded default pose. Useful to isolate MuJoCo plant/default-pose issues.",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", default="simulation_motor_upper_lower.mp4")
    parser.add_argument("--viewer_width", type=int, default=1280)
    parser.add_argument("--viewer_height", type=int, default=720)
    parser.add_argument("--show_menus", action="store_true", help="Show MuJoCo viewer's left-side overlay menus.")
    parser.add_argument("--init_height", type=float, default=None)
    parser.add_argument("--no_auto_base_height", action="store_true")
    parser.add_argument("--print_every", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    args.params_dir = (
        resolve_path(args.params_dir, os.path.dirname(os.path.abspath(args.config)))
        if args.params_dir
        else cfg.source_params_dir
    )
    args.load_model = resolve_path(args.load_model) if args.load_model else cfg.torchscript_policy
    args.load_onnx = resolve_path(args.load_onnx) if args.load_onnx else cfg.onnx_policy
    args.policy_backend = args.policy_backend or cfg.policy_backend
    if args.policy_io is not None:
        cfg.policy_io = args.policy_io
    if args.upper_lower_action_mode is not None:
        cfg.upper_lower_action_mode = args.upper_lower_action_mode
    if args.upper_lower_observation_source is not None:
        cfg.upper_lower_observation_source = args.upper_lower_observation_source
    validate_policy_joint_order(cfg)
    apply_run_params(cfg, args.params_dir)
    if args.clip_policy_targets_to_joint_limits is not None:
        cfg.clip_policy_targets_to_joint_limits = args.clip_policy_targets_to_joint_limits
    if args.clip_control_targets_to_joint_limits is not None:
        cfg.clip_control_targets_to_joint_limits = args.clip_control_targets_to_joint_limits
    if args.ankle_motor_kp_scale is not None:
        cfg.ankle_motor_kp_scale = args.ankle_motor_kp_scale
    if args.ankle_motor_kd_scale is not None:
        cfg.ankle_motor_kd_scale = args.ankle_motor_kd_scale
    if args.ankle_motor_tau_scale is not None:
        cfg.ankle_motor_tau_scale = args.ankle_motor_tau_scale
    if args.ankle_command_rel_clip is not None:
        cfg.ankle_command_rel_clip = None if args.ankle_command_rel_clip < 0.0 else args.ankle_command_rel_clip
    if args.ankle_command_rate_limit is not None:
        cfg.ankle_command_rate_limit = (
            None if args.ankle_command_rate_limit < 0.0 else args.ankle_command_rate_limit
        )
    if args.tendon_stiffness is not None:
        cfg.tendon_stiffness = args.tendon_stiffness
    if args.tendon_damping is not None:
        cfg.tendon_damping = args.tendon_damping
    if args.tendon_frictionloss is not None:
        cfg.tendon_frictionloss = args.tendon_frictionloss
    if args.policy_action_clip is not None:
        cfg.action_clip = None if args.policy_action_clip < 0.0 else args.policy_action_clip
    args.model_path = resolve_path(args.model_path) if args.model_path else cfg.model_path
    args.ankle_model_path = resolve_path(args.ankle_model_path) if args.ankle_model_path else cfg.ankle_model_path
    args.policy_ankle_model_path = (
        resolve_path(args.policy_ankle_model_path)
        if args.policy_ankle_model_path
        else cfg.policy_ankle_model_path
    )
    args.init_height = args.init_height if args.init_height is not None else (cfg.init_height or cfg.fallback_init_height)
    args.cmd_vel = tuple(args.cmd_vel) if args.cmd_vel is not None else cfg.default_cmd_vel
    args.sim_duration = args.sim_duration if args.sim_duration is not None else cfg.sim_duration
    args.print_every = args.print_every if args.print_every is not None else cfg.print_every

    CommandState.set_initial(tuple(args.cmd_vel))
    if args.policy_backend == "onnx":
        if not args.load_onnx and not args.hold_default:
            raise FileNotFoundError(
                "No local ONNX policy is configured. Provide --load_onnx with a local policy.onnx file."
            )
        policy = OnnxPolicy(args.load_onnx) if args.load_onnx else None
        if policy is not None:
            print(f"[INFO] loaded ONNX policy with {policy.backend}: {args.load_onnx}")
    else:
        if not args.load_model and not args.hold_default:
            raise FileNotFoundError(
                "No local TorchScript policy is configured. Provide --load_model with a local policy.pt file."
            )
        policy = TorchScriptPolicy(args.load_model) if args.load_model else None
        if policy is not None:
            print(f"[INFO] loaded TorchScript policy: {args.load_model}")
    print(
        "[INFO] base obs mode: "
        f"ang_vel={'qvel' if args.base_ang_vel_from_qvel else 'imu_gyro'}, "
        f"orientation={'imu_framequat' if args.sensor_orientation else 'qpos_root'}"
    )
    run(policy, args, cfg)


if __name__ == "__main__":
    main()
