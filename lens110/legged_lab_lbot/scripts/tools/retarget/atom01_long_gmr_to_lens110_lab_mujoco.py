"""Convert atom01_long motions to Lens110 lab motion data using MuJoCo FK.

This avoids Isaac/Omniverse startup for the retarget data generation step.
It reorders the 23-DOF atom01_long data into Lens110's 21-DOF order, grounds
the root height against the Lens110 foot collision boxes, and uses the Lens110
MuJoCo model to compute key body world positions.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import joblib
import mujoco
import numpy as np
from scipy.optimize import least_squares


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = REPO_ROOT / "source/legged_lab/legged_lab/data/MotionData/atom01_long_lab"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "source/legged_lab/legged_lab/data/MotionData/lens110_lab"
DEFAULT_MODEL = REPO_ROOT / "source/legged_lab/legged_lab/data/Robots/model_humanoid_lens110/mjcf/lens110_21dof.xml"
DEFAULT_SOURCE_MODEL = (
    REPO_ROOT / "source/legged_lab/legged_lab/data/Robots/atom01_long_base_link/atom01_long_base_link.xml"
)

SOURCE_GMR_DOF_NAMES = [
    "left_thigh_yaw_joint",
    "left_thigh_roll_joint",
    "left_thigh_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_thigh_yaw_joint",
    "right_thigh_roll_joint",
    "right_thigh_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "torso_joint",
    "left_arm_pitch_joint",
    "left_arm_roll_joint",
    "left_arm_yaw_joint",
    "left_elbow_pitch_joint",
    "left_elbow_yaw_joint",
    "right_arm_pitch_joint",
    "right_arm_roll_joint",
    "right_arm_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_yaw_joint",
]

SOURCE_LAB_DOF_NAMES = [
    "left_thigh_yaw_joint",
    "right_thigh_yaw_joint",
    "torso_joint",
    "left_thigh_roll_joint",
    "right_thigh_roll_joint",
    "left_arm_pitch_joint",
    "right_arm_pitch_joint",
    "left_thigh_pitch_joint",
    "right_thigh_pitch_joint",
    "left_arm_roll_joint",
    "right_arm_roll_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_arm_yaw_joint",
    "right_arm_yaw_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_elbow_pitch_joint",
    "right_elbow_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_elbow_yaw_joint",
    "right_elbow_yaw_joint",
]

TARGET_SOURCE_DOF_NAMES = [
    "left_thigh_pitch_joint",
    "left_thigh_roll_joint",
    "left_thigh_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_thigh_pitch_joint",
    "right_thigh_roll_joint",
    "right_thigh_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "torso_joint",
    "right_arm_pitch_joint",
    "right_arm_roll_joint",
    "right_arm_yaw_joint",
    "right_elbow_pitch_joint",
    "left_arm_pitch_joint",
    "left_arm_roll_joint",
    "left_arm_yaw_joint",
    "left_elbow_pitch_joint",
]

TARGET_JOINT_NAMES = [
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
    "torso_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
]

TARGET_LEG_JOINTS = {
    "left": [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
    ],
    "right": [
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ],
}

SOURCE_FOOT_BODY_NAMES = {
    "left": "left_ankle_roll_link",
    "right": "right_ankle_roll_link",
}

TARGET_FOOT_BODY_NAMES = {
    "left": "left_ankle_roll_link",
    "right": "right_ankle_roll_link",
}

TARGET_DOF_SCALE = np.asarray(
    [
        1.0,  # left_hip_pitch_joint
        1.0,  # left_hip_roll_joint
        1.0,  # left_hip_yaw_joint
        1.0,  # left_knee_joint
        1.0,  # left_ankle_pitch_joint
        1.0,  # left_ankle_roll_joint
        1.0,  # right_hip_pitch_joint
        1.0,  # right_hip_roll_joint
        1.0,  # right_hip_yaw_joint
        1.0,  # right_knee_joint
        1.0,  # right_ankle_pitch_joint
        1.0,  # right_ankle_roll_joint
        1.0,  # torso_yaw_joint
        1.0,  # right_shoulder_pitch_joint
        1.0,  # right_shoulder_roll_joint
        -1.0,  # right_shoulder_yaw_joint
        -0.9505836575875487,  # right_elbow_joint
        1.0,  # left_shoulder_pitch_joint
        1.0,  # left_shoulder_roll_joint
        -1.0,  # left_shoulder_yaw_joint
        -0.9505836575875487,  # left_elbow_joint
    ],
    dtype=np.float64,
)

TARGET_DOF_OFFSET = np.asarray(
    [
        0.0,  # left_hip_pitch_joint
        0.0,  # left_hip_roll_joint
        0.0,  # left_hip_yaw_joint
        0.0,  # left_knee_joint
        0.0,  # left_ankle_pitch_joint
        0.0,  # left_ankle_roll_joint
        0.0,  # right_hip_pitch_joint
        0.0,  # right_hip_roll_joint
        0.0,  # right_hip_yaw_joint
        0.0,  # right_knee_joint
        0.0,  # right_ankle_pitch_joint
        0.0,  # right_ankle_roll_joint
        0.0,  # torso_yaw_joint
        0.0,  # right_shoulder_pitch_joint
        0.0,  # right_shoulder_roll_joint
        0.0,  # right_shoulder_yaw_joint
        -0.8635836575875487,  # right_elbow_joint
        0.0,  # left_shoulder_pitch_joint
        0.0,  # left_shoulder_roll_joint
        0.0,  # left_shoulder_yaw_joint
        -0.8635836575875487,  # left_elbow_joint
    ],
    dtype=np.float64,
)

ARM_NEUTRAL_DOF_POS = {
    "left_shoulder_pitch_joint": 0.4,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": -0.8,
    "right_shoulder_pitch_joint": 0.4,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": -0.8,
}

SOURCE_ARM_BODY_CHAINS = [
    ("left_arm_pitch_link", "left_elbow_pitch_link", "left_elbow_yaw_link"),
    ("right_arm_pitch_link", "right_elbow_pitch_link", "right_elbow_yaw_link"),
]

TARGET_ARM_BODY_CHAINS = [
    ("left_shoulder_pitch_link", "left_elbow_link", "left_hand"),
    ("right_shoulder_pitch_link", "right_elbow_link", "right_hand"),
]

SOURCE_HEIGHT_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]

TARGET_HEIGHT_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]

KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
]

FOOT_GEOM_NAMES = [
    "left_ankle_roll_collision",
    "right_ankle_roll_collision",
    "left_ankle_pitch_collision",
    "right_ankle_pitch_collision",
]


def _install_numpy_pickle_shim() -> None:
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


def _load_motion(path: Path) -> dict:
    _install_numpy_pickle_shim()
    return joblib.load(path)


def _source_names(motion: dict, path: Path) -> list[str]:
    names = motion.get("dof_names")
    if names is None:
        names = motion.get("joint_names")
    if names is None:
        if "_lab" in path.parent.name:
            return list(SOURCE_LAB_DOF_NAMES)
        return list(SOURCE_GMR_DOF_NAMES)
    return [str(name) for name in names]


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quat, axis=1, keepdims=True)
    return quat / np.maximum(norms, 1.0e-8)


def _xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return quat_xyzw[:, [3, 0, 1, 2]]


def _root_rot_wxyz(motion: dict, path: Path, root_rot_format: str) -> np.ndarray:
    root_rot = _normalize_quat(np.asarray(motion["root_rot"], dtype=np.float64))
    fmt = root_rot_format
    if fmt == "auto":
        fmt = str(motion.get("root_rot_format", "")).lower()
    if fmt == "auto" or not fmt:
        fmt = "wxyz" if "_lab" in path.parent.name else "xyzw"
    if fmt == "wxyz":
        return root_rot
    if fmt == "xyzw":
        return _xyzw_to_wxyz(root_rot)
    raise ValueError(f"Unsupported root rotation format: {root_rot_format}")


def _joint_qpos_indices(model: mujoco.MjModel, joint_names: list[str]) -> np.ndarray:
    indices: list[int] = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Joint not found in MuJoCo model: {name}")
        indices.append(int(model.jnt_qposadr[joint_id]))
    return np.asarray(indices, dtype=np.int32)


def _joint_qpos_index_map(model: mujoco.MjModel, joint_names: list[str]) -> dict[str, int]:
    return {name: int(index) for name, index in zip(joint_names, _joint_qpos_indices(model, joint_names))}


def _joint_limits(model: mujoco.MjModel, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    lower: list[float] = []
    upper: list[float] = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Joint not found in MuJoCo model: {name}")
        lower.append(float(model.jnt_range[joint_id, 0]))
        upper.append(float(model.jnt_range[joint_id, 1]))
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def _joint_index_map(joint_names: list[str]) -> dict[str, int]:
    return {name: index for index, name in enumerate(joint_names)}


def _arm_neutral_pose(joint_names: list[str]) -> np.ndarray:
    neutral = np.zeros(len(joint_names), dtype=np.float64)
    for name, value in ARM_NEUTRAL_DOF_POS.items():
        neutral[joint_names.index(name)] = value
    return neutral


def _body_ids(model: mujoco.MjModel, body_names: list[str]) -> np.ndarray:
    ids: list[int] = []
    for name in body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"Body not found in MuJoCo model: {name}")
        ids.append(body_id)
    return np.asarray(ids, dtype=np.int32)


def _body_position(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body not found in MuJoCo model: {body_name}")
    return data.xpos[body_id].copy()


def _body_id(model: mujoco.MjModel, body_name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body not found in MuJoCo model: {body_name}")
    return int(body_id)


def _root_to_mean_body_height(model_path: Path, body_names: list[str]) -> float:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    body_heights = [_body_position(model, data, body_name)[2] for body_name in body_names]
    root_height = float(data.qpos[2] - np.mean(body_heights))
    if root_height <= 1.0e-8:
        raise ValueError(f"Invalid root/body height from {model_path}: {root_height}")
    return root_height


def _auto_body_motion_scale(source_model: Path, target_model: Path) -> float:
    source_height = _root_to_mean_body_height(source_model, SOURCE_HEIGHT_BODY_NAMES)
    target_height = _root_to_mean_body_height(target_model, TARGET_HEIGHT_BODY_NAMES)
    return float(target_height / source_height)


def _parse_body_motion_scale(scale_arg: str, source_model: Path, target_model: Path) -> float:
    if scale_arg.lower() == "auto":
        return _auto_body_motion_scale(source_model, target_model)
    scale = float(scale_arg)
    if scale <= 0.0:
        raise ValueError(f"--body-motion-scale must be positive or 'auto', got {scale_arg}")
    return scale


def _mean_arm_chain_length(model_path: Path, body_chains: list[tuple[str, str, str]]) -> float:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    lengths = []
    for shoulder_name, elbow_name, hand_name in body_chains:
        shoulder_pos = _body_position(model, data, shoulder_name)
        elbow_pos = _body_position(model, data, elbow_name)
        hand_pos = _body_position(model, data, hand_name)
        lengths.append(float(np.linalg.norm(elbow_pos - shoulder_pos) + np.linalg.norm(hand_pos - elbow_pos)))
    return float(np.mean(lengths))


def _auto_arm_motion_scale(source_model: Path, target_model: Path) -> float:
    source_arm_length = _mean_arm_chain_length(source_model, SOURCE_ARM_BODY_CHAINS)
    target_arm_length = _mean_arm_chain_length(target_model, TARGET_ARM_BODY_CHAINS)
    if target_arm_length <= 1.0e-8:
        raise ValueError(f"Invalid target arm length from {target_model}: {target_arm_length}")
    return float(np.clip(source_arm_length / target_arm_length, 0.35, 1.25))


def _parse_arm_motion_scale(scale_arg: str, source_model: Path, target_model: Path) -> float:
    if scale_arg.lower() == "auto":
        return _auto_arm_motion_scale(source_model, target_model)
    scale = float(scale_arg)
    if scale <= 0.0:
        raise ValueError(f"--arm-motion-scale must be positive or 'auto', got {scale_arg}")
    return scale


def _scale_arm_motion(dof_pos: np.ndarray, scale: float, neutral: np.ndarray, joint_names: list[str]) -> np.ndarray:
    if np.isclose(scale, 1.0):
        return dof_pos
    scaled = dof_pos.copy()
    joint_indices = _joint_index_map(joint_names)
    arm_indices = [joint_indices[name] for name in ARM_NEUTRAL_DOF_POS]
    scaled[:, arm_indices] = neutral[arm_indices] + scale * (scaled[:, arm_indices] - neutral[arm_indices])
    return scaled


def _scale_root_motion(root_pos: np.ndarray, scale: float) -> np.ndarray:
    if np.isclose(scale, 1.0):
        return root_pos.copy()
    scaled = root_pos.copy()
    anchor_xy = root_pos[0, :2].copy()
    scaled[:, :2] = anchor_xy + scale * (root_pos[:, :2] - anchor_xy)
    scaled[:, 2] = root_pos[:, 2] * scale
    return scaled


def _reset_free_root(data: mujoco.MjData) -> None:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0:3] = 0.0
    data.qpos[3:7] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _source_foot_local_positions(
    source_model: mujoco.MjModel,
    source_qpos_indices: np.ndarray,
    source_dof_pos: np.ndarray,
) -> dict[str, np.ndarray]:
    data = mujoco.MjData(source_model)
    body_ids = {side: _body_id(source_model, body_name) for side, body_name in SOURCE_FOOT_BODY_NAMES.items()}
    positions = {
        "left": np.zeros((source_dof_pos.shape[0], 3), dtype=np.float64),
        "right": np.zeros((source_dof_pos.shape[0], 3), dtype=np.float64),
    }
    for frame in range(source_dof_pos.shape[0]):
        _reset_free_root(data)
        data.qpos[source_qpos_indices] = source_dof_pos[frame]
        mujoco.mj_forward(source_model, data)
        for side, body_id in body_ids.items():
            positions[side][frame] = data.xpos[body_id]
    return positions


def _neutral_foot_local_positions(
    model: mujoco.MjModel,
    foot_body_names: dict[str, str],
    joint_values: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    data = mujoco.MjData(model)
    _reset_free_root(data)
    if joint_values:
        qpos_map = _joint_qpos_index_map(model, list(joint_values.keys()))
        for joint_name, value in joint_values.items():
            data.qpos[qpos_map[joint_name]] = value
    mujoco.mj_forward(model, data)
    return {side: _body_position(model, data, body_name) for side, body_name in foot_body_names.items()}


def _set_target_joint_values(
    data: mujoco.MjData,
    qpos_map: dict[str, int],
    joint_values: dict[str, float],
) -> None:
    for joint_name, value in joint_values.items():
        data.qpos[qpos_map[joint_name]] = value


def _solve_leg_ik(
    model: mujoco.MjModel,
    qpos_map: dict[str, int],
    foot_body_name: str,
    leg_joint_names: list[str],
    desired_foot_position: np.ndarray,
    initial_guess: np.ndarray,
    reference_guess: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    data = mujoco.MjData(model)
    foot_body_id = _body_id(model, foot_body_name)

    def residual(q: np.ndarray) -> np.ndarray:
        _reset_free_root(data)
        for joint_name, value in zip(leg_joint_names, q):
            data.qpos[qpos_map[joint_name]] = value
        mujoco.mj_forward(model, data)
        position_error = (data.xpos[foot_body_id] - desired_foot_position) * 18.0
        regularization = (q - reference_guess) * 0.08
        return np.concatenate([position_error, regularization], axis=0)

    result = least_squares(
        residual,
        np.clip(initial_guess, lower, upper),
        bounds=(lower, upper),
        method="trf",
        max_nfev=80,
        xtol=1.0e-6,
        ftol=1.0e-6,
        gtol=1.0e-6,
    )
    return np.clip(result.x, lower, upper)


def _apply_leg_ik(
    path: Path,
    motion: dict,
    source_names: list[str],
    source_model: mujoco.MjModel,
    target_model: mujoco.MjModel,
    target_qpos_map: dict[str, int],
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    body_motion_scale: float,
    dof_pos: np.ndarray,
) -> np.ndarray:
    source_indices = [source_names.index(name) for name in SOURCE_LAB_DOF_NAMES]
    source_dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)[:, source_indices]
    source_qpos_indices = _joint_qpos_indices(source_model, SOURCE_LAB_DOF_NAMES)

    source_foot_local = _source_foot_local_positions(source_model, source_qpos_indices, source_dof_pos)
    source_neutral_foot = _neutral_foot_local_positions(source_model, SOURCE_FOOT_BODY_NAMES)
    target_neutral_foot = _neutral_foot_local_positions(target_model, TARGET_FOOT_BODY_NAMES)
    target_joint_index = _joint_index_map(TARGET_JOINT_NAMES)
    target_limit_index = {name: target_joint_index[name] for name in TARGET_JOINT_NAMES}

    previous_solution = {
        side: np.asarray([dof_pos[0, target_joint_index[name]] for name in TARGET_LEG_JOINTS[side]], dtype=np.float64)
        for side in ("left", "right")
    }

    for frame in range(dof_pos.shape[0]):
        for side in ("left", "right"):
            leg_joint_names = TARGET_LEG_JOINTS[side]
            leg_indices = [target_limit_index[name] for name in leg_joint_names]
            lower = joint_lower[leg_indices]
            upper = joint_upper[leg_indices]
            reference = np.asarray([dof_pos[frame, target_joint_index[name]] for name in leg_joint_names], dtype=np.float64)
            desired = target_neutral_foot[side] + body_motion_scale * (
                source_foot_local[side][frame] - source_neutral_foot[side]
            )
            solution = _solve_leg_ik(
                target_model,
                target_qpos_map,
                TARGET_FOOT_BODY_NAMES[side],
                leg_joint_names,
                desired,
                initial_guess=0.75 * previous_solution[side] + 0.25 * reference,
                reference_guess=reference,
                lower=lower,
                upper=upper,
            )
            previous_solution[side] = solution
            for joint_name, value in zip(leg_joint_names, solution):
                dof_pos[frame, target_joint_index[joint_name]] = value

    return dof_pos


def _geom_ids(model: mujoco.MjModel, geom_names: list[str]) -> np.ndarray:
    ids: list[int] = []
    for name in geom_names:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            raise ValueError(f"Geom not found in MuJoCo model: {name}")
        ids.append(geom_id)
    return np.asarray(ids, dtype=np.int32)


def _box_geom_min_z(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float:
    size = model.geom_size[geom_id]
    corners = np.asarray(
        [
            [sx, sy, sz]
            for sx in (-size[0], size[0])
            for sy in (-size[1], size[1])
            for sz in (-size[2], size[2])
        ],
        dtype=np.float64,
    )
    geom_xmat = data.geom_xmat[geom_id].reshape(3, 3)
    world_corners = data.geom_xpos[geom_id] + corners @ geom_xmat.T
    return float(world_corners[:, 2].min())


def _foot_min_z(model: mujoco.MjModel, data: mujoco.MjData, foot_geom_ids: np.ndarray) -> float:
    return min(_box_geom_min_z(model, data, int(geom_id)) for geom_id in foot_geom_ids)


def convert_motion(
    path: Path,
    model: mujoco.MjModel,
    source_model: mujoco.MjModel | None,
    qpos_indices: np.ndarray,
    qpos_map: dict[str, int],
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    key_body_ids: np.ndarray,
    foot_geom_ids: np.ndarray,
    root_rot_format: str,
    ground: bool,
    ground_clearance: float,
    clip_joint_limits: bool,
    arm_motion_scale: float,
    body_motion_scale: float,
    arm_neutral: np.ndarray,
    leg_ik: bool,
) -> dict:
    motion = _load_motion(path)
    source_names = _source_names(motion, path)
    source_indices = [source_names.index(name) for name in TARGET_SOURCE_DOF_NAMES]

    root_pos = _scale_root_motion(np.asarray(motion["root_pos"], dtype=np.float64), body_motion_scale)
    root_rot_wxyz = _root_rot_wxyz(motion, path, root_rot_format)
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)[:, source_indices] * TARGET_DOF_SCALE + TARGET_DOF_OFFSET
    dof_pos = _scale_arm_motion(dof_pos, arm_motion_scale, arm_neutral, TARGET_JOINT_NAMES)
    if clip_joint_limits:
        dof_pos = np.clip(dof_pos, joint_lower, joint_upper)
    if leg_ik:
        if source_model is None:
            raise ValueError("--leg-ik requires a valid --source-model")
        dof_pos = _apply_leg_ik(
            path,
            motion,
            source_names,
            source_model,
            model,
            qpos_map,
            joint_lower,
            joint_upper,
            body_motion_scale,
            dof_pos,
        )
        if clip_joint_limits:
            dof_pos = np.clip(dof_pos, joint_lower, joint_upper)

    data = mujoco.MjData(model)
    key_body_pos = np.zeros((root_pos.shape[0], len(key_body_ids), 3), dtype=np.float32)
    grounded_root_pos = root_pos.copy()

    for frame in range(root_pos.shape[0]):
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.qpos[0:3] = grounded_root_pos[frame]
        data.qpos[3:7] = root_rot_wxyz[frame]
        data.qpos[qpos_indices] = dof_pos[frame]
        mujoco.mj_forward(model, data)
        if ground:
            grounded_root_pos[frame, 2] -= _foot_min_z(model, data, foot_geom_ids) - ground_clearance
            data.qpos[2] = grounded_root_pos[frame, 2]
            mujoco.mj_forward(model, data)
        key_body_pos[frame] = data.xpos[key_body_ids]

    return {
        "fps": motion["fps"],
        "root_pos": grounded_root_pos.astype(np.float32),
        "root_rot": root_rot_wxyz.astype(np.float32),
        "dof_pos": dof_pos.astype(np.float32),
        "dof_names": np.asarray(TARGET_JOINT_NAMES),
        "loop_mode": 0,
        "key_body_pos": key_body_pos,
        "key_body_names": list(KEY_BODY_NAMES),
        "root_rot_format": "wxyz",
        "source_dof_names": source_names,
        "retarget_info": {
            "method": "atom01_long_to_lens110_mujoco_leg_ik_scaled_body_and_arms" if leg_ik else "atom01_long_to_lens110_mujoco_fk_scaled_body_and_arms",
            "arm_motion_scale": float(arm_motion_scale),
            "body_motion_scale": float(body_motion_scale),
            "leg_ik": bool(leg_ik),
            "input_file": str(path),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert atom01_long motions to Lens110 lab data with MuJoCo FK.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--root-rot-format",
        choices=["auto", "xyzw", "wxyz"],
        default="auto",
        help="Input root_rot format. auto treats *_lab as wxyz and *_gmr as xyzw.",
    )
    parser.add_argument("--no-ground", action="store_true", help="Disable per-frame Lens110 foot-ground height correction.")
    parser.add_argument("--ground-clearance", type=float, default=0.0, help="Target minimum foot clearance above z=0.")
    parser.add_argument("--no-clip-joint-limits", action="store_true", help="Disable Lens110 joint-limit clipping.")
    parser.add_argument(
        "--source-model",
        type=Path,
        default=DEFAULT_SOURCE_MODEL,
        help="Source robot MuJoCo XML used to compute automatic arm length scaling.",
    )
    parser.add_argument(
        "--arm-motion-scale",
        default="auto",
        help="Scale Lens110 arm joint deviations around its neutral arm pose. Use 'auto' for source_arm_len / target_arm_len.",
    )
    parser.add_argument(
        "--body-motion-scale",
        default="auto",
        help=(
            "Scale root trajectory by Lens110/source root-to-ankle height. "
            "Use 'auto' for target_height / source_height."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip input files whose output file already exists.",
    )
    parser.add_argument(
        "--leg-ik",
        action="store_true",
        help="Use source foot FK plus Lens110 leg IK instead of direct leg joint remapping.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.model))
    source_model = mujoco.MjModel.from_xml_path(str(args.source_model)) if args.leg_ik else None
    qpos_indices = _joint_qpos_indices(model, TARGET_JOINT_NAMES)
    qpos_map = _joint_qpos_index_map(model, TARGET_JOINT_NAMES)
    joint_lower, joint_upper = _joint_limits(model, TARGET_JOINT_NAMES)
    key_body_ids = _body_ids(model, KEY_BODY_NAMES)
    foot_geom_ids = _geom_ids(model, FOOT_GEOM_NAMES)
    arm_neutral = _arm_neutral_pose(TARGET_JOINT_NAMES)
    arm_motion_scale = _parse_arm_motion_scale(args.arm_motion_scale, args.source_model, args.model)
    body_motion_scale = _parse_body_motion_scale(args.body_motion_scale, args.source_model, args.model)
    print(f"Using arm_motion_scale={arm_motion_scale:.4f}")
    print(f"Using body_motion_scale={body_motion_scale:.4f}")

    input_files = sorted(args.input_dir.glob("*.pkl"))
    if not input_files:
        raise FileNotFoundError(f"No .pkl files found under {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    converted_count = 0
    skipped_count = 0
    for path in input_files:
        output_path = args.output_dir / path.name
        if args.skip_existing and output_path.exists():
            print(f"Skipping existing {output_path}")
            skipped_count += 1
            continue
        print(f"Converting {path.name}")
        converted = convert_motion(
            path=path,
            model=model,
            source_model=source_model,
            qpos_indices=qpos_indices,
            qpos_map=qpos_map,
            joint_lower=joint_lower,
            joint_upper=joint_upper,
            key_body_ids=key_body_ids,
            foot_geom_ids=foot_geom_ids,
            root_rot_format=args.root_rot_format,
            ground=not args.no_ground,
            ground_clearance=args.ground_clearance,
            clip_joint_limits=not args.no_clip_joint_limits,
            arm_motion_scale=arm_motion_scale,
            body_motion_scale=body_motion_scale,
            arm_neutral=arm_neutral,
            leg_ik=args.leg_ik,
        )
        with output_path.open("wb") as file:
            pickle.dump(converted, file)
        converted_count += 1
        print(f"Saved {output_path}")

    print(f"Done. Converted {converted_count} files to {args.output_dir}; skipped {skipped_count} existing files.")


if __name__ == "__main__":
    main()
