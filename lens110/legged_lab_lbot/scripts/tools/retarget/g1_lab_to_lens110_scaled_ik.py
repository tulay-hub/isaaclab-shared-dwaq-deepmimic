"""Retarget Unitree G1 29DoF motion data to Lens110 21DoF with scaling and IK.

This script converts flat Legged Lab G1 motion pickle files under:

    data/MotionData/g1_lab/

into Lens110-compatible motion pickle files under:

    data/MotionData/lens110_lab/

The output preserves the Legged Lab motion-data schema:
    - fps
    - root_pos
    - root_rot   (wxyz)
    - dof_pos
    - loop_mode
    - key_body_pos

Extra metadata such as ``dof_names`` and ``key_body_names`` is included to make
future inspection easier, but the runtime loader does not depend on them.
"""

from __future__ import annotations

import argparse
import pickle
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = next(
    (
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "projects").is_dir() and (parent / "frameworks").is_dir()
    ),
    REPO_ROOT,
)
G1_RETARGET_CONFIG = Path(__file__).resolve().parent / "config" / "g1.yaml"


def _load_source_dof_names(config_path: Path) -> list[str]:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    dof_names = list(config["lab_dof_names"])
    if len(dof_names) != 29:
        raise ValueError(f"Expected 29 G1 lab dof names from {config_path}, got {len(dof_names)}")
    return dof_names


SOURCE_DOF_NAMES = _load_source_dof_names(G1_RETARGET_CONFIG)

SOURCE_REFERENCE_JOINT_POS = {
    "left_hip_pitch_joint": -0.1,
    "right_hip_pitch_joint": -0.1,
    "left_knee_joint": 0.3,
    "right_knee_joint": 0.3,
    "left_ankle_pitch_joint": -0.2,
    "right_ankle_pitch_joint": -0.2,
    "left_shoulder_pitch_joint": 0.3,
    "right_shoulder_pitch_joint": 0.3,
    "left_shoulder_roll_joint": 0.25,
    "right_shoulder_roll_joint": -0.25,
    "left_elbow_joint": 0.97,
    "right_elbow_joint": 0.97,
    "left_wrist_roll_joint": 0.15,
    "right_wrist_roll_joint": -0.15,
}

TARGET_REFERENCE_JOINT_POS = {
    "left_hip_pitch_joint": -0.14,
    "left_hip_roll_joint": 0.01,
    "left_hip_yaw_joint": -0.1,
    "left_knee_joint": 0.36,
    "left_ankle_pitch_joint": -0.20,
    "left_ankle_roll_joint": -0.20,
    "right_hip_pitch_joint": -0.14,
    "right_hip_roll_joint": -0.01,
    "right_hip_yaw_joint": 0.1,
    "right_knee_joint": 0.36,
    "right_ankle_pitch_joint": -0.20,
    "right_ankle_roll_joint": -0.20,
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

SOURCE_FOOT_LINKS = {
    "left": "left_ankle_roll_link",
    "right": "right_ankle_roll_link",
}
SOURCE_TOE_LINKS = {
    "left": "left_toe_link",
    "right": "right_toe_link",
}
SOURCE_ELBOW_LINKS = {
    "left": "left_elbow_link",
    "right": "right_elbow_link",
}
SOURCE_HAND_LINKS = {
    "left": "left_wrist_yaw_link",
    "right": "right_wrist_yaw_link",
}
SOURCE_SHOULDER_LINKS = {
    "left": "left_shoulder_pitch_link",
    "right": "right_shoulder_pitch_link",
}

TARGET_KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
]

TARGET_FOOT_LINKS = {
    "left": "left_ankle_roll_link",
    "right": "right_ankle_roll_link",
}
TARGET_TOE_LOCAL_OFFSETS = {
    "left": np.array([0.12, 0.0, -0.03], dtype=np.float64),
    "right": np.array([0.12, 0.0, -0.03], dtype=np.float64),
}
TARGET_ELBOW_LINKS = {
    "left": "left_elbow_link",
    "right": "right_elbow_link",
}
TARGET_HAND_LINKS = {
    "left": "left_elbow_link",
    "right": "right_elbow_link",
}
TARGET_SHOULDER_LINKS = {
    "left": "left_shoulder_pitch_link",
    "right": "right_shoulder_pitch_link",
}
TARGET_HAND_LOCAL_OFFSETS = {
    "left": np.array([-0.009, 0.0, -0.228], dtype=np.float64),
    "right": np.array([-0.009, 0.0, -0.228], dtype=np.float64),
}

STAND_STILL_TAGS = ("stand",)
MOVING_TAGS = ("walk", "run", "turn", "side_step", "move")

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

TARGET_ARM_JOINTS = {
    "left": [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
    ],
    "right": [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
    ],
}

TARGET_DIRECT_MAP = {
    "torso_yaw_joint": "waist_yaw_joint",
    "left_shoulder_pitch_joint": "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint": "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint": "left_shoulder_yaw_joint",
    "left_elbow_joint": "left_elbow_joint",
    "right_shoulder_pitch_joint": "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint": "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint": "right_shoulder_yaw_joint",
    "right_elbow_joint": "right_elbow_joint",
    "left_hip_pitch_joint": "left_hip_pitch_joint",
    "left_hip_roll_joint": "left_hip_roll_joint",
    "left_hip_yaw_joint": "left_hip_yaw_joint",
    "left_knee_joint": "left_knee_joint",
    "left_ankle_pitch_joint": "left_ankle_pitch_joint",
    "left_ankle_roll_joint": "left_ankle_roll_joint",
    "right_hip_pitch_joint": "right_hip_pitch_joint",
    "right_hip_roll_joint": "right_hip_roll_joint",
    "right_hip_yaw_joint": "right_hip_yaw_joint",
    "right_knee_joint": "right_knee_joint",
    "right_ankle_pitch_joint": "right_ankle_pitch_joint",
    "right_ankle_roll_joint": "right_ankle_roll_joint",
}

ARM_REFERENCE_SIGN = {
    "left_shoulder_pitch_joint": 1.0,
    "left_shoulder_roll_joint": 1.0,
    "left_shoulder_yaw_joint": 1.0,
    "left_elbow_joint": -1.0,
    "right_shoulder_pitch_joint": 1.0,
    "right_shoulder_roll_joint": 1.0,
    "right_shoulder_yaw_joint": 1.0,
    "right_elbow_joint": -1.0,
}

DEFAULT_LIMIT = np.pi


def _load_pickle(path: Path):
    """Load pickles written by numpy/joblib across numpy version changes."""
    import numpy as np

    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    with path.open("rb") as file:
        return pickle.load(file)


def _save_pickle(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def _wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    return quat_wxyz[..., [1, 2, 3, 0]]


def _xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return quat_xyzw[..., [3, 0, 1, 2]]


def _origin_matrix(origin):
    xyz = np.zeros(3, dtype=np.float64)
    rpy = np.zeros(3, dtype=np.float64)
    if origin is not None:
        if origin.attrib.get("xyz"):
            xyz = np.array([float(value) for value in origin.attrib["xyz"].split()], dtype=np.float64)
        if origin.attrib.get("rpy"):
            rpy = np.array([float(value) for value in origin.attrib["rpy"].split()], dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    transform[:3, 3] = xyz
    return transform


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    axis_norm = np.linalg.norm(axis)
    if axis_norm > 0.0:
        transform[:3, :3] = Rotation.from_rotvec(axis / axis_norm * angle).as_matrix()
    return transform


@dataclass
class JointInfo:
    name: str
    joint_type: str
    parent: str
    child: str
    axis: np.ndarray
    origin: np.ndarray
    lower: float
    upper: float


class KinematicTree:
    def __init__(self, urdf_path: Path):
        root = ET.parse(urdf_path).getroot()
        self.children_by_parent: dict[str, list[JointInfo]] = {}
        self.joints: dict[str, JointInfo] = {}
        self.joint_order: list[str] = []
        child_links = set()
        all_links = {link.attrib["name"] for link in root.findall("link")}

        for joint in root.findall("joint"):
            joint_type = joint.attrib.get("type", "fixed")
            name = joint.attrib["name"]
            parent = joint.find("parent").attrib["link"]
            child = joint.find("child").attrib["link"]
            axis_node = joint.find("axis")
            axis_xyz = "0 0 1"
            if axis_node is not None:
                axis_xyz = axis_node.attrib.get("xyz", axis_xyz)
            axis = np.array([float(value) for value in axis_xyz.split()], dtype=np.float64)
            limit_node = joint.find("limit")
            lower = -DEFAULT_LIMIT
            upper = DEFAULT_LIMIT
            if limit_node is not None:
                lower = float(limit_node.attrib.get("lower", lower))
                upper = float(limit_node.attrib.get("upper", upper))
            info = JointInfo(
                name=name,
                joint_type=joint_type,
                parent=parent,
                child=child,
                axis=axis,
                origin=_origin_matrix(joint.find("origin")),
                lower=lower,
                upper=upper,
            )
            self.children_by_parent.setdefault(parent, []).append(info)
            self.joints[name] = info
            if joint_type != "fixed":
                self.joint_order.append(name)
            child_links.add(child)

        self.base_link = next((link for link in all_links if link not in child_links), "pelvis")
        self._chain_cache: dict[str, list[JointInfo]] = {}

    def chain_to_link(self, target_link: str) -> list[JointInfo]:
        if target_link in self._chain_cache:
            return self._chain_cache[target_link]
        stack = [(self.base_link, [])]
        while stack:
            link_name, path = stack.pop()
            if link_name == target_link:
                self._chain_cache[target_link] = path
                return path
            for joint in self.children_by_parent.get(link_name, []):
                stack.append((joint.child, path + [joint]))
        raise ValueError(f"Failed to find chain to {target_link}.")

    def forward_kinematics(
        self,
        joint_values: dict[str, float],
        target_link: str,
    ) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        for joint in self.chain_to_link(target_link):
            transform = transform @ joint.origin
            if joint.joint_type != "fixed":
                transform = transform @ _axis_rotation(joint.axis, joint_values.get(joint.name, 0.0))
        return transform

    def joint_bounds(self, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
        lower = np.array([self.joints[name].lower for name in joint_names], dtype=np.float64)
        upper = np.array([self.joints[name].upper for name in joint_names], dtype=np.float64)
        return lower, upper

    def joint_axis_in_base(self, joint_name: str, joint_values: dict[str, float]) -> np.ndarray:
        joint_info = self.joints[joint_name]
        transform = np.eye(4, dtype=np.float64)
        for joint in self.chain_to_link(joint_info.parent):
            transform = transform @ joint.origin
            if joint.joint_type != "fixed":
                transform = transform @ _axis_rotation(joint.axis, joint_values.get(joint.name, 0.0))
        transform = transform @ joint_info.origin
        axis = transform[:3, :3] @ joint_info.axis
        norm = np.linalg.norm(axis)
        if norm <= 1.0e-8:
            return axis
        return axis / norm


def _joint_dict_from_vector(joint_names: list[str], joint_values: np.ndarray) -> dict[str, float]:
    return {joint_name: float(value) for joint_name, value in zip(joint_names, joint_values)}


def _point_on_link(
    tree: KinematicTree,
    joint_values: dict[str, float],
    target_link: str,
    local_offset: np.ndarray | None = None,
) -> np.ndarray:
    transform = tree.forward_kinematics(joint_values, target_link)
    offset = np.zeros(3, dtype=np.float64) if local_offset is None else np.asarray(local_offset, dtype=np.float64)
    return transform[:3, 3] + transform[:3, :3] @ offset


def _axis_sign_map(
    source_tree: KinematicTree,
    target_tree: KinematicTree,
    target_joint_names: list[str],
) -> dict[str, float]:
    signs: dict[str, float] = {}
    for target_name in target_joint_names:
        if target_name in ARM_REFERENCE_SIGN:
            signs[target_name] = ARM_REFERENCE_SIGN[target_name]
            continue
        source_name = TARGET_DIRECT_MAP.get(target_name)
        if not source_name or source_name not in source_tree.joints or target_name not in target_tree.joints:
            signs[target_name] = 1.0
            continue
        source_axis = source_tree.joint_axis_in_base(source_name, SOURCE_REFERENCE_JOINT_POS)
        target_axis = target_tree.joint_axis_in_base(target_name, TARGET_REFERENCE_JOINT_POS)
        signs[target_name] = 1.0 if float(np.dot(source_axis, target_axis)) >= 0.0 else -1.0
    return signs


def _reference_target_joints(
    source_joint_dict: dict[str, float],
    target_joint_names: list[str],
    axis_signs: dict[str, float],
) -> np.ndarray:
    return np.array(
        [
            TARGET_REFERENCE_JOINT_POS.get(joint_name, 0.0)
            + axis_signs.get(joint_name, 1.0)
            * (
                source_joint_dict.get(TARGET_DIRECT_MAP.get(joint_name, ""), 0.0)
                - SOURCE_REFERENCE_JOINT_POS.get(TARGET_DIRECT_MAP.get(joint_name, ""), 0.0)
            )
            for joint_name in target_joint_names
        ],
        dtype=np.float64,
    )


def _reference_target_arm_joints(
    source_joint_dict: dict[str, float],
    target_joint_names: list[str],
    axis_signs: dict[str, float],
) -> np.ndarray:
    reference = []
    for joint_name in target_joint_names:
        source_name = TARGET_DIRECT_MAP.get(joint_name, "")
        source_value = source_joint_dict.get(source_name, 0.0)
        source_reference = SOURCE_REFERENCE_JOINT_POS.get(source_name, 0.0)
        target_reference = TARGET_REFERENCE_JOINT_POS.get(joint_name, 0.0)
        sign = axis_signs.get(joint_name, ARM_REFERENCE_SIGN.get(joint_name, 1.0))
        reference.append(target_reference + sign * (source_value - source_reference))
    return np.asarray(reference, dtype=np.float64)


def _neutral_height(tree: KinematicTree, foot_link: str) -> float:
    return abs(float(tree.forward_kinematics({}, foot_link)[2, 3]))


def _neutral_reach(tree: KinematicTree, start_link: str, end_link: str) -> float:
    start = tree.forward_kinematics({}, start_link)[:3, 3]
    end = tree.forward_kinematics({}, end_link)[:3, 3]
    return float(np.linalg.norm(end - start))


def _scaled_root_positions(root_pos: np.ndarray, horizontal_scale: float, vertical_scale: float) -> np.ndarray:
    scaled = root_pos.copy()
    scaled[:, 0] = root_pos[0, 0] + horizontal_scale * (root_pos[:, 0] - root_pos[0, 0])
    scaled[:, 1] = root_pos[0, 1] + horizontal_scale * (root_pos[:, 1] - root_pos[0, 1])
    scaled[:, 2] = root_pos[:, 2] * vertical_scale
    return scaled


def _scaled_world_positions(
    world_pos: np.ndarray,
    anchor_xy: np.ndarray,
    horizontal_scale: float,
    vertical_scale: float,
) -> np.ndarray:
    scaled = np.asarray(world_pos, dtype=np.float64).copy()
    scaled[..., 0] = anchor_xy[0] + horizontal_scale * (scaled[..., 0] - anchor_xy[0])
    scaled[..., 1] = anchor_xy[1] + horizontal_scale * (scaled[..., 1] - anchor_xy[1])
    scaled[..., 2] = scaled[..., 2] * vertical_scale
    return scaled


def _is_stand_still_clip(clip_name: str, root_pos: np.ndarray) -> bool:
    lower_name = clip_name.lower()
    has_stand_tag = any(tag in lower_name for tag in STAND_STILL_TAGS)
    has_motion_tag = any(tag in lower_name for tag in MOVING_TAGS)
    planar_span = float(np.linalg.norm(np.ptp(root_pos[:, :2], axis=0)))
    return has_stand_tag and not has_motion_tag and planar_span < 0.12


def _upright_root_rotation(root_rot_wxyz: np.ndarray, blend: float | np.ndarray) -> np.ndarray:
    blend_array = np.asarray(blend, dtype=np.float64)
    if blend_array.ndim == 0:
        blend_array = np.full(root_rot_wxyz.shape[0], float(blend_array), dtype=np.float64)
    blend_array = np.clip(blend_array, 0.0, 1.0)
    euler_xyz = Rotation.from_quat(_wxyz_to_xyzw(root_rot_wxyz)).as_euler("xyz")
    euler_xyz[:, 0] *= 1.0 - blend_array
    euler_xyz[:, 1] *= 1.0 - blend_array
    return _xyzw_to_wxyz(Rotation.from_euler("xyz", euler_xyz).as_quat())


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, [(pad, pad), (0, 0)], mode="edge")
    kernel = np.full(window, 1.0 / window, dtype=np.float64)
    smoothed = np.empty_like(values, dtype=np.float64)
    for col in range(values.shape[1]):
        smoothed[:, col] = np.convolve(padded[:, col], kernel, mode="valid")
    return smoothed


def _limit_joint_step(solution: np.ndarray, previous: np.ndarray, max_step: float) -> np.ndarray:
    if max_step <= 0.0:
        return solution
    return previous + np.clip(solution - previous, -max_step, max_step)


def _detect_support_sides(source_foot_world: dict[str, np.ndarray]) -> list[str]:
    left = np.asarray(source_foot_world["left"], dtype=np.float64)
    right = np.asarray(source_foot_world["right"], dtype=np.float64)

    def _speed(track: np.ndarray) -> np.ndarray:
        velocity = np.zeros(track.shape[0], dtype=np.float64)
        if track.shape[0] > 1:
            velocity[1:] = np.linalg.norm(track[1:] - track[:-1], axis=1)
            velocity[0] = velocity[1]
        return velocity

    left_score = left[:, 2] + 0.35 * _speed(left)
    right_score = right[:, 2] + 0.35 * _speed(right)

    support_sides: list[str] = []
    previous = "left" if left_score[0] <= right_score[0] else "right"
    hysteresis = 0.025
    for frame_index in range(left.shape[0]):
        if abs(left_score[frame_index] - right_score[frame_index]) < hysteresis:
            support = previous
        else:
            support = "left" if left_score[frame_index] <= right_score[frame_index] else "right"
        support_sides.append(support)
        previous = support
    return support_sides


def _solve_link_ik(
    tree: KinematicTree,
    joint_names: list[str],
    target_link: str,
    desired_position: np.ndarray,
    desired_rotation: np.ndarray | None,
    initial_guess: np.ndarray,
    reference_guess: np.ndarray,
    position_weight: float,
    rotation_weight: float,
    regularization_weight: float,
    local_point: np.ndarray | None = None,
    extra_point_targets: list[tuple[np.ndarray, np.ndarray, float]] | None = None,
    max_nfev: int = 80,
) -> np.ndarray:
    lower, upper = tree.joint_bounds(joint_names)
    initial = np.clip(initial_guess, lower, upper)
    local_offset = np.zeros(3, dtype=np.float64) if local_point is None else np.asarray(local_point, dtype=np.float64)

    def residual(q: np.ndarray) -> np.ndarray:
        joint_values = _joint_dict_from_vector(joint_names, q)
        transform = tree.forward_kinematics(joint_values, target_link)
        current_point = transform[:3, 3] + transform[:3, :3] @ local_offset
        residuals = [(current_point - desired_position) * position_weight]
        if extra_point_targets:
            for point_offset, point_target, point_weight in extra_point_targets:
                point_offset = np.asarray(point_offset, dtype=np.float64)
                point_target = np.asarray(point_target, dtype=np.float64)
                point_position = transform[:3, 3] + transform[:3, :3] @ point_offset
                residuals.append((point_position - point_target) * point_weight)
        if desired_rotation is not None and rotation_weight > 0.0:
            current_rotation = Rotation.from_matrix(transform[:3, :3])
            target_rotation = Rotation.from_matrix(desired_rotation)
            rotation_error = (current_rotation.inv() * target_rotation).as_rotvec() * rotation_weight
            residuals.append(rotation_error)
        residuals.append((q - reference_guess) * regularization_weight)
        return np.concatenate(residuals, axis=0)

    result = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        method="trf",
        max_nfev=max_nfev,
        xtol=1.0e-6,
        ftol=1.0e-6,
        gtol=1.0e-6,
    )
    return np.clip(result.x, lower, upper)


def _cyclic_velocity(values: np.ndarray, dt: float) -> np.ndarray:
    velocity = np.zeros_like(values)
    velocity[:-1] = (values[1:] - values[:-1]) / dt
    velocity[-1] = velocity[-2]
    return velocity


def _compute_key_body_positions_world(
    root_pos: np.ndarray,
    root_rot_wxyz: np.ndarray,
    target_tree: KinematicTree,
    target_dof_names: list[str],
    target_dof_pos: np.ndarray,
) -> np.ndarray:
    num_frames = target_dof_pos.shape[0]
    key_body_pos = np.zeros((num_frames, len(TARGET_KEY_BODY_NAMES), 3), dtype=np.float32)
    root_rot_body_to_world = Rotation.from_quat(_wxyz_to_xyzw(root_rot_wxyz))

    for frame_index in range(num_frames):
        joint_dict = _joint_dict_from_vector(target_dof_names, target_dof_pos[frame_index])
        for body_index, body_name in enumerate(TARGET_KEY_BODY_NAMES):
            local_position = target_tree.forward_kinematics(joint_dict, body_name)[:3, 3]
            world_position = root_pos[frame_index] + root_rot_body_to_world[frame_index].apply(local_position)
            key_body_pos[frame_index, body_index] = world_position.astype(np.float32)
    return key_body_pos


def _retarget_motion(
    motion_dict: dict,
    clip_name: str,
    source_tree: KinematicTree,
    target_tree: KinematicTree,
    target_dof_names: list[str],
    stride_scale_factor: float,
    root_upright_blend: float,
    stand_root_upright_blend: float,
    stand_root_position_blend: float,
    root_smoothing_window: int,
    arm_smoothing_window: int,
    leg_max_step: float,
    arm_max_step: float,
) -> dict:
    source_root_pos = np.asarray(motion_dict["root_pos"], dtype=np.float64)
    source_root_rot = np.asarray(motion_dict["root_rot"], dtype=np.float64)
    source_dof_pos = np.asarray(motion_dict["dof_pos"], dtype=np.float64)

    if source_dof_pos.shape[1] != len(SOURCE_DOF_NAMES):
        raise ValueError(
            f"Expected source dof_pos width {len(SOURCE_DOF_NAMES)}, got {source_dof_pos.shape[1]}."
        )

    fps = float(motion_dict["fps"])
    source_leg_height = abs(
        float(source_tree.forward_kinematics(SOURCE_REFERENCE_JOINT_POS, SOURCE_FOOT_LINKS["left"])[2, 3])
    )
    target_leg_height = abs(
        float(target_tree.forward_kinematics(TARGET_REFERENCE_JOINT_POS, TARGET_FOOT_LINKS["left"])[2, 3])
    )
    leg_scale = target_leg_height / source_leg_height
    stride_scale = leg_scale * stride_scale_factor
    is_stand_still = _is_stand_still_clip(clip_name, source_root_pos)
    axis_signs = _axis_sign_map(source_tree, target_tree, target_dof_names)

    source_shoulder_reference_transform = {
        side: source_tree.forward_kinematics(SOURCE_REFERENCE_JOINT_POS, SOURCE_SHOULDER_LINKS[side])
        for side in ("left", "right")
    }
    source_shoulder_reference = {
        side: source_shoulder_reference_transform[side][:3, 3] for side in ("left", "right")
    }
    source_elbow_reference = {
        side: source_tree.forward_kinematics(SOURCE_REFERENCE_JOINT_POS, SOURCE_ELBOW_LINKS[side])[:3, 3]
        for side in ("left", "right")
    }
    source_hand_reference = {
        side: _point_on_link(source_tree, SOURCE_REFERENCE_JOINT_POS, SOURCE_HAND_LINKS[side])
        for side in ("left", "right")
    }
    target_shoulder_reference_transform = {
        side: target_tree.forward_kinematics(TARGET_REFERENCE_JOINT_POS, TARGET_SHOULDER_LINKS[side])
        for side in ("left", "right")
    }
    target_shoulder_reference = {
        side: target_shoulder_reference_transform[side][:3, 3] for side in ("left", "right")
    }
    target_elbow_reference = {
        side: target_tree.forward_kinematics(TARGET_REFERENCE_JOINT_POS, TARGET_ELBOW_LINKS[side])[:3, 3]
        for side in ("left", "right")
    }
    target_hand_reference = {
        side: _point_on_link(
            target_tree,
            TARGET_REFERENCE_JOINT_POS,
            TARGET_HAND_LINKS[side],
            TARGET_HAND_LOCAL_OFFSETS[side],
        )
        for side in ("left", "right")
    }
    source_foot_reference = {
        side: source_tree.forward_kinematics(SOURCE_REFERENCE_JOINT_POS, SOURCE_FOOT_LINKS[side])[:3, 3]
        for side in ("left", "right")
    }
    source_toe_reference = {
        side: source_tree.forward_kinematics(SOURCE_REFERENCE_JOINT_POS, SOURCE_TOE_LINKS[side])[:3, 3]
        for side in ("left", "right")
    }
    target_foot_reference = {
        side: target_tree.forward_kinematics(TARGET_REFERENCE_JOINT_POS, TARGET_FOOT_LINKS[side])[:3, 3]
        for side in ("left", "right")
    }
    target_toe_reference = {
        side: _point_on_link(
            target_tree,
            TARGET_REFERENCE_JOINT_POS,
            TARGET_FOOT_LINKS[side],
            TARGET_TOE_LOCAL_OFFSETS[side],
        )
        for side in ("left", "right")
    }

    source_arm_reach = float(np.linalg.norm(source_hand_reference["left"] - source_shoulder_reference["left"]))
    target_arm_reach = float(np.linalg.norm(target_hand_reference["left"] - target_shoulder_reference["left"]))
    arm_scale = target_arm_reach / source_arm_reach

    anchor_xy = source_root_pos[0, :2].copy()
    scaled_root_pos_nominal = _scaled_root_positions(
        source_root_pos,
        horizontal_scale=stride_scale,
        vertical_scale=leg_scale,
    )
    target_root_pos = scaled_root_pos_nominal.copy()
    root_blend = stand_root_upright_blend if is_stand_still else root_upright_blend
    target_root_rot = _upright_root_rotation(source_root_rot, root_blend).astype(np.float32)
    target_dof_pos = np.zeros((source_dof_pos.shape[0], len(target_dof_names)), dtype=np.float64)

    source_root_rot_body_to_world = Rotation.from_quat(_wxyz_to_xyzw(source_root_rot))
    target_root_rot_body_to_world = Rotation.from_quat(_wxyz_to_xyzw(target_root_rot))
    source_foot_world = {}
    if "key_body_pos" in motion_dict and np.asarray(motion_dict["key_body_pos"]).ndim == 3:
        key_body_pos = np.asarray(motion_dict["key_body_pos"], dtype=np.float64)
        if key_body_pos.shape[1] >= 2:
            source_foot_world["left"] = key_body_pos[:, 0, :]
            source_foot_world["right"] = key_body_pos[:, 1, :]
    if len(source_foot_world) != 2:
        source_foot_world = {
            side: np.zeros((source_dof_pos.shape[0], 3), dtype=np.float64) for side in ("left", "right")
        }
        for frame_index in range(source_dof_pos.shape[0]):
            source_joint_dict = _joint_dict_from_vector(SOURCE_DOF_NAMES, source_dof_pos[frame_index])
            for side in ("left", "right"):
                local_position = source_tree.forward_kinematics(source_joint_dict, SOURCE_FOOT_LINKS[side])[:3, 3]
                source_foot_world[side][frame_index] = (
                    source_root_pos[frame_index] + source_root_rot_body_to_world[frame_index].apply(local_position)
                )

    target_foot_world = {
        side: _scaled_world_positions(
            source_foot_world[side],
            anchor_xy=anchor_xy,
            horizontal_scale=stride_scale,
            vertical_scale=leg_scale,
        )
        for side in ("left", "right")
    }
    support_sides = _detect_support_sides(source_foot_world)

    prev_leg_solution = {
        side: np.array([TARGET_REFERENCE_JOINT_POS.get(name, 0.0) for name in TARGET_LEG_JOINTS[side]], dtype=np.float64)
        for side in ("left", "right")
    }
    prev_arm_solution = {
        side: np.array([TARGET_REFERENCE_JOINT_POS.get(name, 0.0) for name in TARGET_ARM_JOINTS[side]], dtype=np.float64)
        for side in ("left", "right")
    }

    for frame_index in range(source_dof_pos.shape[0]):
        source_joint_dict = _joint_dict_from_vector(SOURCE_DOF_NAMES, source_dof_pos[frame_index])
        target_joint_dict = {joint_name: float(TARGET_REFERENCE_JOINT_POS.get(joint_name, 0.0)) for joint_name in target_dof_names}
        target_joint_dict["torso_yaw_joint"] = float(
            _reference_target_joints(source_joint_dict, ["torso_yaw_joint"], axis_signs)[0]
        )
        root_rotation = target_root_rot_body_to_world[frame_index]
        support_side = support_sides[frame_index]
        swing_side = "right" if support_side == "left" else "left"

        source_support_local = source_tree.forward_kinematics(source_joint_dict, SOURCE_FOOT_LINKS[support_side])[:3, 3]
        source_support_toe_local = source_tree.forward_kinematics(source_joint_dict, SOURCE_TOE_LINKS[support_side])[
            :3, 3
        ]
        desired_support_local = target_foot_reference[support_side] + leg_scale * (
            source_support_local - source_foot_reference[support_side]
        )
        desired_support_toe_local = target_toe_reference[support_side] + leg_scale * (
            source_support_toe_local - source_toe_reference[support_side]
        )
        support_reference = _reference_target_joints(source_joint_dict, TARGET_LEG_JOINTS[support_side], axis_signs)
        support_solution = _solve_link_ik(
            target_tree,
            TARGET_LEG_JOINTS[support_side],
            TARGET_FOOT_LINKS[support_side],
            desired_support_local,
            desired_rotation=None,
            initial_guess=0.4 * prev_leg_solution[support_side] + 0.6 * support_reference,
            reference_guess=support_reference,
            position_weight=8.0,
            rotation_weight=0.0,
            regularization_weight=0.18,
            extra_point_targets=[(TARGET_TOE_LOCAL_OFFSETS[support_side], desired_support_toe_local, 10.0)],
        )
        if frame_index > 0:
            support_solution = _limit_joint_step(support_solution, prev_leg_solution[support_side], leg_max_step)
        prev_leg_solution[support_side] = support_solution
        for joint_name, value in zip(TARGET_LEG_JOINTS[support_side], support_solution):
            target_joint_dict[joint_name] = float(value)

        support_local_actual = target_tree.forward_kinematics(target_joint_dict, TARGET_FOOT_LINKS[support_side])[:3, 3]
        support_world_target = target_foot_world[support_side][frame_index]
        target_root_pos_frame = support_world_target - root_rotation.apply(support_local_actual)

        swing_world_target = target_foot_world[swing_side][frame_index]
        desired_swing_local = root_rotation.inv().apply(swing_world_target - target_root_pos_frame)
        source_swing_toe_local = source_tree.forward_kinematics(source_joint_dict, SOURCE_TOE_LINKS[swing_side])[
            :3, 3
        ]
        desired_swing_toe_local = target_toe_reference[swing_side] + leg_scale * (
            source_swing_toe_local - source_toe_reference[swing_side]
        )
        swing_reference = _reference_target_joints(source_joint_dict, TARGET_LEG_JOINTS[swing_side], axis_signs)
        swing_solution = _solve_link_ik(
            target_tree,
            TARGET_LEG_JOINTS[swing_side],
            TARGET_FOOT_LINKS[swing_side],
            desired_swing_local,
            desired_rotation=None,
            initial_guess=prev_leg_solution[swing_side],
            reference_guess=swing_reference,
            position_weight=22.0,
            rotation_weight=0.0,
            regularization_weight=0.04,
            extra_point_targets=[(TARGET_TOE_LOCAL_OFFSETS[swing_side], desired_swing_toe_local, 14.0)],
        )
        if frame_index > 0:
            swing_solution = _limit_joint_step(swing_solution, prev_leg_solution[swing_side], leg_max_step)
        prev_leg_solution[swing_side] = swing_solution
        for joint_name, value in zip(TARGET_LEG_JOINTS[swing_side], swing_solution):
            target_joint_dict[joint_name] = float(value)

        support_local_refined = target_tree.forward_kinematics(target_joint_dict, TARGET_FOOT_LINKS[support_side])[:3, 3]
        target_root_pos_frame = support_world_target - root_rotation.apply(support_local_refined)
        target_root_pos_frame[2] = max(target_root_pos_frame[2], scaled_root_pos_nominal[frame_index, 2] - 0.08)
        if frame_index > 0 and support_sides[frame_index] == support_sides[frame_index - 1]:
            target_root_pos_frame = 0.65 * target_root_pos_frame + 0.35 * target_root_pos[frame_index - 1]
        else:
            target_root_pos_frame = 0.8 * target_root_pos_frame + 0.2 * scaled_root_pos_nominal[frame_index]
        target_root_pos[frame_index] = target_root_pos_frame

        for side in ("left", "right"):
            source_hand_transform = source_tree.forward_kinematics(source_joint_dict, SOURCE_HAND_LINKS[side])
            source_hand_position = source_hand_transform[:3, 3]
            source_elbow_position = source_tree.forward_kinematics(source_joint_dict, SOURCE_ELBOW_LINKS[side])[:3, 3]
            desired_elbow_position = target_elbow_reference[side] + arm_scale * (
                source_elbow_position - source_elbow_reference[side]
            )
            desired_hand_position = target_hand_reference[side] + arm_scale * (
                source_hand_position - source_hand_reference[side]
            )
            arm_reference = _reference_target_arm_joints(source_joint_dict, TARGET_ARM_JOINTS[side], axis_signs)
            arm_regularization = 0.7 * prev_arm_solution[side] + 0.3 * arm_reference
            arm_solution = _solve_link_ik(
                target_tree,
                TARGET_ARM_JOINTS[side],
                TARGET_ELBOW_LINKS[side],
                desired_hand_position,
                desired_rotation=None,
                initial_guess=0.75 * prev_arm_solution[side] + 0.25 * arm_reference,
                reference_guess=arm_regularization,
                position_weight=16.0,
                rotation_weight=0.0,
                regularization_weight=0.16,
                local_point=TARGET_HAND_LOCAL_OFFSETS[side],
                extra_point_targets=[(np.zeros(3, dtype=np.float64), desired_elbow_position, 9.0)],
                max_nfev=100,
            )
            if frame_index > 0:
                arm_solution = _limit_joint_step(arm_solution, prev_arm_solution[side], arm_max_step)
            prev_arm_solution[side] = arm_solution
            for joint_name, value in zip(TARGET_ARM_JOINTS[side], arm_solution):
                target_joint_dict[joint_name] = float(value)

        target_dof_pos[frame_index] = np.array([target_joint_dict[name] for name in target_dof_names], dtype=np.float64)

    if root_smoothing_window > 1:
        target_root_pos = _moving_average(target_root_pos, root_smoothing_window)
    if is_stand_still:
        stable_root = np.median(target_root_pos, axis=0)
        target_root_pos = (1.0 - stand_root_position_blend) * target_root_pos + stand_root_position_blend * stable_root

    if arm_smoothing_window > 1:
        arm_indices = [
            target_dof_names.index(joint_name)
            for side in ("left", "right")
            for joint_name in TARGET_ARM_JOINTS[side]
        ]
        target_dof_pos[:, arm_indices] = _moving_average(target_dof_pos[:, arm_indices], arm_smoothing_window)

    key_body_pos = _compute_key_body_positions_world(
        target_root_pos,
        target_root_rot,
        target_tree,
        target_dof_names,
        target_dof_pos,
    )

    return {
        "fps": fps,
        "root_pos": target_root_pos.astype(np.float32),
        "root_rot": target_root_rot.astype(np.float32),
        "dof_pos": target_dof_pos.astype(np.float32),
        "loop_mode": int(motion_dict.get("loop_mode", 0)),
        "key_body_pos": key_body_pos,
        "dof_names": target_dof_names,
        "key_body_names": TARGET_KEY_BODY_NAMES,
        "root_rot_format": "wxyz",
        "source_dof_names": SOURCE_DOF_NAMES,
        "retarget_info": {
            "method": "scaled_ik_g1_to_lens110",
            "leg_scale": float(leg_scale),
            "stride_scale": float(stride_scale),
            "stride_scale_factor": float(stride_scale_factor),
            "arm_scale": float(arm_scale),
            "axis_signs": {joint_name: float(axis_signs[joint_name]) for joint_name in target_dof_names},
            "is_stand_still": bool(is_stand_still),
            "root_upright_blend": float(root_blend),
            "stand_root_position_blend": float(stand_root_position_blend if is_stand_still else 0.0),
            "root_smoothing_window": int(root_smoothing_window),
            "arm_smoothing_window": int(arm_smoothing_window),
            "leg_max_step": float(leg_max_step),
            "arm_max_step": float(arm_max_step),
            "source_config_file": str(G1_RETARGET_CONFIG),
        },
    }


def _iter_motion_files(source_root: Path):
    for motion_file in sorted(source_root.glob("*.pkl")):
        yield Path("."), motion_file


def _prepare_output_root(output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Retarget G1 motion data to Lens110 with scaling and IK.")
    parser.add_argument(
        "--source_root",
        default=str(REPO_ROOT / "source/legged_lab/legged_lab/data/MotionData/g1_lab"),
    )
    parser.add_argument(
        "--output_root",
        default=str(REPO_ROOT / "source/legged_lab/legged_lab/data/MotionData/lens110_lab"),
    )
    parser.add_argument(
        "--source_urdf",
        default=str(WORKSPACE_ROOT / "tools/retargeting/gmr_lens110/assets/unitree_g1/g1_custom_collision_29dof.urdf"),
    )
    parser.add_argument(
        "--target_urdf",
        default=str(REPO_ROOT / "source/legged_lab/legged_lab/data/Robots/model_humanoid_lens110/lens110_21dof.urdf"),
    )
    parser.add_argument(
        "--stride_scale_factor",
        type=float,
        default=0.9,
        help="Extra safety factor applied to horizontal root/foot stride scaling after leg_scale.",
    )
    parser.add_argument(
        "--root_upright_blend",
        type=float,
        default=0.10,
        help="Blend root roll/pitch toward upright while preserving yaw for moving clips.",
    )
    parser.add_argument(
        "--stand_root_upright_blend",
        type=float,
        default=0.92,
        help="Blend root roll/pitch toward upright for stand-still clips.",
    )
    parser.add_argument(
        "--stand_root_position_blend",
        type=float,
        default=0.85,
        help="Blend stand-still root positions toward their median to remove visible jitter.",
    )
    parser.add_argument(
        "--root_smoothing_window",
        type=int,
        default=5,
        help="Odd moving-average window for target root positions. Values <= 1 disable smoothing.",
    )
    parser.add_argument(
        "--arm_smoothing_window",
        type=int,
        default=5,
        help="Odd moving-average window for target arm joints. Values <= 1 disable smoothing.",
    )
    parser.add_argument(
        "--leg_max_step",
        type=float,
        default=0.18,
        help="Maximum leg IK joint change per frame in radians. Values <= 0 disable clamping.",
    )
    parser.add_argument(
        "--arm_max_step",
        type=float,
        default=0.10,
        help="Maximum arm IK joint change per frame in radians. Values <= 0 disable clamping.",
    )
    parser.add_argument(
        "--clean_output",
        action="store_true",
        help="Remove existing pkl files from the output directory before regenerating it.",
    )
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    source_tree = KinematicTree(Path(args.source_urdf))
    target_tree = KinematicTree(Path(args.target_urdf))
    target_dof_names = target_tree.joint_order

    if len(target_dof_names) != 21:
        raise ValueError(f"Expected 21 target DoFs, got {len(target_dof_names)}: {target_dof_names}")

    if args.clean_output:
        if output_root.is_dir() and not output_root.is_symlink():
            for motion_file in output_root.glob("*.pkl"):
                motion_file.unlink()
        elif output_root.is_symlink():
            output_root.unlink()

    _prepare_output_root(output_root)

    converted_count = 0
    for relative_dir, motion_file in _iter_motion_files(source_root):
        source_motion = _load_pickle(motion_file)
        target_motion = _retarget_motion(
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
        output_path = output_root / relative_dir / motion_file.name
        _save_pickle(output_path, target_motion)
        converted_count += 1
        print(f"[OK] {motion_file.name} -> {output_path}")

    print(f"[DONE] Converted {converted_count} motions into {output_root}")


if __name__ == "__main__":
    main()
