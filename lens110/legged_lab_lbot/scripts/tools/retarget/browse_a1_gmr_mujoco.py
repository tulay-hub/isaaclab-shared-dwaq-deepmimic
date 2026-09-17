"""Browse GMR motion pickle files with a simple Tk + MuJoCo UI.

This script defaults to:
    source/legged_lab/legged_lab/data/MotionData/lens110_deepminic_lab

It opens a small Tk file browser and a separate MuJoCo passive viewer window.
You must provide a compatible MuJoCo model path if auto-discovery cannot find one.

Examples:
    conda run -n gmr python legged_lab/scripts/tools/retarget/browse_a1_gmr_mujoco.py \
        --model /path/to/a1.xml

    MUJOCO_MODEL=/path/to/robot.xml \
    conda run -n gmr python legged_lab/scripts/tools/retarget/browse_a1_gmr_mujoco.py
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import mujoco
import numpy as np
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MOTION_ROOT = ROOT / "source/legged_lab/legged_lab/data/MotionData/lens110_deepminic_lab"
ROBOT_ROOT = ROOT / "source/legged_lab/legged_lab/data/Robots"
A1_ENV_MODEL = next(
    (Path(os.environ[name]).expanduser() for name in ("MUJOCO_MODEL", "A1_MUJOCO_MODEL") if os.environ.get(name)),
    None,
)

A1_DOF_NAMES = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]

ATOM01_GMR_DOF_NAMES = [
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

ATOM01_LAB_DOF_NAMES = [
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

ATOM02_GMR_DOF_NAMES = [
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
    "waist_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_joint",
]

ATOM02_LAB_DOF_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_roll_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_yaw_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_joint",
    "right_wrist_joint",
]

LENS110_DOF_NAMES = [
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
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]

MODEL_CANDIDATES = [
    ROBOT_ROOT / "model_humanoid_lens110/mjcf/lens110_21dof.xml",
    ROBOT_ROOT / "model_humanoid_lens110/lens110_21dof.urdf",
    ROBOT_ROOT / "atom01/mjcf/atom01.xml",
    ROBOT_ROOT / "atom01/urdf/atom01.urdf",
    ROBOT_ROOT / "atom02/mjcf/atom02_flat.xml",
    ROBOT_ROOT / "atom02/mjcf/robot.xml",
    ROBOT_ROOT / "atom02/urdf/atom02.urdf",
]


@dataclass(frozen=True)
class RobotProfile:
    label: str
    motion_root: Path
    model_candidates: tuple[Path, ...]
    root_rot_format: str
    fallback_dof_names: tuple[str, ...]


ROBOT_PROFILES = {
    "lens110": RobotProfile(
        label="lens110",
        motion_root=ROOT / "source/legged_lab/legged_lab/data/MotionData/lens110_lab",
        model_candidates=(
            ROBOT_ROOT / "model_humanoid_lens110/mjcf/lens110_21dof.xml",
            ROBOT_ROOT / "model_humanoid_lens110/lens110_21dof.urdf",
        ),
        root_rot_format="wxyz",
        fallback_dof_names=tuple(LENS110_DOF_NAMES),
    ),
    "atom01_long_lab": RobotProfile(
        label="atom01_long_lab",
        motion_root=ROOT / "source/legged_lab/legged_lab/data/MotionData/atom01_long_lab",
        model_candidates=(
            ROBOT_ROOT / "atom01_long_base_link/atom01_long_base_link.xml",
            ROBOT_ROOT / "atom01_long_base_link/atom01_long_base_link.urdf",
        ),
        root_rot_format="wxyz",
        fallback_dof_names=tuple(ATOM01_LAB_DOF_NAMES),
    ),
    "atom01_lab": RobotProfile(
        label="atom01_lab",
        motion_root=ROOT / "source/legged_lab/legged_lab/data/MotionData/atom01_lab",
        model_candidates=(
            ROBOT_ROOT / "atom01/mjcf/atom01.xml",
            ROBOT_ROOT / "atom01/urdf/atom01.urdf",
        ),
        root_rot_format="wxyz",
        fallback_dof_names=tuple(ATOM01_LAB_DOF_NAMES),
    ),
    "atom01_gmr": RobotProfile(
        label="atom01_gmr",
        motion_root=ROOT / "source/legged_lab/legged_lab/data/MotionData/atom01_gmr",
        model_candidates=(
            ROBOT_ROOT / "atom01/mjcf/atom01.xml",
            ROBOT_ROOT / "atom01/urdf/atom01.urdf",
        ),
        root_rot_format="xyzw",
        fallback_dof_names=tuple(ATOM01_GMR_DOF_NAMES),
    ),
    "atom02_lab": RobotProfile(
        label="atom02_lab",
        motion_root=ROOT / "source/legged_lab/legged_lab/data/MotionData/atom02_lab",
        model_candidates=(
            ROBOT_ROOT / "atom02/mjcf/atom02_flat.xml",
            ROBOT_ROOT / "atom02/mjcf/robot.xml",
            ROBOT_ROOT / "atom02/urdf/atom02.urdf",
        ),
        root_rot_format="wxyz",
        fallback_dof_names=tuple(ATOM02_LAB_DOF_NAMES),
    ),
    "atom02_gmr": RobotProfile(
        label="atom02_gmr",
        motion_root=ROOT / "source/legged_lab/legged_lab/data/MotionData/atom02_gmr",
        model_candidates=(
            ROBOT_ROOT / "atom02/mjcf/atom02_flat.xml",
            ROBOT_ROOT / "atom02/mjcf/robot.xml",
            ROBOT_ROOT / "atom02/urdf/atom02.urdf",
        ),
        root_rot_format="xyzw",
        fallback_dof_names=tuple(ATOM02_GMR_DOF_NAMES),
    ),
    "a1_gmr": RobotProfile(
        label="a1_gmr",
        motion_root=ROOT / "source/legged_lab/legged_lab/data/MotionData/a1_gmr",
        # No compatible A1 model is bundled in this Lens110 workspace.
        # Pass --model (or A1_MUJOCO_MODEL) explicitly when browsing A1 data.
        model_candidates=tuple(path for path in (A1_ENV_MODEL,) if path is not None),
        root_rot_format="xyzw",
        fallback_dof_names=tuple(A1_DOF_NAMES),
    ),
}


def _first_existing_path(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if str(path) and path.is_file():
            return path
    return None


def _profile_key_for_motion_root(motion_root: Path) -> str | None:
    motion_root_str = str(motion_root).lower()
    for key in ROBOT_PROFILES:
        if key in motion_root_str:
            return key
    if "lens110" in motion_root_str:
        return "lens110"
    if "atom01_long" in motion_root_str:
        return "atom01_long_lab" if "_lab" in motion_root_str else "atom01_gmr"
    if "atom01" in motion_root_str:
        return "atom01_lab" if "_lab" in motion_root_str else "atom01_gmr"
    if "atom02" in motion_root_str:
        return "atom02_lab" if "_lab" in motion_root_str else "atom02_gmr"
    if "a1" in motion_root_str:
        return "a1_gmr"
    return None


def _candidate_model_paths_for_motion_root(motion_root: Path) -> list[Path]:
    profile_key = _profile_key_for_motion_root(motion_root)
    if profile_key is not None:
        if profile_key == "a1_gmr":
            return list(ROBOT_PROFILES[profile_key].model_candidates)
        return list(ROBOT_PROFILES[profile_key].model_candidates) + MODEL_CANDIDATES

    motion_root_str = str(motion_root).lower()
    prioritized: list[Path] = []

    if "lens110" in motion_root_str:
        prioritized.extend(
            [
                ROBOT_ROOT / "model_humanoid_lens110/mjcf/lens110_21dof.xml",
                ROBOT_ROOT / "model_humanoid_lens110/lens110_21dof.urdf",
            ]
        )
    elif "atom01" in motion_root_str:
        prioritized.extend(
            [
                ROBOT_ROOT / "atom01/mjcf/atom01.xml",
                ROBOT_ROOT / "atom01/urdf/atom01.urdf",
                ROBOT_ROOT / "atom01_long_base_link/atom01_long_base_link.xml",
                ROBOT_ROOT / "atom01_long_base_link/atom01_long_base_link.urdf",
            ]
        )
    elif "atom02" in motion_root_str:
        prioritized.extend(
            [
                ROBOT_ROOT / "atom02/mjcf/atom02_flat.xml",
                ROBOT_ROOT / "atom02/mjcf/robot.xml",
                ROBOT_ROOT / "atom02/urdf/atom02.urdf",
            ]
        )

    prioritized.extend(MODEL_CANDIDATES)

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for path in prioritized:
        if path in seen:
            continue
        seen.add(path)
        unique_candidates.append(path)
    return unique_candidates


def _fallback_dof_names_for_motion(motion_path: Path, dof_dim: int) -> list[str]:
    profile_key = _profile_key_for_motion_root(motion_path.parent)
    if profile_key is not None:
        names = list(ROBOT_PROFILES[profile_key].fallback_dof_names)
        if len(names) == dof_dim:
            return names

    if dof_dim == len(A1_DOF_NAMES):
        return list(A1_DOF_NAMES)
    if dof_dim == len(LENS110_DOF_NAMES):
        return list(LENS110_DOF_NAMES)
    if dof_dim == len(ATOM02_LAB_DOF_NAMES):
        return list(ATOM02_LAB_DOF_NAMES)
    if dof_dim == len(ATOM01_LAB_DOF_NAMES):
        return list(ATOM01_LAB_DOF_NAMES)

    raise ValueError(
        f"Motion file has no dof_names/joint_names and no fallback is known for {dof_dim} DOFs: {motion_path}"
    )


def _install_numpy_pickle_shim() -> None:
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


def _normalize_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    norm = np.linalg.norm(quat_wxyz)
    if norm < 1.0e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat_wxyz / norm


def _xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return quat_xyzw[..., [3, 0, 1, 2]]


def _infer_root_rot_wxyz(root_rot: np.ndarray, root_rot_format: str | None) -> np.ndarray:
    root_rot = np.asarray(root_rot, dtype=np.float64)
    if root_rot.ndim != 2 or root_rot.shape[1] != 4:
        raise ValueError(f"Expected root_rot shape (num_frames, 4), got {root_rot.shape}")

    fmt = (root_rot_format or "auto").lower()
    if fmt == "wxyz":
        return root_rot
    if fmt == "xyzw":
        return _xyzw_to_wxyz(root_rot)

    mean_first = float(np.mean(np.abs(root_rot[:, 0])))
    mean_last = float(np.mean(np.abs(root_rot[:, 3])))
    if mean_first >= mean_last:
        return root_rot
    return _xyzw_to_wxyz(root_rot)


def _build_joint_qpos_map(model: mujoco.MjModel, dof_names: list[str]) -> np.ndarray:
    qpos_indices: list[int] = []
    for joint_name in dof_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint '{joint_name}' not found in model.")
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"Joint '{joint_name}' is not a hinge joint.")
        qpos_indices.append(int(model.jnt_qposadr[joint_id]))
    return np.asarray(qpos_indices, dtype=np.int32)


@dataclass
class MotionClip:
    path: Path
    frames: int
    fps: float
    root_pos: np.ndarray
    root_rot_wxyz: np.ndarray
    dof_pos: np.ndarray
    dof_names: list[str]
    qpos_indices: np.ndarray
    motion_dict: dict[str, Any]


def load_motion_clip(model: mujoco.MjModel, motion_path: Path, root_rot_format_override: str | None) -> MotionClip:
    _install_numpy_pickle_shim()

    try:
        motion = joblib.load(motion_path)
    except Exception:
        with motion_path.open("rb") as file:
            motion = pickle.load(file)

    if not isinstance(motion, dict):
        raise ValueError(f"Unsupported motion object type: {type(motion)!r}")
    if not {"root_pos", "root_rot", "dof_pos"}.issubset(motion.keys()):
        raise ValueError("Motion file must contain root_pos, root_rot, and dof_pos.")

    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    root_rot = _infer_root_rot_wxyz(
        np.asarray(motion["root_rot"], dtype=np.float64),
        root_rot_format_override or motion.get("root_rot_format"),
    )
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)
    fps = float(motion.get("fps", 30.0))

    raw_dof_names = motion.get("dof_names")
    if raw_dof_names is None:
        raw_dof_names = motion.get("joint_names")

    if dof_pos.ndim != 2:
        raise ValueError(f"Expected dof_pos shape (num_frames, num_dofs), got {dof_pos.shape}")

    if raw_dof_names is None:
        dof_names = _fallback_dof_names_for_motion(motion_path, int(dof_pos.shape[1]))
    else:
        dof_names = [str(name) for name in raw_dof_names]

    if len(dof_names) != dof_pos.shape[1]:
        raise ValueError(
            f"dof_names length {len(dof_names)} does not match dof_pos width {dof_pos.shape[1]} for {motion_path}"
        )

    qpos_indices = _build_joint_qpos_map(model, dof_names)
    return MotionClip(
        path=motion_path,
        frames=int(root_pos.shape[0]),
        fps=fps,
        root_pos=root_pos,
        root_rot_wxyz=root_rot,
        dof_pos=dof_pos,
        dof_names=dof_names,
        qpos_indices=qpos_indices,
        motion_dict=motion,
    )


class MotionPlayer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.clip: MotionClip | None = None
        self.viewer_alive = False

        self.current_frame = 0
        self.paused = True
        self.loop = True
        self.speed = 1.0
        self.z_offset = 0.0
        self.track_camera = False

    def load(self, model_path: Path, motion_path: Path, root_rot_format_override: str | None) -> None:
        self.stop()

        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        clip = load_motion_clip(model, motion_path, root_rot_format_override)

        with self._lock:
            self.model = model
            self.data = data
            self.clip = clip
            self.current_frame = 0
            self.paused = True
            self.viewer_alive = False

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None
        with self._lock:
            self.viewer_alive = False

    def is_loaded(self) -> bool:
        with self._lock:
            return self.clip is not None

    def toggle_pause(self) -> None:
        with self._lock:
            self.paused = not self.paused

    def set_loop(self, enabled: bool) -> None:
        with self._lock:
            self.loop = enabled

    def set_track_camera(self, enabled: bool) -> None:
        with self._lock:
            self.track_camera = enabled

    def set_speed(self, speed: float) -> None:
        with self._lock:
            self.speed = max(0.05, float(speed))

    def set_z_offset(self, z_offset: float) -> None:
        with self._lock:
            self.z_offset = float(z_offset)

    def set_frame(self, frame_index: int) -> None:
        with self._lock:
            if self.clip is None:
                return
            self.current_frame = int(np.clip(frame_index, 0, self.clip.frames - 1))

    def step(self, delta: int) -> None:
        with self._lock:
            if self.clip is None:
                return
            next_frame = self.current_frame + int(delta)
            self.current_frame = int(np.clip(next_frame, 0, self.clip.frames - 1))
            self.paused = True

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            clip = self.clip
            return {
                "loaded": clip is not None,
                "viewer_alive": self.viewer_alive,
                "current_frame": self.current_frame,
                "paused": self.paused,
                "loop": self.loop,
                "speed": self.speed,
                "z_offset": self.z_offset,
                "track_camera": self.track_camera,
                "frames": 0 if clip is None else clip.frames,
                "fps": 0.0 if clip is None else clip.fps,
                "motion_path": None if clip is None else clip.path,
            }

    def _run(self) -> None:
        assert self.model is not None
        assert self.data is not None
        assert self.clip is not None

        model = self.model
        data = self.data
        clip = self.clip

        import mujoco.viewer

        with mujoco.viewer.launch_passive(model, data) as viewer:
            with self._lock:
                self.viewer_alive = True
            self._configure_camera(viewer)

            next_tick = time.perf_counter()
            while viewer.is_running() and not self._stop_event.is_set():
                with self._lock:
                    frame_index = self.current_frame
                    paused = self.paused
                    loop = self.loop
                    speed = self.speed
                    z_offset = self.z_offset
                    track_camera = self.track_camera

                self._write_frame(data, clip, frame_index, z_offset)
                mujoco.mj_forward(model, data)

                if track_camera:
                    self._configure_camera(viewer)
                else:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

                viewer.sync()

                if not paused:
                    with self._lock:
                        if self.clip is None:
                            break
                        if self.current_frame >= self.clip.frames - 1:
                            if loop:
                                self.current_frame = 0
                            else:
                                self.current_frame = self.clip.frames - 1
                                self.paused = True
                        else:
                            self.current_frame += 1

                frame_dt = 1.0 / max(1.0e-6, clip.fps * speed)
                next_tick += frame_dt
                sleep_time = next_tick - time.perf_counter()
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
                else:
                    next_tick = time.perf_counter()

        with self._lock:
            self.viewer_alive = False

    @staticmethod
    def _write_frame(data: mujoco.MjData, clip: MotionClip, frame_index: int, z_offset: float) -> None:
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        if data.ctrl is not None:
            data.ctrl[:] = 0.0

        data.qpos[0:3] = clip.root_pos[frame_index] + np.array([0.0, 0.0, z_offset], dtype=np.float64)
        data.qpos[3:7] = _normalize_quat_wxyz(clip.root_rot_wxyz[frame_index])
        data.qpos[clip.qpos_indices] = clip.dof_pos[frame_index]

    @staticmethod
    def _configure_camera(viewer: Any) -> None:
        cam = viewer.cam
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = 0
        cam.distance = 2.5
        cam.azimuth = 135.0
        cam.elevation = -18.0


class MotionBrowserApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.root = tk.Tk()
        self.root.title("Motion MuJoCo Browser")
        self.root.geometry("1180x760")

        self.player = MotionPlayer()
        self._all_files: list[Path] = []
        self._filtered_files: list[Path] = []
        self._slider_dragging = False
        self._curve_channels: dict[str, np.ndarray] = {}

        profile = ROBOT_PROFILES[args.robot]
        model_path = args.model if args.model else _first_existing_path(list(profile.model_candidates))
        motion_root = args.motion_root if args.motion_root else profile.motion_root

        self.robot_var = tk.StringVar(value=args.robot)
        self.model_var = tk.StringVar(value=str(model_path) if model_path else "")
        self.motion_root_var = tk.StringVar(value=str(motion_root))
        self.filter_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready.")
        self.file_info_var = tk.StringVar(value="No motion selected.")
        self.speed_var = tk.StringVar(value="1.0")
        self.loop_var = tk.BooleanVar(value=True)
        self.track_cam_var = tk.BooleanVar(value=bool(args.track_camera))
        self.z_offset_var = tk.DoubleVar(value=0.0)
        self.root_rot_format_var = tk.StringVar(value=args.root_rot_format or profile.root_rot_format)
        self.frame_label_var = tk.StringVar(value="Frame 0 / 0")
        self.curve_channel_var = tk.StringVar(value="root_z")

        self._configure_style()
        self._build_ui()
        self._bind_events()
        self.refresh_file_list()
        self._poll_player()

        if not self.model_var.get():
            self.status_var.set("No model found automatically. Set --model or MUJOCO_MODEL.")

    def _configure_style(self) -> None:
        ui_font = ("DejaVu Sans", 11)
        mono_font = ("DejaVu Sans Mono", 10)
        title_font = ("DejaVu Sans", 11, "bold")
        small_font = ("DejaVu Sans", 10)

        self.root.option_add("*Font", ui_font)
        self.root.configure(bg="#eef2f7")

        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(family=ui_font[0], size=ui_font[1])
        fixed_font = tkfont.nametofont("TkFixedFont")
        fixed_font.configure(family=mono_font[0], size=mono_font[1])
        heading_font = tkfont.nametofont("TkHeadingFont")
        heading_font.configure(family=title_font[0], size=title_font[1], weight=title_font[2])

        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background="#eef2f7", foreground="#18212f", fieldbackground="#ffffff")
        style.configure("TFrame", background="#eef2f7")
        style.configure("TLabelframe", background="#eef2f7", borderwidth=1, relief="solid")
        style.configure("TLabelframe.Label", background="#eef2f7", foreground="#0f172a", font=title_font)
        style.configure("TLabel", background="#eef2f7", foreground="#18212f", font=ui_font)
        style.configure("Info.TLabel", background="#eef2f7", foreground="#475569", font=small_font)
        style.configure("TButton", background="#d8e1ec", foreground="#0f172a", padding=(10, 6), relief="flat")
        style.configure("Accent.TButton", background="#3b82f6", foreground="#ffffff", padding=(12, 6), relief="flat")

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        top = ttk.Frame(self.root, padding=10)
        top.grid(row=0, column=0, sticky="nsew")
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Robot").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.robot_box = ttk.Combobox(
            top,
            textvariable=self.robot_var,
            values=list(ROBOT_PROFILES.keys()),
            state="readonly",
            width=22,
        )
        self.robot_box.grid(row=0, column=1, sticky="w")
        ttk.Button(top, text="Apply Robot", command=self._apply_robot_profile).grid(row=0, column=2, padx=(8, 0))

        ttk.Label(top, text="Model XML/URDF").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        model_entry = ttk.Entry(top, textvariable=self.model_var)
        model_entry.grid(row=1, column=1, sticky="ew", pady=(8, 0))
        ttk.Button(top, text="Browse", command=self._browse_model).grid(row=1, column=2, padx=(8, 0), pady=(8, 0))

        ttk.Label(top, text="Motion Root").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        root_entry = ttk.Entry(top, textvariable=self.motion_root_var)
        root_entry.grid(row=2, column=1, sticky="ew", pady=(8, 0))
        ttk.Button(top, text="Browse", command=self._browse_motion_root).grid(row=2, column=2, padx=(8, 0), pady=(8, 0))
        ttk.Button(top, text="Refresh", command=self.refresh_file_list).grid(row=2, column=3, padx=(8, 0), pady=(8, 0))

        ttk.Label(top, text="Filter").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        filter_entry = ttk.Entry(top, textvariable=self.filter_var)
        filter_entry.grid(row=3, column=1, sticky="ew", pady=(8, 0))
        self.filter_entry = filter_entry
        ttk.Button(top, text="Clear", command=lambda: self.filter_var.set("")).grid(row=3, column=2, padx=(8, 0), pady=(8, 0))

        ttk.Label(top, text="Root Rot").grid(row=4, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        root_rot_format_box = ttk.Combobox(
            top,
            textvariable=self.root_rot_format_var,
            values=["auto", "wxyz", "xyzw"],
            state="readonly",
            width=18,
        )
        root_rot_format_box.grid(row=4, column=1, sticky="w", pady=(8, 0))
        ttk.Label(top, text="GMR usually uses xyzw; lab data usually uses wxyz").grid(
            row=4, column=2, columnspan=2, sticky="w", pady=(8, 0)
        )

        center = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        center.grid(row=1, column=0, sticky="nsew")
        center.columnconfigure(0, weight=2)
        center.columnconfigure(1, weight=5)
        center.rowconfigure(0, weight=1)

        left = ttk.LabelFrame(center, text="Motion Files", padding=8)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)

        self.file_listbox = tk.Listbox(
            left,
            exportselection=False,
            bg="#ffffff",
            fg="#0f172a",
            selectbackground="#dbeafe",
            selectforeground="#0f172a",
            activestyle="none",
            font=("DejaVu Sans Mono", 10),
            highlightthickness=1,
            relief="flat",
            width=28,
        )
        self.file_listbox.grid(row=0, column=0, sticky="nsew")
        file_scroll = ttk.Scrollbar(left, orient="vertical", command=self.file_listbox.yview)
        file_scroll.grid(row=0, column=1, sticky="ns")
        self.file_listbox.config(yscrollcommand=file_scroll.set)

        file_buttons = ttk.Frame(left)
        file_buttons.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(file_buttons, text="Load Selected", style="Accent.TButton", command=self.load_selected_motion).pack(side="left")
        ttk.Button(file_buttons, text="Load Next", command=lambda: self._load_relative_selection(1)).pack(side="left", padx=(8, 0))
        ttk.Button(file_buttons, text="Load Prev", command=lambda: self._load_relative_selection(-1)).pack(side="left", padx=(8, 0))

        right = ttk.LabelFrame(center, text="Frame View", padding=8)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        self.curve_summary_var = tk.StringVar(value="Load a motion to inspect its frame curve.")
        ttk.Label(right, textvariable=self.curve_summary_var, style="Info.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )

        self.curve_canvas = tk.Canvas(
            right,
            bg="#ffffff",
            highlightthickness=1,
            highlightbackground="#cbd5e1",
            height=360,
        )
        self.curve_canvas.grid(row=1, column=0, sticky="nsew")
        self.curve_canvas.bind("<Configure>", self._redraw_curve_view)

        curve_bar = ttk.Frame(right)
        curve_bar.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        curve_bar.columnconfigure(1, weight=1)

        ttk.Label(curve_bar, text="Joint / Signal").grid(row=0, column=0, sticky="w")
        self.curve_channel_box = ttk.Combobox(
            curve_bar,
            textvariable=self.curve_channel_var,
            values=["root_z"],
            state="readonly",
            width=44,
        )
        self.curve_channel_box.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        controls = ttk.LabelFrame(self.root, text="Playback", padding=10)
        controls.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        controls.columnconfigure(7, weight=1)

        self.play_button = ttk.Button(controls, text="Play", style="Accent.TButton", command=self.toggle_play)
        self.play_button.grid(row=0, column=0, sticky="w")
        ttk.Button(controls, text="Prev Frame", command=lambda: self.player.step(-1)).grid(row=0, column=1, padx=(8, 0))
        ttk.Button(controls, text="Next Frame", command=lambda: self.player.step(1)).grid(row=0, column=2, padx=(8, 0))

        ttk.Label(controls, text="Speed").grid(row=0, column=3, padx=(16, 8))
        self.speed_box = ttk.Combobox(
            controls,
            textvariable=self.speed_var,
            values=["0.25", "0.5", "1.0", "1.5", "2.0", "4.0"],
            width=6,
            state="readonly",
        )
        self.speed_box.grid(row=0, column=4, sticky="w")

        ttk.Checkbutton(controls, text="Loop", variable=self.loop_var, command=self._sync_playback_options).grid(row=0, column=5, padx=(16, 0))
        ttk.Checkbutton(controls, text="Track Camera", variable=self.track_cam_var, command=self._sync_playback_options).grid(row=0, column=6, padx=(8, 0))

        z_frame = ttk.Frame(controls)
        z_frame.grid(row=0, column=7, sticky="e")
        ttk.Label(z_frame, text="Z Offset").pack(side="left")
        self.z_spinbox = ttk.Spinbox(
            z_frame,
            from_=-1.0,
            to=2.0,
            increment=0.01,
            width=8,
            textvariable=self.z_offset_var,
            command=self._sync_playback_options,
        )
        self.z_spinbox.pack(side="left", padx=(8, 0))

        self.frame_scale = tk.Scale(
            controls,
            from_=0,
            to=0,
            orient="horizontal",
            showvalue=False,
            resolution=1,
            command=self._on_slider_changed,
            bg="#eef2f7",
            fg="#18212f",
            highlightthickness=0,
            troughcolor="#d8e1ec",
        )
        self.frame_scale.grid(row=1, column=0, columnspan=8, sticky="ew", pady=(10, 0))
        self.frame_scale.bind("<ButtonPress-1>", self._on_slider_press)
        self.frame_scale.bind("<ButtonRelease-1>", self._on_slider_release)

        ttk.Label(controls, textvariable=self.frame_label_var).grid(row=2, column=0, columnspan=4, sticky="w")

        status_bar = ttk.Label(self.root, textvariable=self.status_var, style="Info.TLabel", anchor="w", padding=(10, 6))
        status_bar.grid(row=3, column=0, sticky="ew")

    def _bind_events(self) -> None:
        self.filter_var.trace_add("write", lambda *_args: self._apply_filter())
        self.motion_root_var.trace_add("write", lambda *_args: self._sync_model_from_motion_root())
        self.robot_box.bind("<<ComboboxSelected>>", lambda _event: self._apply_robot_profile())
        self.file_listbox.bind("<Double-Button-1>", lambda _event: self.load_selected_motion())
        self.speed_box.bind("<<ComboboxSelected>>", lambda _event: self._sync_playback_options())
        self.z_spinbox.bind("<Return>", lambda _event: self._sync_playback_options())
        self.curve_channel_box.bind("<<ComboboxSelected>>", lambda _event: self._redraw_curve_view())
        self.root.bind("<space>", lambda _event: self.toggle_play())
        self.root.bind("<Left>", lambda _event: self.player.step(-1))
        self.root.bind("<Right>", lambda _event: self.player.step(1))
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _apply_robot_profile(self) -> None:
        profile = ROBOT_PROFILES[self.robot_var.get()]
        auto_model = _first_existing_path(list(profile.model_candidates))
        if auto_model is not None:
            self.model_var.set(str(auto_model))
        self.motion_root_var.set(str(profile.motion_root))
        self.root_rot_format_var.set(profile.root_rot_format)
        self.refresh_file_list()
        self.status_var.set(f"Robot profile applied: {profile.label}")

    def _sync_model_from_motion_root(self) -> None:
        motion_root = Path(self.motion_root_var.get()).expanduser()
        profile_key = _profile_key_for_motion_root(motion_root)
        if profile_key is not None:
            self.robot_var.set(profile_key)
            self.root_rot_format_var.set(ROBOT_PROFILES[profile_key].root_rot_format)
        auto_model = _first_existing_path(_candidate_model_paths_for_motion_root(motion_root))
        if auto_model is not None:
            self.model_var.set(str(auto_model))

    def _browse_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Select MuJoCo XML or URDF",
            filetypes=[("MuJoCo or URDF", "*.xml *.urdf"), ("All files", "*.*")],
        )
        if path:
            self.model_var.set(path)

    def _browse_motion_root(self) -> None:
        path = filedialog.askdirectory(title="Select Motion Root")
        if path:
            self.motion_root_var.set(path)
            self.refresh_file_list()

    def refresh_file_list(self) -> None:
        motion_root = Path(self.motion_root_var.get()).expanduser()
        if not motion_root.is_dir():
            self._all_files = []
            self._filtered_files = []
            self.file_listbox.delete(0, tk.END)
            self.status_var.set(f"Motion root not found: {motion_root}")
            return

        self._all_files = sorted(motion_root.rglob("*.pkl"))
        self._apply_filter()
        self.status_var.set(f"Found {len(self._all_files)} pkl files under {motion_root}")

    def _apply_filter(self) -> None:
        pattern = self.filter_var.get().strip().lower()
        motion_root = Path(self.motion_root_var.get()).expanduser()
        self.file_listbox.delete(0, tk.END)

        if not pattern:
            self._filtered_files = list(self._all_files)
        else:
            self._filtered_files = [path for path in self._all_files if pattern in str(path.relative_to(motion_root)).lower()]

        for path in self._filtered_files:
            self.file_listbox.insert(tk.END, str(path.relative_to(motion_root)))

        if self._filtered_files:
            self.file_listbox.selection_set(0)
        self.status_var.set(f"Showing {len(self._filtered_files)} / {len(self._all_files)} files")

    def _selected_motion_path(self) -> Path | None:
        selection = self.file_listbox.curselection()
        if not selection:
            return None
        return self._filtered_files[int(selection[0])]

    def _load_relative_selection(self, delta: int) -> None:
        if not self._filtered_files:
            return
        selection = self.file_listbox.curselection()
        index = 0 if not selection else int(selection[0])
        next_index = int(np.clip(index + delta, 0, len(self._filtered_files) - 1))
        self.file_listbox.selection_clear(0, tk.END)
        self.file_listbox.selection_set(next_index)
        self.file_listbox.see(next_index)
        self.load_selected_motion()

    def load_selected_motion(self) -> None:
        motion_path = self._selected_motion_path()
        if motion_path is None:
            messagebox.showwarning("No Selection", "Select a motion file first.")
            return

        model_path = Path(self.model_var.get()).expanduser()
        if not model_path.is_file():
            messagebox.showerror(
                "Model Missing",
                "Model XML/URDF not found.\n\n"
                "Please set --model, choose one with the Browse button, or export MUJOCO_MODEL.",
            )
            return

        root_rot_format_override = self.root_rot_format_var.get().strip() or "auto"
        self.status_var.set(f"Loading {motion_path.name} ...")
        self.root.update_idletasks()

        try:
            self.player.load(model_path=model_path, motion_path=motion_path, root_rot_format_override=root_rot_format_override)
            self._sync_playback_options()
            self._update_motion_info()
            self.status_var.set(f"Loaded {motion_path}")
        except Exception as exc:
            messagebox.showerror("Load Failed", f"{exc}")
            self.status_var.set(f"Failed to load {motion_path.name}")

    def _update_motion_info(self) -> None:
        snapshot = self.player.snapshot()
        clip = self.player.clip
        if clip is None:
            self._curve_channels = {}
            self.curve_summary_var.set("No motion loaded.")
            self._redraw_curve_view()
            return

        root_z = clip.root_pos[:, 2]
        self._curve_channels = self._build_curve_channels(clip)
        self.curve_channel_box.configure(values=list(self._curve_channels.keys()))
        if self.curve_channel_var.get() not in self._curve_channels:
            self.curve_channel_var.set("root_z" if "root_z" in self._curve_channels else next(iter(self._curve_channels)))
        self.curve_summary_var.set(
            f"{clip.path.name} | frames {clip.frames} | fps {clip.fps:.2f} | root_z [{float(np.min(root_z)):.3f}, {float(np.max(root_z)):.3f}]"
        )
        self._redraw_curve_view()
        self.frame_scale.configure(to=max(0, clip.frames - 1))
        self.frame_label_var.set(f"Frame 0 / {clip.frames - 1}")
        self.file_info_var.set(f"{clip.path.name} | {clip.fps:.2f} fps")

    def _build_curve_channels(self, clip: MotionClip) -> dict[str, np.ndarray]:
        channels: dict[str, np.ndarray] = {
            "root_x": clip.root_pos[:, 0],
            "root_y": clip.root_pos[:, 1],
            "root_z": clip.root_pos[:, 2],
            "root_rot_w": clip.root_rot_wxyz[:, 0],
            "root_rot_x": clip.root_rot_wxyz[:, 1],
            "root_rot_y": clip.root_rot_wxyz[:, 2],
            "root_rot_z": clip.root_rot_wxyz[:, 3],
        }
        for idx, name in enumerate(clip.dof_names):
            channels[name] = clip.dof_pos[:, idx]
        return channels

    def _redraw_curve_view(self, _event: tk.Event | None = None) -> None:
        canvas = getattr(self, "curve_canvas", None)
        clip = self.player.clip
        if canvas is None or clip is None:
            return
        canvas.delete("all")

        width = max(1, int(canvas.winfo_width()))
        height = max(1, int(canvas.winfo_height()))
        pad_l, pad_r, pad_t, pad_b = 48, 16, 20, 30
        plot_w = max(1, width - pad_l - pad_r)
        plot_h = max(1, height - pad_t - pad_b)

        channel_name = self.curve_channel_var.get()
        values = self._curve_channels.get(channel_name)
        if values is None:
            canvas.create_text(width // 2, height // 2, text="No curve selected", fill="#64748b")
            return
        if len(values) == 0:
            canvas.create_text(width // 2, height // 2, text="No curve selected", fill="#64748b")
            return

        v_min = float(np.min(values))
        v_max = float(np.max(values))
        if np.isclose(v_min, v_max):
            v_min -= 1.0
            v_max += 1.0
        margin = 0.1 * (v_max - v_min)
        v_min -= margin
        v_max += margin

        def map_xy(frame_idx: int, value: float) -> tuple[float, float]:
            if len(values) == 1:
                x = pad_l + plot_w / 2
            else:
                x = pad_l + plot_w * frame_idx / (len(values) - 1)
            y = pad_t + plot_h * (1.0 - (value - v_min) / (v_max - v_min))
            return x, y

        canvas.create_rectangle(pad_l, pad_t, width - pad_r, height - pad_b, outline="#cbd5e1")

        for frac in np.linspace(0.0, 1.0, 5):
            y = pad_t + plot_h * frac
            value = v_max - (v_max - v_min) * frac
            canvas.create_line(pad_l, y, width - pad_r, y, fill="#e2e8f0")
            canvas.create_text(pad_l - 6, y, text=f"{value:.2f}", anchor="e", fill="#64748b", font=("DejaVu Sans Mono", 9))

        for frac in np.linspace(0.0, 1.0, 5):
            x = pad_l + plot_w * frac
            frame = int(round((len(values) - 1) * frac))
            canvas.create_line(x, pad_t, x, height - pad_b, fill="#e2e8f0")
            canvas.create_text(x, height - pad_b + 10, text=str(frame), anchor="n", fill="#64748b", font=("DejaVu Sans Mono", 9))

        points = []
        for i, value in enumerate(values):
            points.extend(map_xy(i, float(value)))
        if len(points) >= 4:
            canvas.create_line(*points, fill="#2563eb", width=2, smooth=False)

        frame = int(self.frame_scale.get()) if self._slider_dragging else int(self.player.current_frame)
        frame = int(np.clip(frame, 0, len(values) - 1))
        x, y = map_xy(frame, float(values[frame]))
        canvas.create_line(x, pad_t, x, height - pad_b, fill="#ef4444", dash=(5, 3), width=2)
        canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#ef4444", outline="#ef4444")
        canvas.create_text(
            width // 2,
            10,
            text=f"{channel_name} | frame {frame}/{len(values) - 1} | value {float(values[frame]):.4f}",
            fill="#0f172a",
            anchor="n",
            font=("DejaVu Sans", 10, "bold"),
        )

    def _sync_playback_options(self) -> None:
        try:
            self.player.set_speed(float(self.speed_var.get()))
        except ValueError:
            self.speed_var.set("1.0")
            self.player.set_speed(1.0)
        self.player.set_loop(bool(self.loop_var.get()))
        self.player.set_track_camera(bool(self.track_cam_var.get()))
        self.player.set_z_offset(float(self.z_offset_var.get()))

    def toggle_play(self) -> None:
        if not self.player.is_loaded():
            return
        self.player.toggle_pause()

    def _on_slider_press(self, _event: tk.Event) -> None:
        self._slider_dragging = True

    def _on_slider_release(self, _event: tk.Event) -> None:
        self._slider_dragging = False
        self.player.set_frame(int(self.frame_scale.get()))

    def _on_slider_changed(self, value: str) -> None:
        if self._slider_dragging:
            self.frame_label_var.set(f"Frame {int(float(value))} / {int(self.frame_scale.cget('to'))}")
            self._redraw_curve_view()

    def _poll_player(self) -> None:
        snapshot = self.player.snapshot()
        if snapshot["loaded"]:
            current_frame = int(snapshot["current_frame"])
            total_frames = int(snapshot["frames"])
            if not self._slider_dragging:
                self.frame_scale.set(current_frame)
                self.frame_label_var.set(f"Frame {current_frame} / {max(0, total_frames - 1)}")
            self.play_button.configure(text="Play" if snapshot["paused"] else "Pause")
            self._redraw_curve_view()
        else:
            self.play_button.configure(text="Play")
        self.root.after(40, self._poll_player)

    def on_close(self) -> None:
        self.player.stop()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browse motion pkl files with a simple Tk + MuJoCo UI.")
    default_robot = "lens110"
    default_profile = ROBOT_PROFILES[default_robot]
    parser.add_argument(
        "--robot",
        choices=list(ROBOT_PROFILES.keys()),
        default=default_robot,
        help="Robot/data profile used to auto-fill model, motion root, root rotation format, and unnamed DOF order.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="MuJoCo XML or URDF compatible with the selected motion root.",
    )
    parser.add_argument(
        "--motion-root",
        type=Path,
        default=None,
        help=f"Directory containing motion pkl files. Defaults to the selected robot profile, e.g. {default_profile.motion_root}.",
    )
    parser.add_argument(
        "--root-rot-format",
        choices=["auto", "wxyz", "xyzw"],
        default=None,
        help="Quaternion convention stored in root_rot.",
    )
    parser.add_argument(
        "--track-camera",
        action="store_true",
        help="Start with the MuJoCo camera locked to the robot. By default the camera is free so you can rotate it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = MotionBrowserApp(args)
    app.run()


if __name__ == "__main__":
    main()
