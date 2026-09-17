#!/usr/bin/env python3
"""Convert T800 npy motion to npz format for whole_body_tracking.

NPY format (from engineaimuaythailab):
  0-2:   base_pos (x, y, z)
  3-6:   base_quat (w, x, y, z)
  7-31:  joint_pos (25 joints, J00..J28)
  32-35: contacts (left_toe, left_heel, right_toe, right_heel)

Usage:
    python scripts/npy_to_npz.py -i /path/to/motion.npy -o output.npz
"""

import argparse
import io
import os
import sys
import zipfile
from pathlib import Path
from typing import List, Optional, Set

import numpy as np


def _candidate_engineai_roots() -> List[Path]:
    """Return likely engineaimuaythailab checkout locations."""
    script_path = Path(__file__).resolve()
    candidates: List[Path] = []

    env_root = os.environ.get("ENGINEAI_LAB_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())

    # Check every parent directory and its parent for a sibling repo.
    for parent in script_path.parents:
        candidates.append(parent / "engineaimuaythailab")
        if parent.parent != parent:
            candidates.append(parent.parent / "engineaimuaythailab")

    # Deduplicate while preserving order.
    deduped: List[Path] = []
    seen: Set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve(strict=False)
        if candidate not in seen:
            seen.add(candidate)
            deduped.append(candidate)
    return deduped


def _resolve_engineai_lab(explicit_root: Optional[str]) -> Path:
    """Locate the engineaimuaythailab checkout that provides T800MotionLoader."""
    candidates: List[Path] = []
    if explicit_root:
        candidates.append(Path(explicit_root).expanduser())
    candidates.extend(_candidate_engineai_roots())

    for root in candidates:
        source_dir = root / "source"
        loader_file = source_dir / "dataset_utils" / "T800_motion_loader.py"
        if loader_file.exists():
            if str(source_dir) not in sys.path:
                sys.path.insert(0, str(source_dir))
            dataset_utils_dir = source_dir / "dataset_utils"
            if str(dataset_utils_dir) not in sys.path:
                sys.path.insert(0, str(dataset_utils_dir))
            return root

    formatted_candidates = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "engineaimuaythailab not found.\n"
        "Set --engineai_lab /path/to/engineaimuaythailab or export ENGINEAI_LAB_ROOT.\n"
        f"Tried:\n{formatted_candidates}"
    )

# T800 body order: this repository only supports DFS.
T800_BODY_ORDER = [
    "LINK_BASE", "LINK_HIP_PITCH_L", "LINK_HIP_ROLL_L", "LINK_HIP_YAW_L", "LINK_KNEE_PITCH_L",
    "LINK_ANKLE_PITCH_L", "LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_L_TOE", "LINK_ANKLE_ROLL_L_HEEL",
    "LINK_HIP_PITCH_R", "LINK_HIP_ROLL_R", "LINK_HIP_YAW_R", "LINK_KNEE_PITCH_R",
    "LINK_ANKLE_PITCH_R", "LINK_ANKLE_ROLL_R", "LINK_ANKLE_ROLL_R_TOE", "LINK_ANKLE_ROLL_R_HEEL",
    "LINK_TORSO_YAW", "LINK_SHOULDER_PITCH_L", "LINK_SHOULDER_ROLL_L", "LINK_SHOULDER_YAW_L",
    "LINK_ELBOW_PITCH_L", "LINK_ELBOW_YAW_L", "LINK_WRIST_PITCH_L", "LINK_WRIST_ROLL_L",
    "LINK_SHOULDER_PITCH_R", "LINK_SHOULDER_ROLL_R", "LINK_SHOULDER_YAW_R",
    "LINK_ELBOW_PITCH_R", "LINK_ELBOW_YAW_R", "LINK_WRIST_PITCH_R", "LINK_WRIST_ROLL_R",
    "LINK_HEAD_PITCH", "LINK_HEAD_YAW",
]

# T800MotionLoader LINK_NAMES (FK output order)
FK_LINK_NAMES = [
    "LINK_BASE",
    "LINK_HIP_ROLL_L",
    "LINK_HIP_YAW_L",
    "LINK_HIP_PITCH_L",
    "LINK_KNEE_PITCH_L",
    "LINK_ANKLE_PITCH_L",
    "LINK_ANKLE_ROLL_L",
    "LINK_HIP_ROLL_R",
    "LINK_HIP_YAW_R",
    "LINK_HIP_PITCH_R",
    "LINK_KNEE_PITCH_R",
    "LINK_ANKLE_PITCH_R",
    "LINK_ANKLE_ROLL_R",
    "LINK_TORSO_YAW",
    "LINK_SHOULDER_PITCH_L",
    "LINK_SHOULDER_ROLL_L",
    "LINK_SHOULDER_YAW_L",
    "LINK_ELBOW_PITCH_L",
    "LINK_ELBOW_YAW_L",
    "LINK_SHOULDER_PITCH_R",
    "LINK_SHOULDER_ROLL_R",
    "LINK_SHOULDER_YAW_R",
    "LINK_ELBOW_PITCH_R",
    "LINK_ELBOW_YAW_R",
    "LINK_HEAD_PITCH",
    "LINK_HEAD_YAW",
]

# Map FK links to body index; for TOE/HEEL/WRIST use parent
CHILD_TO_PARENT = {
    "LINK_ANKLE_ROLL_L_TOE": "LINK_ANKLE_ROLL_L",
    "LINK_ANKLE_ROLL_L_HEEL": "LINK_ANKLE_ROLL_L",
    "LINK_ANKLE_ROLL_R_TOE": "LINK_ANKLE_ROLL_R",
    "LINK_ANKLE_ROLL_R_HEEL": "LINK_ANKLE_ROLL_R",
    "LINK_WRIST_PITCH_L": "LINK_ELBOW_YAW_L",
    "LINK_WRIST_ROLL_L": "LINK_ELBOW_YAW_L",
    "LINK_WRIST_PITCH_R": "LINK_ELBOW_YAW_R",
    "LINK_WRIST_ROLL_R": "LINK_ELBOW_YAW_R",
}


def save_npz_non_zip64(output_path: Path, **arrays: np.ndarray) -> None:
    """Write an npz file compatible with cnpy by disabling ZIP64 extensions."""
    with zipfile.ZipFile(
        output_path, mode="w", compression=zipfile.ZIP_DEFLATED, allowZip64=False, compresslevel=6
    ) as zf:
        for name, array in arrays.items():
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            zf.writestr(f"{name}.npy", buffer.getvalue())


def main():
    parser = argparse.ArgumentParser(description="Convert npy to npz for whole_body_tracking")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input .npy file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output .npz file")
    parser.add_argument("--fps", type=float, default=50.0, help="Output fps (default 50)")
    parser.add_argument("--input_fps", type=float, default=30.0, help="Input npy fps (default 30)")
    parser.add_argument(
        "--engineai_lab",
        type=str,
        default=None,
        help="Path to engineaimuaythailab checkout. If omitted, auto-detect or use ENGINEAI_LAB_ROOT.",
    )
    args = parser.parse_args()

    engineai_root = _resolve_engineai_lab(args.engineai_lab)
    print(f"[npy_to_npz] Using engineaimuaythailab from {engineai_root}")
    from T800_motion_loader import T800MotionLoader

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        print(f"[ERROR] Input not found: {input_path}")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load npy and compute FK via T800MotionLoader
    input_dt = 1.0 / args.input_fps
    target_dt = 1.0 / args.fps
    loader = T800MotionLoader(
        str(input_path),
        csv_dt=input_dt,
        target_dt=target_dt,
        device="cpu",
        fix_base_height=False,
    )
    motion = loader.get_motion_data()

    T = motion.frame_count
    joint_pos = motion.joint_pos.numpy().astype(np.float32)
    joint_vel = motion.joint_vel.numpy().astype(np.float32)
    link_pos = motion.link_pos.numpy().astype(np.float32)  # [T, 25, 3]
    link_quat = motion.link_quat.numpy().astype(np.float32)  # [T, 25, 4]
    link_lin_vel = motion.link_lin_vel.numpy().astype(np.float32)
    link_ang_vel = motion.link_ang_vel.numpy().astype(np.float32)

    # Build body arrays in the repository-standard DFS T800 body order.
    num_bodies = len(T800_BODY_ORDER)
    print(f"[npy_to_npz] Using DFS body order ({num_bodies} bodies)")
    body_pos_w = np.zeros((T, num_bodies, 3), dtype=np.float32)
    body_quat_w = np.zeros((T, num_bodies, 4), dtype=np.float32)
    body_lin_vel_w = np.zeros((T, num_bodies, 3), dtype=np.float32)
    body_ang_vel_w = np.zeros((T, num_bodies, 3), dtype=np.float32)

    fk_name_to_idx = {n: i for i, n in enumerate(FK_LINK_NAMES)}

    for body_idx, body_name in enumerate(T800_BODY_ORDER):
        if body_name in fk_name_to_idx:
            fk_idx = fk_name_to_idx[body_name]
            body_pos_w[:, body_idx, :] = link_pos[:, fk_idx, :]
            body_quat_w[:, body_idx, :] = link_quat[:, fk_idx, :]
            body_lin_vel_w[:, body_idx, :] = link_lin_vel[:, fk_idx, :]
            body_ang_vel_w[:, body_idx, :] = link_ang_vel[:, fk_idx, :]
        elif body_name in CHILD_TO_PARENT:
            parent_name = CHILD_TO_PARENT[body_name]
            parent_idx = fk_name_to_idx.get(parent_name)
            if parent_idx is not None:
                body_pos_w[:, body_idx, :] = link_pos[:, parent_idx, :]
                body_quat_w[:, body_idx, :] = link_quat[:, parent_idx, :]
                body_lin_vel_w[:, body_idx, :] = link_lin_vel[:, parent_idx, :]
                body_ang_vel_w[:, body_idx, :] = link_ang_vel[:, parent_idx, :]
            else:
                body_pos_w[:, body_idx, :] = link_pos[:, 0, :]  # fallback to base
                body_quat_w[:, body_idx, :] = link_quat[:, 0, :]
                body_lin_vel_w[:, body_idx, :] = link_lin_vel[:, 0, :]
                body_ang_vel_w[:, body_idx, :] = link_ang_vel[:, 0, :]
        else:
            body_pos_w[:, body_idx, :] = link_pos[:, 0, :]
            body_quat_w[:, body_idx, :] = link_quat[:, 0, :]
            body_lin_vel_w[:, body_idx, :] = link_lin_vel[:, 0, :]
            body_ang_vel_w[:, body_idx, :] = link_ang_vel[:, 0, :]

    save_npz_non_zip64(
        output_path,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        fps=np.array([args.fps], dtype=np.float32),
    )
    print(f"[OK] Saved {output_path}")
    print(f"     Frames: {T}, FPS: {args.fps}, Bodies: {num_bodies}, Joints: {joint_pos.shape[1]}")
    print(f"     Base Z range: [{body_pos_w[:, 0, 2].min():.3f}, {body_pos_w[:, 0, 2].max():.3f}]")


if __name__ == "__main__":
    main()
