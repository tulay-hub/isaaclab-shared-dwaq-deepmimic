import argparse
import pathlib
import pickle

import numpy as np
from scipy.spatial.transform import Rotation as R


NAMES = [
    "49_18_stageii_balance_one_leg",
    "49_19_stageii_balance_one_leg",
    "49_20_stageii_balance_one_leg",
]
LEG_JOINTS = [
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]


def q(arr):
    return np.round(arr, 3)


def summarize_source(name, src_root):
    data = np.load(src_root / f"{name}.npz", allow_pickle=True)
    root_euler = R.from_rotvec(data["root_orient"]).as_euler("XYZ", degrees=True)
    trans = data["trans"]
    print(f"SRC {name}")
    print(f"  frames={len(trans)} fps={float(data['mocap_frame_rate']):.3f}")
    print(f"  root_euler_deg min={q(root_euler.min(axis=0))} max={q(root_euler.max(axis=0))} mean={q(root_euler.mean(axis=0))}")
    print(f"  trans min={q(trans.min(axis=0))} max={q(trans.max(axis=0))} mean={q(trans.mean(axis=0))}")


def summarize_pkl(name, pkl_root):
    with (pkl_root / f"{name}.pkl").open("rb") as file:
        data = pickle.load(file)
    # Stored lens110 retarget pkl root_rot is xyzw. SciPy also expects xyzw.
    root_euler = R.from_quat(data["root_rot"]).as_euler("XYZ", degrees=True)
    dof = data["dof_pos"]
    dof_names = data["dof_names"]
    index = {joint: idx for idx, joint in enumerate(dof_names)}
    print(f"PKL {name}")
    print(f"  frames={len(dof)} fps={float(data['fps']):.3f}")
    print(f"  root_pos min={q(data['root_pos'].min(axis=0))} max={q(data['root_pos'].max(axis=0))} mean={q(data['root_pos'].mean(axis=0))}")
    print(f"  root_euler_deg min={q(root_euler.min(axis=0))} max={q(root_euler.max(axis=0))} mean={q(root_euler.mean(axis=0))}")
    for joint in LEG_JOINTS:
        if joint not in index:
            continue
        values = dof[:, index[joint]]
        print(f"  {joint:24s} min={values.min(): .3f} max={values.max(): .3f} mean={values.mean(): .3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=pathlib.Path, required=True, help="Local directory containing the 49 source .npz files.")
    parser.add_argument("--pkl_root", type=pathlib.Path, required=True, help="Local directory containing the 49 retargeted .pkl files.")
    args = parser.parse_args()
    for name in NAMES:
        summarize_source(name, args.src_root)
        summarize_pkl(name, args.pkl_root)
        print()


if __name__ == "__main__":
    main()
