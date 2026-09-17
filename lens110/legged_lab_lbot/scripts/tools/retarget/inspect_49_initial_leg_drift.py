import argparse
import pathlib
import pickle

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


WORKSPACE_ROOT = next(
    (
        parent
        for parent in pathlib.Path(__file__).resolve().parents
        if (parent / "projects").is_dir() and (parent / "frameworks").is_dir()
    ),
    pathlib.Path(__file__).resolve().parents[3],
)
DEFAULT_XML = WORKSPACE_ROOT / "tools/retargeting/gmr_lens110/assets/lens110_21dof/mjcf/lens110_21dof.xml"


def load_motion(path):
    with path.open("rb") as file:
        return pickle.load(file)


def qpos_from_motion(motion, frame):
    return np.concatenate(
        [
            motion["root_pos"][frame],
            motion["root_rot"][frame, [3, 0, 1, 2]],
            motion["dof_pos"][frame],
        ]
    )


def body_local_pos(data, body_name):
    pelvis_pos = data.body("pelvis").xpos.copy()
    pelvis_mat = data.body("pelvis").xmat.reshape(3, 3).copy()
    world_delta = data.body(body_name).xpos.copy() - pelvis_pos
    return pelvis_mat.T @ world_delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=pathlib.Path, default=DEFAULT_XML)
    parser.add_argument("--backup", type=pathlib.Path, required=True, help="Local backup 49 motion .pkl")
    parser.add_argument("--current", type=pathlib.Path, required=True, help="Local current 49 motion .pkl")
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    for label, path in (("backup", args.backup), ("current", args.current)):
        motion = load_motion(path)
        index = {name: idx for idx, name in enumerate(motion["dof_names"])}
        print(f"\n{label}: {path}")
        print("frame root_rpy_deg Lfoot_local_xyz Rfoot_local_xyz stance_y Lhip_roll Lankle_roll Rhip_roll Rankle_roll")
        for frame in range(0, min(30, len(motion["dof_pos"])), 2):
            data.qpos[:] = qpos_from_motion(motion, frame)
            mujoco.mj_forward(model, data)
            left_foot = body_local_pos(data, "left_ankle_roll_link")
            right_foot = body_local_pos(data, "right_ankle_roll_link")
            root_rpy = R.from_quat(motion["root_rot"][frame]).as_euler("XYZ", degrees=True)
            dof = motion["dof_pos"][frame]
            print(
                frame,
                np.round(root_rpy, 2),
                np.round(left_foot, 4),
                np.round(right_foot, 4),
                f"{(left_foot[1] - right_foot[1]): .4f}",
                f"{dof[index['left_hip_roll_joint']]: .4f}",
                f"{dof[index['left_ankle_roll_joint']]: .4f}",
                f"{dof[index['right_hip_roll_joint']]: .4f}",
                f"{dof[index['right_ankle_roll_joint']]: .4f}",
            )


if __name__ == "__main__":
    main()
