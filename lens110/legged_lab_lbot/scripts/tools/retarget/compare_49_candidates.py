import argparse
import pickle
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


WORKSPACE_ROOT = next(
    (
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "projects").is_dir() and (parent / "frameworks").is_dir()
    ),
    Path(__file__).resolve().parents[3],
)
DEFAULT_XML = WORKSPACE_ROOT / "tools/retargeting/gmr_lens110/assets/lens110_21dof/mjcf/lens110_21dof.xml"


def qpos(motion, frame):
    return np.concatenate([motion["root_pos"][frame], motion["root_rot"][frame, [3, 0, 1, 2]], motion["dof_pos"][frame]])


def local(data, name):
    pelvis_pos = data.body("pelvis").xpos.copy()
    pelvis_mat = data.body("pelvis").xmat.reshape(3, 3).copy()
    return pelvis_mat.T @ (data.body(name).xpos.copy() - pelvis_pos)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--raw-ground", type=Path, required=True)
    parser.add_argument("--formal", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    paths = {"raw": args.raw, "raw_ground": args.raw_ground, "formal": args.formal, "backup": args.backup}
    for label, path in paths.items():
        with open(path, "rb") as file:
            motion = pickle.load(file)
        index = {name: idx for idx, name in enumerate(motion["dof_names"])}
        print(f"\n{label}: {path}")
        for frame in [0, 5, 10, 19, 30, 60, 100, 150, 220]:
            if frame >= len(motion["root_pos"]):
                continue
            data.qpos[:] = qpos(motion, frame)
            mujoco.mj_forward(model, data)
            root_rpy = R.from_quat(motion["root_rot"][frame]).as_euler("XYZ", degrees=True)
            lfoot = local(data, "left_ankle_roll_link")
            rfoot = local(data, "right_ankle_roll_link")
            lhand = local(data, "left_hand")
            rhand = local(data, "right_hand")
            dof = motion["dof_pos"][frame]
            print(
                f"f={frame:03d}",
                "rpy=", np.round(root_rpy, 1),
                "lf=", np.round(lfoot, 2),
                "rf=", np.round(rfoot, 2),
                "lh=", np.round(lhand, 2),
                "rh=", np.round(rhand, 2),
                "l/r_sh_pitch=",
                round(float(dof[index["left_shoulder_pitch_joint"]]), 2),
                round(float(dof[index["right_shoulder_pitch_joint"]]), 2),
            )


if __name__ == "__main__":
    main()
