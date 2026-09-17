import argparse
import pickle
from pathlib import Path

import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True)
    parser.add_argument(
        "--xml",
        default=str(
            next(
                (
                    parent
                    for parent in Path(__file__).resolve().parents
                    if (parent / "projects").is_dir() and (parent / "frameworks").is_dir()
                ),
                Path(__file__).resolve().parents[3],
            )
            / "tools/retargeting/gmr_lens110/assets/lens110_21dof/mjcf/lens110_21dof.xml"
        ),
    )
    args = parser.parse_args()

    with open(args.pkl, "rb") as file:
        motion = pickle.load(file)

    print("keys", sorted(motion.keys()))
    print("fps", motion["fps"])
    print("root_pos", motion["root_pos"].shape)
    print("root_rot", motion["root_rot"].shape)
    print("dof_pos", motion["dof_pos"].shape)
    for idx, name in enumerate(motion["dof_names"]):
        values = motion["dof_pos"][:, idx]
        print(
            f"{idx:02d} {name:28s} "
            f"min={values.min(): .4f} max={values.max(): .4f} mean={values.mean(): .4f}"
        )

    model = mujoco.MjModel.from_xml_path(args.xml)
    print("\nxml joint limits")
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        qpos_addr = model.jnt_qposadr[joint_id]
        if qpos_addr < 7:
            continue
        print(
            f"{qpos_addr - 7:02d} {name:28s} "
            f"limited={bool(model.jnt_limited[joint_id])} "
            f"range={model.jnt_range[joint_id].tolist()}"
        )


if __name__ == "__main__":
    main()
