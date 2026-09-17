"""MuJoCo 完整动作播放器: 参考 npz (root pos/quat + 21 关节) 运动学回放。

用于检查过渡段/起始动作是否自然、脚底是否贴地:
    python mujoco_play_npz_full.py --npz <xxx.npz> [--start 0] [--duration 5] [--fps 100]
"""

import argparse
import os
import time

import mujoco
import mujoco.viewer
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MJCF = os.path.abspath(
    os.path.join(HERE, "..", "..", "..", "assets", "mjcf", "lens110_21dof_sim_flatfoot.xml")
)

USD_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "torso_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--mjcf", default=DEFAULT_MJCF)
    parser.add_argument("--fps", type=float, default=100.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--duration", type=float, default=40.0, help="播放秒数 (默认 40s)")
    args = parser.parse_args()

    motion = np.load(args.npz)
    # 优先使用 npz 自带的 joint_names (若存在), 否则用内置 USD 顺序
    if "joint_names" in motion.files:
        joint_names = [str(n) for n in motion["joint_names"]]
        print(f"[play-npz] 使用 npz 内 joint_names ({len(joint_names)} 个)")
    else:
        joint_names = USD_JOINT_NAMES
    jp = motion["joint_pos"].astype(np.float64)
    bp = motion["body_pos_w"].astype(np.float64)
    bq = motion["body_quat_w"].astype(np.float64)
    n = jp.shape[0]
    print(f"[play-npz] {n} 帧 @ {args.fps:.0f}Hz, 播放起点帧 {args.start}")

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    data = mujoco.MjData(model)
    qpos_map = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): model.jnt_qposadr[i]
        for i in range(model.njnt)
    }
    dt = 1.0 / args.fps
    end = min(n, args.start + int(args.duration * args.fps))
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("[play-npz] Esc 或关窗口退出")
        last = time.time()
        for i in range(args.start, end):
            if not viewer.is_running():
                break
            data.qpos[:3] = bp[i, 0]
            data.qpos[3:7] = bq[i, 0]
            for j, name in enumerate(joint_names):
                data.qpos[qpos_map[name]] = jp[i, j]
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            if i % 20 == 0:
                # 脚底最低点 (两个踝 box 角点)
                low = 1e9
                for gname in ("left_ankle_roll_collision", "right_ankle_roll_collision"):
                    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
                    p = data.geom_xpos[gid]
                    mat = data.geom_xmat[gid].reshape(3, 3)
                    sz = model.geom_size[gid]
                    for sx in (-1, 1):
                        for sy in (-1, 1):
                            for szz in (-1, 1):
                                low = min(low, float((p + mat @ (sz * np.array([sx, sy, szz])))[2]))
                print(f"  [帧 {i:4d} ({i / args.fps:5.2f}s)] root_z={data.qpos[2]:.4f} "
                      f"脚底最低 z={low:.4f}")
            viewer.sync()
            sleep = dt - (time.time() - last)
            if sleep > 0:
                time.sleep(sleep)
            last = time.time()
    print("[play-npz] 结束")


if __name__ == "__main__":
    main()
