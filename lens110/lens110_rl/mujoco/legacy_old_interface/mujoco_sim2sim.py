"""MuJoCo sim2sim: 用训练好的 ONNX 策略做闭环推理验证。

与 Isaac 训练环境对齐:
  - 策略 100Hz / 物理 500Hz (decimation 5)
  - 观察 114 维: [command(42) | anchor_ori_6d(6) | base_ang_vel(3) |
    joint_pos_rel(21) | joint_vel_rel(21) | last_action(21)]
  - 动作: q_des = 参考关节角 + scale * action
  - PD: kp/kd 与训练一致 (腿120/4 踝55/2 腰45/1.5 臂35/1.2)

用法 (gmr 环境, 需 onnxruntime):
    python mujoco_sim2sim.py --onnx policy.onnx \
        --motion motion/lens110_amp_100hz.npz \
        --mjcf assets/mjcf/lens110_21dof.xml
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort

# 与训练环境一致的逐关节 action_scale (USD 顺序)
ACTION_SCALE = np.array([
    0.2, 0.1, 0.1, 0.2, 0.12, 0.08,
    0.2, 0.1, 0.1, 0.2, 0.12, 0.08,
    0.1, 0.15, 0.15, 0.1, 0.15, 0.15, 0.15, 0.1, 0.15,
], dtype=np.float32)

# USD 顺序关节名 (与训练 npz 一致)
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

# 训练 kp/kd (USD 顺序, 与资产配置一致)
KP = np.array([
    120, 120, 45, 120, 120, 35, 35, 55, 55, 35, 35, 120, 120, 35, 35, 55, 55, 35, 35, 55, 55,
], dtype=np.float64)
KD = np.array([
    4, 4, 1.5, 4, 4, 1.2, 1.2, 2, 2, 1.2, 1.2, 4, 4, 1.2, 1.2, 2, 2, 1.2, 1.2, 2, 2,
], dtype=np.float64)

POLICY_HZ = 100.0
PHYSICS_HZ = 500.0
DECIMATION = 5


def quat_to_6d(q_wxyz: np.ndarray) -> np.ndarray:
    """wxyz 四元数 -> 6D 旋转表示 (前两列)。"""
    w, x, y, z = q_wxyz
    # 旋转矩阵
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    return R[:, :2].flatten()


def quat_mul_wxyz(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_inv_wxyz(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rotate_wxyz(q, v):
    """用 wxyz 四元数旋转向量 v (世界->body 用 inv)。"""
    w, x, y, z = q
    t2 = 2 * np.cross(np.array([x, y, z]), v)
    return v + w * t2 + np.cross(np.array([x, y, z]), t2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--motion", required=True)
    parser.add_argument("--mjcf", required=True)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    motion = np.load(args.motion)
    fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    ref_pos = motion["joint_pos"]  # (N,21) USD 顺序
    ref_vel = motion["joint_vel"]
    ref_quat = motion["body_quat_w"][:, 0]  # (N,4) wxyz, pelvis
    n_frames = ref_pos.shape[0]
    print(f"[sim2sim] 动作 {n_frames} 帧 @{fps:.0f}Hz ({n_frames/fps:.1f}s)")

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    data = mujoco.MjData(model)
    # 运行时把 position actuator 的 kp/kv 改成训练值
    # MuJoCo position actuator: gainprm = [kp, kv, 0], biasprm = [-kp, -kv, 0]
    act_name_to_idx = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
                       for i in range(model.nu)}
    # ctrl 索引 -> USD 关节索引 (actuator 按关节名排列, 顺序可能与 USD 不同)
    ctrl_to_usd = np.array([USD_JOINT_NAMES.index(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))
        for i in range(model.nu)], dtype=int)
    for usd_i, name in enumerate(USD_JOINT_NAMES):
        aidx = act_name_to_idx[name]
        kp, kd = KP[usd_i], KD[usd_i]
        # MuJoCo position actuator: gainprm=[kp, kv, ...], biasprm=[0, -kp, -kv, ...]
        model.actuator_gainprm[aidx, 0] = kp
        model.actuator_gainprm[aidx, 1] = kd
        model.actuator_biasprm[aidx, 1] = -kp
        model.actuator_biasprm[aidx, 2] = -kd
    # 关节点名 -> qpos 索引
    joint_qpos = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name and name != "root":
            joint_qpos[name] = model.jnt_qposadr[i]
    joint_dof = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name and name != "root":
            joint_dof[name] = model.jnt_dofadr[i]
    usd_qpos = np.array([joint_qpos[n] for n in USD_JOINT_NAMES])
    usd_dof = np.array([joint_dof[n] for n in USD_JOINT_NAMES])

    # 初始: 参考第 0 帧 (关节角 + pelvis 位置/朝向)
    data.qpos[:3] = motion["body_pos_w"][0, 0]  # pelvis 位置
    data.qpos[3:7] = motion["body_quat_w"][0, 0]  # wxyz
    data.qpos[usd_qpos] = ref_pos[0]
    mujoco.mj_forward(model, data)

    # 运行时不默认 (joint_pos_rel 的基准): 与训练资产 default_joint_pos 一致
    # (v29 合同 runtime_default_q, USD 顺序)
    default_q = np.array([
        -0.10060572624206543,  # left_hip_pitch
        -0.09989992529153824,  # right_hip_pitch
        -0.00285041774623096,  # torso_yaw
        0.0026404764503240585,  # left_hip_roll
        0.0026751249097287655,  # right_hip_roll
        0.0017803874798119068,  # left_shoulder_pitch
        -0.0005099154077470303,  # right_shoulder_pitch
        0.0019215415231883526,  # left_hip_yaw
        -0.001625739736482501,  # right_hip_yaw
        0.15245753526687622,  # left_shoulder_roll
        -0.1522899866104126,  # right_shoulder_roll
        0.19745133817195892,  # left_knee
        0.19945533573627472,  # right_knee
        0.002760404720902443,  # left_shoulder_yaw
        -0.0017441902309656143,  # right_shoulder_yaw
        -0.10183628648519516,  # left_ankle_pitch
        -0.09765433520078659,  # right_ankle_pitch
        -0.25036799907684326,  # left_elbow
        -0.2508581280708313,  # right_elbow
        0.00027221417985856533,  # left_ankle_roll
        0.0019795182161033154,  # right_ankle_roll
    ])

    last_action = np.zeros(21, dtype=np.float32)
    frame = 0
    policy_step = 0
    prev_q = data.qpos[usd_qpos].copy()
    prev_time = time.time()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and frame < n_frames:
            # --- 策略推理 (100Hz) ---
            if policy_step % DECIMATION == 0:
                q = data.qpos[usd_qpos].copy()
                # 关节速度 (数值差分)
                qvel = data.qvel[usd_dof].copy()
                # 基座角速度 (body 系): 用 root 角速度近似
                base_ang_vel = data.qvel[3:6].copy()
                # 参考 anchor 朝向 6d: 相对机器人当前朝向
                robot_q_wxyz = data.qpos[3:7].copy()
                ref_q_wxyz = ref_quat[frame]
                rel_q = quat_mul_wxyz(quat_inv_wxyz(robot_q_wxyz), ref_q_wxyz)
                rel_ori_6d = quat_to_6d(rel_q)
                # base_ang_vel: 世界系 -> body 系
                base_ang_vel = quat_rotate_wxyz(quat_inv_wxyz(robot_q_wxyz), data.qvel[3:6])
                obs = np.concatenate([
                    ref_pos[frame], ref_vel[frame],           # 42
                    rel_ori_6d,                                # 6
                    base_ang_vel,                              # 3
                    q - default_q,                             # 21
                    qvel,                                      # 21
                    last_action,                               # 21
                ]).astype(np.float32).reshape(1, -1)
                assert obs.shape[1] == 114, obs.shape
                out = sess.run(None, {"obs": obs})[0][0]
                last_action = out.astype(np.float32)
                q_des = ref_pos[frame] + ACTION_SCALE * out
                target_q = q_des

            # --- 物理步 (500Hz, 5 次 = 1 策略步) ---
            for _ in range(DECIMATION):
                # position actuator: ctrl = 目标关节角, MuJoCo 内部做 PD
                data.ctrl[:] = target_q[ctrl_to_usd]
                mujoco.mj_step(model, data)

            frame += 1
            policy_step += 1
            if frame % 100 == 0:
                print(f"[sim2sim] 帧 {frame}/{n_frames} ({frame/fps:.1f}s) "
                      f"根高度 {data.qpos[2]:.3f} 关节速度峰值 {np.abs(data.qvel[usd_qpos-7]).max():.2f}")

            # 实时速率
            if args.speed > 0:
                dt_target = 1.0 / (POLICY_HZ * args.speed)
                sleep = dt_target - (time.time() - prev_time)
                if sleep > 0:
                    time.sleep(sleep)
                prev_time = time.time()

    print(f"[sim2sim] 完成 {frame} 帧")


if __name__ == "__main__":
    main()
