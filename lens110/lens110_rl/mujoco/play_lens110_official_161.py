"""Lens110 161 维策略的官方播放器适配版。

基于 Lens110_Dance_Sim2Real_v2_20260825/replay/mujoco_sim2sim_official.py 的环境处理
(官方模型 lens110_21dof.xml, 保留官方脚底盒/接触/摩擦/膝盖限位/质心),
但观测换成我们的 161 维, 动作换成 21 维参考中心残差:
    q_des = 参考当前帧关节角 + 0.25 * action (不 clip)
PD 默认从 deploy_config.yaml 读取 (训练 PD), 也可用官方强 PD。
"""

import argparse
import os
import time

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import yaml


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

# 官方强 PD (USD 顺序), 仅在未提供 deploy 时使用
OFFICIAL_KP = np.array([
    120, 120, 45, 120, 120, 35, 35, 120, 120, 35, 35, 120, 120, 35, 35, 55, 55, 35, 35, 55, 55,
], dtype=np.float64)
OFFICIAL_KD = np.array([
    4, 4, 1.5, 4, 4, 1.2, 1.2, 4, 4, 1.2, 1.2, 4, 4, 1.2, 1.2, 2, 2, 1.2, 1.2, 2, 2,
], dtype=np.float64)

POLICY_HZ = 100.0


def quat_to_rotmat_wxyz(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def root_rot_tan_norm(q_wxyz):
    r = quat_to_rotmat_wxyz(q_wxyz)
    return np.concatenate([r[:, 0], r[:, 2]])


def foot_contact_flags(model, data):
    flags = np.zeros(2, dtype=np.float32)
    force = np.zeros(6, dtype=np.float64)
    for i in range(data.ncon):
        c = data.contact[i]
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
        pair = n1 + "|" + n2
        if "left_ankle_roll_collision" in pair:
            mujoco.mj_contactForce(model, data, i, force)
            if force[0] > 1.0:
                flags[0] = 1.0
        if "right_ankle_roll_collision" in pair:
            mujoco.mj_contactForce(model, data, i, force)
            if force[0] > 1.0:
                flags[1] = 1.0
    return flags


def flatten_feet_init(model, data, joint_qpos):
    """出生时把左右脚底校平 (roll/pitch≈0), 确保双脚都贴地 -> 脚触地 obs=[1,1]。"""
    body_name_to_id = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(model.nbody)}
    pairs = [
        ("left_ankle_roll_link", "left_ankle_pitch_joint", "left_ankle_roll_joint"),
        ("right_ankle_roll_link", "right_ankle_pitch_joint", "right_ankle_roll_joint"),
    ]
    for bname, pj, rj in pairs:
        if bname not in body_name_to_id or pj not in joint_qpos or rj not in joint_qpos:
            continue
        bid = body_name_to_id[bname]
        pq, rq = joint_qpos[pj], joint_qpos[rj]
        for _ in range(10):
            q = data.xquat[bid]
            w, x, y, z = q
            roll = np.degrees(np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))
            pitch = np.degrees(np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
            if abs(roll) < 0.05 and abs(pitch) < 0.05:
                break
            droll = float(np.clip(roll * np.pi / 180.0, -0.3, 0.3))
            dpitch = float(np.clip(pitch * np.pi / 180.0, -0.3, 0.3))
            data.qpos[rq] -= droll
            data.qpos[pq] -= dpitch
            mujoco.mj_forward(model, data)


def build_obs_161(model, data, usd_qpos, usd_dof, ref_pos, ref_quat, frame, n_frames):
    q = data.qpos[usd_qpos].copy()
    qvel = data.qvel[usd_dof].copy()
    root_q = data.qpos[3:7].copy()
    parts = [
        root_rot_tan_norm(root_q),          # 6
        data.qvel[3:6].copy(),              # 3 角速度
        q,                                  # 21 关节角
        qvel,                               # 21 关节速度
    ]
    for i in range(4):
        parts.append(root_rot_tan_norm(ref_quat[min(frame + i, n_frames - 1)]))  # 24
    for i in range(4):
        parts.append(ref_pos[min(frame + i, n_frames - 1)])                       # 84
    parts.append(foot_contact_flags(model, data))                                # 2
    obs = np.concatenate(parts).astype(np.float32)
    assert obs.shape[0] == 161, obs.shape
    return obs.reshape(1, -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--motion", required=True)
    parser.add_argument("--mjcf", required=True)
    parser.add_argument("--deploy", default=None, help="deploy_config.yaml (训练 PD)")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--rigid_feet",
        type=float,
        default=None,
        help="脚底接触刚性化: 指定 solref 时间常数 (如 0.005/0.01), 越小越硬; "
             "同时提高求解迭代, 减少脚底弹跳",
    )
    parser.add_argument(
        "--stiff_limits",
        action="store_true",
        default=False,
        help="关节限位刚性化 (jnt_solref=0.001): 防止踝roll等关节过冲限位, "
             "消除脚侧翻产生的横向冲击力",
    )
    args = parser.parse_args()

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    motion = np.load(args.motion)
    fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    ref_pos = motion["joint_pos"].astype(np.float64)         # (N,21) USD 顺序
    ref_quat = motion["body_quat_w"][:, 0]                   # (N,4) wxyz
    ref_pos_w = motion["body_pos_w"][:, 0]                   # (N,3)
    ref_lin_vel_w = motion["body_lin_vel_w"][:, 0]
    ref_ang_vel_w = motion["body_ang_vel_w"][:, 0]
    n_frames = ref_pos.shape[0]
    print(f"[sim2sim] 动作 {n_frames} 帧 @{fps:.0f}Hz ({n_frames/fps:.1f}s)")

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    data = mujoco.MjData(model)
    DECIMATION = max(1, int(round(1.0 / (POLICY_HZ * model.opt.timestep))))
    print(f"[sim2sim] 模型 timestep={model.opt.timestep}, decimation={DECIMATION} "
          f"(物理 {1.0/model.opt.timestep:.0f}Hz / 策略 {POLICY_HZ:.0f}Hz)")

    # ---- 官方播放器的环境处理 (脚底盒保留模型原始, 碰撞/摩擦/膝盖限位/质心) ----
    foot_geom_names = [
        "left_ankle_roll_collision", "right_ankle_roll_collision",
        "left_ankle_pitch_collision", "right_ankle_pitch_collision",
    ]
    foot_geom_ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in foot_geom_names}
    foot_geom_ids.discard(-1)
    for gid in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if gname in {"floor", "ground", "plane"} or (gid in foot_geom_ids):
            model.geom_contype[gid] = 1
            model.geom_conaffinity[gid] = 15
        else:
            model.geom_contype[gid] = 1
            model.geom_conaffinity[gid] = 0
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id >= 0:
        model.geom_friction[floor_id] = [0.7, 0.005, 0.0001]
    if args.rigid_feet is not None:
        # 刚性脚底接触: 更小 timeconst = 更硬; 高阻尼抑制弹跳
        rigid_solref = np.array([float(args.rigid_feet), 1.0])
        rigid_solimp = np.array([0.95, 0.99, 0.001, 0.5, 2.0])
        for gid in list(foot_geom_ids) + ([floor_id] if floor_id >= 0 else []):
            model.geom_solref[gid] = rigid_solref
            model.geom_solimp[gid] = rigid_solimp
        model.opt.iterations = max(int(model.opt.iterations), 500)
        print(f"[sim2sim] 脚底接触刚性化 (solref={args.rigid_feet}, 迭代={model.opt.iterations})")
    if args.stiff_limits:
        for jid in range(model.njnt):
            if model.jnt_limited[jid]:
                model.jnt_solref[jid] = np.array([0.001, 1.0])
                model.jnt_solimp[jid] = np.array([0.95, 0.99, 0.001, 0.5, 2.0])
        print("[sim2sim] 关节限位刚性化 (jnt_solref=0.001)")
    import re as _re
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name and _re.fullmatch(".*_knee_joint", name):
            model.jnt_range[jid] = (-0.087, 2.443)

    # 执行器/关节索引 (按名字映射, 与顺序无关)
    act_name_to_idx = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
                       for i in range(model.nu)}
    joint_qpos = {}
    joint_dof = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name and name != "root":
            joint_qpos[name] = model.jnt_qposadr[i]
            joint_dof[name] = model.jnt_dofadr[i]
    usd_qpos = np.array([joint_qpos[n] for n in USD_JOINT_NAMES])
    usd_dof = np.array([joint_dof[n] for n in USD_JOINT_NAMES])
    ctrl_to_usd = np.array([USD_JOINT_NAMES.index(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)) for i in range(model.nu)], dtype=int)

    # PD: 默认训练 PD (deploy), 否则官方强 PD
    if args.deploy:
        with open(args.deploy, "r", encoding="utf-8") as f:
            deploy = yaml.safe_load(f)
        kp = np.asarray(deploy["joint_stiffness"], dtype=np.float64)
        kd = np.asarray(deploy["joint_damping"], dtype=np.float64)
        action_clip = deploy.get("action_clip", None)
        print("[sim2sim] 使用训练 PD (deploy_config)")
    else:
        kp, kd = OFFICIAL_KP, OFFICIAL_KD
        action_clip = None
        print("[sim2sim] 使用官方强 PD")
    for usd_i, name in enumerate(USD_JOINT_NAMES):
        aidx = act_name_to_idx[name]
        model.actuator_gaintype[aidx] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_biastype[aidx] = mujoco.mjtBias.mjBIAS_AFFINE
        model.actuator_gainprm[aidx, :] = 0.0
        model.actuator_gainprm[aidx, 0] = kp[usd_i]
        model.actuator_biasprm[aidx, :] = 0.0
        model.actuator_biasprm[aidx, 1] = -kp[usd_i]
        model.actuator_biasprm[aidx, 2] = -kd[usd_i]
        effort = 80.0 if usd_i in (0, 1, 3, 4, 11, 12) else 36.0
        model.actuator_forcelimited[aidx] = 1
        model.actuator_forcerange[aidx, 0] = -effort
        model.actuator_forcerange[aidx, 1] = effort

    # 初始状态: 参考第 0 帧 + 速度 + 脚底贴地 (与官方一致)
    data.qpos[:3] = ref_pos_w[0]
    data.qpos[3:7] = ref_quat[0]
    data.qpos[usd_qpos] = ref_pos[0]
    data.qvel[:] = 0.0
    data.qvel[:3] = ref_lin_vel_w[0]
    data.qvel[3:6] = ref_ang_vel_w[0]
    data.qvel[usd_dof] = motion["joint_vel"][0]
    mujoco.mj_forward(model, data)
    # 让两只脚的主支撑盒 (ankle_roll_collision) 都贴地: 按两只脚 roll 盒底部的最大值抬升,
    # 允许较低那只轻微压入 (几毫米), 物理接触会自动化解 -> 初始脚触地 obs=[1,1]。
    roll_geoms = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
        for n in ("left_ankle_roll_collision", "right_ankle_roll_collision")
    ]
    roll_bottoms = [float(data.geom_xpos[g, 2] - model.geom_size[g, 2]) for g in roll_geoms]
    data.qpos[2] += 0.002 - max(roll_bottoms)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    # 校平脚底, 保证双脚都贴地 -> 脚触地 obs=[1,1] (与 Isaac 训练一致)
    joint_qpos = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): int(model.jnt_qposadr[i])
                  for i in range(model.njnt) if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) not in (None, "root")}
    flatten_feet_init(model, data, joint_qpos)
    mujoco.mj_forward(model, data)
    flags = foot_contact_flags(model, data)
    print(f"[sim2sim] 初始脚触地 obs = {flags} (期望 [1,1])")
    print(f"[sim2sim] 初始 root_z={data.qpos[2]:.3f} (脚底贴地)")

    frame = 0
    play_state = {"speed": float(args.speed), "paused": False, "frame_reset": False}

    def on_key(keycode: int) -> None:
        if keycode == 32:
            play_state["paused"] = not play_state["paused"]
        elif keycode in (91, 45):
            play_state["speed"] = max(0.1, play_state["speed"] * 0.8)
        elif keycode in (93, 61):
            play_state["speed"] = min(10.0, play_state["speed"] * 1.25)
        elif keycode in (ord("r"), ord("R")):
            play_state["frame_reset"] = True

    print("[player] 快捷键: Space 暂停 | [ / ] 减速/加速 | R 重播 | Esc 退出")
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        start_time = time.time()
        while viewer.is_running() and frame < n_frames:
            if play_state.get("frame_reset", False):
                play_state["frame_reset"] = False
                frame = 0
                start_time = time.time()
                data.qpos[:3] = ref_pos_w[0]
                data.qpos[3:7] = ref_quat[0]
                data.qpos[usd_qpos] = ref_pos[0]
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                roll_bottoms = [
                    float(data.geom_xpos[g, 2] - model.geom_size[g, 2])
                    for g in (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
                              for n in ("left_ankle_roll_collision", "right_ankle_roll_collision"))
                ]
                data.qpos[2] += 0.002 - max(roll_bottoms)
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                flatten_feet_init(model, data, joint_qpos)
                mujoco.mj_forward(model, data)
                continue
            if play_state["paused"]:
                time.sleep(0.02)
                continue

            # 策略推理 (100Hz)
            obs = build_obs_161(model, data, usd_qpos, usd_dof, ref_pos, ref_quat, frame, n_frames)
            out = sess.run(None, {"obs": obs})[0][0].astype(np.float32)
            if action_clip is not None:
                out = np.clip(out, -float(action_clip), float(action_clip))
            # 21 维参考中心残差: q_des = 参考 + 0.25 * action (不 clip)
            q_des_usd = ref_pos[frame] + 0.25 * out
            # 目标限位钳制: Isaac 驱动器会把目标硬钳在关节范围内, MuJoCo 关节限位是软约束,
            # 不钳制会导致踝 roll 等关节冲出限位 (脚侧翻 50°)。这里手动对齐。
            for j, name in enumerate(USD_JOINT_NAMES):
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid >= 0 and model.jnt_limited[jid]:
                    lo, hi = model.jnt_range[jid]
                    q_des_usd[j] = min(max(q_des_usd[j], lo), hi)
            target_q = q_des_usd[ctrl_to_usd]

            for _ in range(DECIMATION):
                data.ctrl[:] = target_q
                mujoco.mj_step(model, data)
            frame += 1
            viewer.sync()
            if frame % 100 == 0:
                print(f"[sim2sim] 帧 {frame}/{n_frames} ({frame/fps:.1f}s) "
                      f"根高度 {data.qpos[2]:.3f} 关节速度峰值 {np.abs(data.qvel[usd_dof]).max():.2f}")
            if play_state["speed"] > 0:
                target_time = start_time + frame / (POLICY_HZ * play_state["speed"])
                sleep = target_time - time.time()
                if sleep > 0:
                    time.sleep(sleep)
    print(f"[sim2sim] 完成 {frame} 帧")


if __name__ == "__main__":
    main()
