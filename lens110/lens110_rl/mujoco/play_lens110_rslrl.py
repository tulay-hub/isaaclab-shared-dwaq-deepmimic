"""Lens110 新框架 (rsl_rl) MuJoCo sim2sim 播放器。

接口: 观测 161 维 (H 版), 动作 21 维参考中心残差:
    q_des = 参考当前帧关节角 + action_scale * action
环境设置与 Lens110_Dance_Sim2Real_v3_20260826 保持一致:
物理 1000Hz / 策略 100Hz, 强 PD (腿120/4 踝55/2 腰45/1.5 臂35/1.2),
摩擦 0.7, 保留 sim_flatfoot 自带 dof 阻尼/armature, 只留脚/地碰撞,
出生时脚底校平 (flatten_feet_init)。

用法:
    python mujoco/play_lens110_rslrl.py \
        --onnx <run>/exported/policy.onnx \
        --motion <motion.npz>
"""

import argparse
import os
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import yaml

# 与 Lens110_ThreeAction_Walk_Marktime11999_Minimal 成功包对齐的脚底碰撞盒
TRAINING_FOOT_BOXES = {
    "left_ankle_roll_collision": ((0.019, 0.009, -0.030), (0.0975, 0.0475, 0.010)),
    "right_ankle_roll_collision": ((0.019, -0.009, -0.030), (0.0975, 0.0475, 0.010)),
    "left_ankle_pitch_collision": ((0.019, 0.009, -0.030), (0.0975, 0.0475, 0.010)),
    "right_ankle_pitch_collision": ((0.019, -0.009, -0.030), (0.0975, 0.0475, 0.010)),
}
FOOT_CONTACT_GEOMS = list(TRAINING_FOOT_BOXES.keys())

# ---- Lens110_Dance_Sim2Real_v3_20260826 环境设置 (USD 顺序) ----
V3_KP = np.array([
    120, 120, 45, 120, 120, 35, 35, 120, 120, 35, 35, 120, 120, 35, 35, 55, 55, 35, 35, 55, 55,
], dtype=np.float64)
V3_KD = np.array([
    4, 4, 1.5, 4, 4, 1.2, 1.2, 4, 4, 1.2, 1.2, 4, 4, 1.2, 1.2, 2, 2, 1.2, 1.2, 2, 2,
], dtype=np.float64)
V3_EFFORT = np.array([
    80, 80, 36, 80, 80, 36, 36, 36, 36, 36, 36, 80, 80, 36, 36, 36, 36, 36, 36, 36, 36,
], dtype=np.float64)


def flatten_feet_init(model: mujoco.MjModel, data: mujoco.MjData, joint_qpos: dict[str, int]) -> None:
    """出生时把双脚底校平 (roll/pitch≈0), 与 v3 播放器一致。"""
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


def _parse_urdf_vec(s: str) -> np.ndarray:
    return np.array([float(x) for x in s.split()]) if s else np.zeros(3)


def apply_urdf_com_and_limits(model: mujoco.MjModel, urdf_path: str) -> None:
    """从训练 URDF 读取质心偏移和关节限位, 写入 MuJoCo 模型。

    该 XML 丢失了 body 质心偏移 (body_ipos 全 0), 且踝关节限位比 URDF 松,
    导致 MuJoCo 动力学与 Isaac 训练模型不一致。
    """
    root = ET.parse(urdf_path).getroot()
    link_com = {}
    for link in root.findall("link"):
        inert = link.find("inertial")
        if inert is None:
            continue
        origin = inert.find("origin")
        link_com[link.get("name")] = (
            _parse_urdf_vec(origin.get("xyz")) if origin is not None else np.zeros(3)
        )
    joint_limits = {}
    for joint in root.findall("joint"):
        lim = joint.find("limit")
        if lim is not None:
            joint_limits[joint.get("name")] = (
                float(lim.get("lower")),
                float(lim.get("upper")),
            )

    for bid in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if name in link_com:
            model.body_ipos[bid] = link_com[name]
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name in joint_limits and model.jnt_limited[jid]:
            model.jnt_range[jid] = joint_limits[name]
    print("[player] URDF 质心偏移/关节限位已写入 MuJoCo")


def quat_to_rotmat_wxyz(q: np.ndarray) -> np.ndarray:
    """wxyz 四元数 -> 3x3 旋转矩阵。"""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def root_rot_tan_norm(q_wxyz: np.ndarray) -> np.ndarray:
    """6D 姿态表示: 旋转矩阵第 0、2 列。"""
    r = quat_to_rotmat_wxyz(q_wxyz)
    return np.concatenate([r[:, 0], r[:, 2]])


def foot_contact_flags(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """左右脚踝触地标志 (2,): 对应碰撞 geom 法向力 > 1N。"""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, help="exported/policy.onnx")
    parser.add_argument("--motion", required=True, help="训练动作 npz (100Hz)")
    parser.add_argument("--mjcf", default=None, help="默认 assets/mjcf/lens110_21dof_sim_flatfoot.xml")
    parser.add_argument("--speed", type=float, default=1.0, help="播放速率倍率 (0=最快)")
    parser.add_argument("--physics_hz", type=float, default=500.0, help="MuJoCo 物理频率 (默认 500)")
    parser.add_argument(
        "--no_elbow_policy",
        action="store_true",
        default=False,
        help="肘关节(小臂)不让策略控制, 直接开环跟参考",
    )
    parser.add_argument(
        "--use_deploy_pd",
        action="store_true",
        default=True,
        help="用 deploy_config 里的训练 PD (腿40/5 踝55/5 臂20/5) (默认)",
    )
    parser.add_argument(
        "--use_v3_pd",
        action="store_false",
        dest="use_deploy_pd",
        help="改用 v3 强 PD (腿120/4 踝55/2 腰45/1.5 臂35/1.2)",
    )
    parser.add_argument(
        "--action_clip_override",
        type=float,
        default=None,
        help="覆盖动作截断范围 (如 1.0), 用于诊断动作过大导致的乱飞",
    )
    parser.add_argument(
        "--flatten_feet",
        action="store_true",
        default=True,
        help="出生时额外校平脚底 (v3 默认开启; 加 --no_flatten_feet 可关闭)",
    )
    parser.add_argument(
        "--no_flatten_feet",
        action="store_false",
        dest="flatten_feet",
        help="关闭出生时脚底校平, 保持与训练参考第0帧完全一致",
    )
    parser.add_argument(
        "--success_pkg_contact",
        action="store_true",
        default=True,
        help="按 Lens110_ThreeAction_Walk_Marktime11999_Minimal 成功包设置接触/动力学: "
             "PGS+implicitfast 200Hz, 摩擦1.0, dof damping=0/frictionloss=0.1/armature=0.01, "
             "只留脚底+地面碰撞 (1/1, 其余0/0)",
    )
    parser.add_argument(
        "--no_success_pkg_contact",
        action="store_false",
        dest="success_pkg_contact",
        help="不用成功包接触设置, 回到 v3 环境 (Newton 1000Hz 摩擦0.7 保留doof阻尼)",
    )
    parser.add_argument(
        "--dof_damping",
        type=float,
        default=None,
        help="额外设置 MuJoCo 关节阻尼 dof_damping (默认按模式: 成功包=0, v3=XML 0.01)",
    )
    args = parser.parse_args()

    POLICY_HZ = 100.0
    PHYSICS_HZ = float(args.physics_hz)
    DECIMATION = max(1, round(PHYSICS_HZ / POLICY_HZ))

    onnx_dir = os.path.dirname(os.path.abspath(args.onnx))
    with open(os.path.join(onnx_dir, "deploy_config.yaml"), "r", encoding="utf-8") as f:
        deploy = yaml.safe_load(f)
    joint_names = list(deploy["joint_names"])
    default_q = np.asarray(deploy["default_joint_pos"], dtype=np.float64)
    action_scale = float(deploy["action_scale"])
    action_mode = str(deploy.get("action_mode", "default_offset"))
    action_clip = args.action_clip_override if args.action_clip_override is not None else deploy.get("action_clip", None)
    print(
        f"[player] 关节数 {len(joint_names)}, action_scale={action_scale}, "
        f"action_mode={action_mode}, action_clip={action_clip}"
    )

    if args.mjcf is None:
        args.mjcf = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "..", "legged_lab_lbot",
                "source/legged_lab/legged_lab/data/Robots/model_humanoid_lens110/mjcf/lens110_21dof_motor.xml",
            )
        )

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    motion = np.load(args.motion)
    fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    ref_pos = motion["joint_pos"]  # (N,21)
    ref_quat = motion["body_quat_w"][:, 0]  # (N,4) wxyz, pelvis
    ref_body_pos = motion["body_pos_w"][:, 0]  # (N,3) pelvis
    n_frames = ref_pos.shape[0]
    print(f"[player] 动作 {n_frames} 帧 @{fps:.0f}Hz ({n_frames/fps:.1f}s)")

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    model.opt.timestep = 1.0 / PHYSICS_HZ
    if args.success_pkg_contact:
        # 成功包: PGS + implicitfast
        model.opt.solver = mujoco.mjtSolver.mjSOL_PGS
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    print(f"[player] 物理 {PHYSICS_HZ:.0f}Hz / 策略 {POLICY_HZ:.0f}Hz (decimation={DECIMATION})")
    urdf_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "legged_lab_lbot",
            "source/legged_lab/legged_lab/data/Robots/model_humanoid_lens110/lens110_21dof.urdf",
        )
    )
    apply_urdf_com_and_limits(model, urdf_path)
    data = mujoco.MjData(model)

    # 与成功包对齐: 重设脚底碰撞盒, 只保留脚底和地面碰撞
    for name, (pos, size) in TRAINING_FOOT_BOXES.items():
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            model.geom_pos[gid] = np.asarray(pos, dtype=np.float64)
            model.geom_size[gid] = np.asarray(size, dtype=np.float64)
            model.geom_type[gid] = mujoco.mjtGeom.mjGEOM_BOX
    # v3 碰撞设置: 脚底+地面 conaffinity=15, 其余 mesh 不参与碰撞 (conaffinity=0)
    foot_geom_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in TRAINING_FOOT_BOXES
    }
    foot_geom_ids.discard(-1)
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        active = geom_name in {"floor", "ground", "plane"} or geom_id in foot_geom_ids
        if args.success_pkg_contact:
            # 成功包: 只有脚底+地面 1/1, 其余 0/0 (完全无自碰撞/无身体碰地)
            model.geom_contype[geom_id] = int(active)
            model.geom_conaffinity[geom_id] = int(active)
        elif active:
            # v3: 脚底+地面 1/15
            model.geom_contype[geom_id] = 1
            model.geom_conaffinity[geom_id] = 15
        else:
            model.geom_contype[geom_id] = 1
            model.geom_conaffinity[geom_id] = 0
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id >= 0:
        model.geom_friction[floor_id] = (
            [1.0, 0.08, 0.004] if args.success_pkg_contact else [0.7, 0.005, 0.0001]
        )

    if args.success_pkg_contact:
        # 成功包: dof damping=0 (阻尼交给 PD kd), frictionloss=0.1, armature=0.01
        for jid in range(model.njnt):
            if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            did = model.jnt_dofadr[jid]
            model.dof_damping[did] = 0.0
            model.dof_frictionloss[did] = 0.1
            model.dof_armature[did] = 0.01
    if args.dof_damping is not None:
        for jid in range(model.njnt):
            if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            model.dof_damping[model.jnt_dofadr[jid]] = args.dof_damping
        print(f"[player] 关节阻尼 dof_damping = {args.dof_damping}")
    # 否则 (v3): 保留 sim_flatfoot 自带 dof damping/frictionloss/armature=0.01

    # 执行器/关节索引映射 (deploy 顺序 -> MuJoCo)
    act_name_to_idx = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(model.nu)
    }
    ctrl_to_joint = np.array([joint_names.index(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)) for i in range(model.nu)], dtype=int)
    for j_idx, name in enumerate(joint_names):
        aidx = act_name_to_idx[name]
        if args.use_deploy_pd:
            # 训练 PD (deploy_config, 与当前 Isaac 训练一致)
            kp = deploy["joint_stiffness"][j_idx]
            kd = deploy["joint_damping"][j_idx]
        else:
            # v3 强 PD (与 Lens110_Dance_Sim2Real_v3_20260826 一致)
            kp = V3_KP[j_idx]
            kd = V3_KD[j_idx]
        # XML 里是 motor 执行器, 这里转成位置 PD 伺服 (与训练 ImplicitActuator 对齐)
        model.actuator_gaintype[aidx] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_biastype[aidx] = mujoco.mjtBias.mjBIAS_AFFINE
        model.actuator_gainprm[aidx, :] = 0.0
        model.actuator_gainprm[aidx, 0] = kp
        model.actuator_biasprm[aidx, :] = 0.0
        model.actuator_biasprm[aidx, 1] = -kp
        model.actuator_biasprm[aidx, 2] = -kd
        model.actuator_ctrllimited[aidx] = 0
        model.actuator_forcelimited[aidx] = 1
        effort = V3_EFFORT[j_idx]
        model.actuator_forcerange[aidx, 0] = -effort
        model.actuator_forcerange[aidx, 1] = effort

    joint_qpos = {}
    joint_dof = {}
    joint_range = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name and name != "root":
            joint_qpos[name] = model.jnt_qposadr[i]
            joint_dof[name] = model.jnt_dofadr[i]
            if model.jnt_limited[i]:
                joint_range[name] = (float(model.jnt_range[i, 0]), float(model.jnt_range[i, 1]))
    usd_qpos = np.array([joint_qpos[n] for n in joint_names])
    usd_dof = np.array([joint_dof[n] for n in joint_names])

    foot_ids = [
        i for i in range(model.ngeom)
        if "ankle_roll_collision" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
        or "ankle_pitch_collision" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
    ]

    def _set_state_from_motion(frame_idx: int) -> None:
        """把机器人设到参考第 frame_idx 帧, 并做脚底贴地修正。"""
        data.qpos[:3] = ref_body_pos[frame_idx]
        data.qpos[3:7] = ref_quat[frame_idx]
        data.qpos[usd_qpos] = ref_pos[frame_idx]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        lowest_z = min(data.geom_xpos[g, 2] - model.geom_size[g, 2] for g in foot_ids)
        data.qpos[2] += 0.002 - lowest_z
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        # 默认关闭: 保持初始状态 = 训练参考第 0 帧 (Isaac 不校平脚底)。
        # v3 的 flatten_feet_init 会把踝 pitch 额外改 ~2°, 造成初始帧不一致。
        if args.flatten_feet:
            flatten_feet_init(model, data, joint_qpos)
            mujoco.mj_forward(model, data)

    _set_state_from_motion(0)
    print(f"[player] 初始 root_z={data.qpos[2]:.3f} (原参考 {ref_body_pos[0, 2]:.3f}, 脚底贴地)")

    last_action = np.zeros(21, dtype=np.float32)
    frame = 0
    policy_step = 0
    play_state = {"speed": float(args.speed), "paused": False}

    def on_key(keycode: int) -> None:
        if keycode == 32:  # Space
            play_state["paused"] = not play_state["paused"]
            print("[player] " + ("已暂停" if play_state["paused"] else "继续播放"))
        elif keycode in (91, 45):  # [ 或 -
            play_state["speed"] = max(0.1, play_state["speed"] * 0.8)
            print(f"[player] 速度 {play_state['speed']:.2f}x")
        elif keycode in (93, 61):  # ] 或 =
            play_state["speed"] = min(10.0, play_state["speed"] * 1.25)
            print(f"[player] 速度 {play_state['speed']:.2f}x")
        elif keycode in (ord("r"), ord("R")):
            play_state["frame_reset"] = True
            print("[player] 从头重播")

    print("[player] 快捷键: Space 暂停 | [ / ] 减速/加速 | R 重播 | Esc 退出")

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        start_time = time.time()
        while viewer.is_running() and frame < n_frames:
            if play_state.get("frame_reset", False):
                play_state["frame_reset"] = False
                frame = 0
                policy_step = 0
                start_time = time.time()
                last_action[:] = 0.0
                _set_state_from_motion(0)
                viewer.sync()
                continue

            if play_state["paused"]:
                time.sleep(0.02)
                continue

            if policy_step % DECIMATION == 0:
                q = data.qpos[usd_qpos].copy()
                qvel = data.qvel[usd_dof].copy()
                root_q = data.qpos[3:7].copy()

                obs_parts = [
                    root_rot_tan_norm(root_q),              # 6
                    data.qvel[3:6].copy(),                  # 3 世界系角速度
                    q,                                      # 21
                    qvel,                                   # 21
                ]
                for i in range(4):
                    idx = min(frame + i, n_frames - 1)
                    obs_parts.append(root_rot_tan_norm(ref_quat[idx]))  # 6*4=24
                obs_parts.append(
                    np.concatenate([ref_pos[min(frame + i, n_frames - 1)] for i in range(4)])
                )  # 84
                obs_parts.append(foot_contact_flags(model, data))  # 2
                obs = np.concatenate(obs_parts).astype(np.float32).reshape(1, -1)
                assert obs.shape[1] == 161, obs.shape

                out = sess.run(None, {"obs": obs})[0][0]
                if action_mode == "reference":
                    # q_des = 参考当前帧关节角 + action_scale * action
                    # (T800 风格: 不 clip, 除非 deploy_config 指定 action_clip)
                    last_action = out.astype(np.float32)
                    if action_clip is not None:
                        last_action = np.clip(last_action, -float(action_clip), float(action_clip))
                    q_des = ref_pos[frame].astype(np.float64) + action_scale * last_action
                else:
                    last_action = out.astype(np.float32)
                    q_des = default_q + action_scale * out
                if args.no_elbow_policy:
                    for j, name in enumerate(joint_names):
                        if "elbow" in name:
                            q_des[j] = float(ref_pos[frame, j])
                # 目标限位保护 (与成功包一致): 超限目标钳制到关节范围, 避免 PD 顶着限位硬推
                for j, name in enumerate(joint_names):
                    if name in joint_range:
                        lo, hi = joint_range[name]
                        q_des[j] = min(max(q_des[j], lo), hi)

            for _ in range(DECIMATION):
                data.ctrl[:] = q_des[ctrl_to_joint]
                mujoco.mj_step(model, data)

            frame += 1
            policy_step += 1
            # 同步画面到 viewer (否则窗口不刷新, 看起来没动)
            viewer.sync()
            if frame % 100 == 0:
                print(
                    f"[player] 帧 {frame}/{n_frames} ({frame/fps:.1f}s) "
                    f"根高度 {data.qpos[2]:.3f} 关节速度峰值 {np.abs(data.qvel[usd_dof]).max():.2f}"
                )

            # 实时速率: 用绝对时间戳精确限速, 避免 sleep 累积误差导致播放过快
            if play_state["speed"] > 0:
                target_time = start_time + frame / (POLICY_HZ * play_state["speed"])
                sleep = target_time - time.time()
                if sleep > 0:
                    time.sleep(sleep)

    print(f"[player] 完成 {frame} 帧 ({frame/fps:.1f}s)")


if __name__ == "__main__":
    main()
