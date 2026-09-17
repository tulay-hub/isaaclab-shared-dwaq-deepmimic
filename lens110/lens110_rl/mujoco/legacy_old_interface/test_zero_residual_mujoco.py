"""MuJoCo 零残差测试: target = 参考帧, 不推理策略, 记录稳定性。

用于系统排查 sim2sim 差异: 零动作下 MuJoCo 能否跟随参考姿态。
用法 (gmr 环境):
    python test_zero_residual_mujoco.py --mjcf assets/mjcf/lens110_21dof_sim_flatfoot.xml
"""

import argparse
import os

import mujoco
import numpy as np

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
CSV_JOINT_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "torso_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
]
USD_TO_CSV = [CSV_JOINT_ORDER.index(n) for n in USD_JOINT_NAMES]
CSV_TO_USD = [USD_JOINT_NAMES.index(n) for n in CSV_JOINT_ORDER]
DEFAULT_Q_USD = np.array([
    0.20553683, 0.21384868, -0.02090896, 0.1582034, -0.19115528, 0.6587707,
    0.62972623, 0.29305127, -0.30445874, 0.94194245, -1.1259768, -0.0530912,
    -0.06851172, -0.47967842, 0.5114301, 0.04094098, 0.05882506, -1.1318603,
    -1.114993, -0.1777105, 0.19954467,
], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjcf", required=True)
    parser.add_argument("--motion", required=True)
    parser.add_argument("--fix_frame0", action="store_true", help="固定 target=第0帧 (motion 不播放)")
    parser.add_argument("--physics_hz", type=float, default=500.0, help="物理频率")
    parser.add_argument("--solref0", type=float, default=None, help="接触 solref[0] 覆盖 (单因素)")
    parser.add_argument("--condim", type=int, default=None, help="接触 condim 覆盖 (单因素)")
    parser.add_argument("--armature", type=float, default=None, help="关节 armature 覆盖 (单因素)")
    parser.add_argument("--ankle_kp", type=float, default=None, help="踝 pitch/roll Kp 覆盖 (单因素, 诊断用)")
    parser.add_argument("--ankle_kd", type=float, default=None, help="踝 pitch/roll Kd 覆盖 (单因素, 诊断用)")
    parser.add_argument("--contact_trace", action="store_true", help="在每个 trace 点记录接触力/root 加速度")
    parser.add_argument("--trace", action="store_true", help="记录详细轨迹 (root/关节/接触)")
    parser.add_argument("--torso_mode", choices=["zero", "ref0", "refmax", "ref"],
                        default="zero",
                        help="腰目标: zero=官方零动作(默认, 对齐训练); "
                             "ref0=帧0参考角; refmax=运动最大腰角; ref=随 motion 播放")
    parser.add_argument("--pelvis_urdf_inertia", action="store_true",
                        help="单因素: 骨盆惯量换成 Isaac USD/URDF 值 "
                             "(mass 2.3981, com -5.8492e-06/8.49e-07/-0.024029)")
    parser.add_argument("--ground_clearance", type=float, default=0.002,
                        help="贴地修正后脚底最低点离地间隙 (默认 0.002=当前现状; "
                             "0=刚好贴地; 负值=轻微穿透)")
    parser.add_argument("--use_mesh_feet", action="store_true",
                        help="单因素: 脚底碰撞从 box 换成真实 ankle_roll STL mesh 凸包 "
                             "(对齐 Isaac convex_hull collider), box 改为纯视觉")
    parser.add_argument("--upright_pitch", action="store_true",
                        help="单因素: 初始 root 姿态去掉俯仰/横滚, 只保留 yaw (站直对照)")
    parser.add_argument("--joint_damping", type=float, default=None,
                        help="单因素: 所有关节阻尼覆盖 (默认 0.01; 成功包 XML 为 5~10)")
    parser.add_argument("--floor_friction", type=float, default=None,
                        help="单因素: 地面摩擦系数覆盖 (当前 0.7)")
    args = parser.parse_args()

    motion = np.load(args.motion)
    ref_pos_usd = motion["joint_pos"]
    ref_quat = motion["body_quat_w"][:, 0]
    ref_pos_w = motion["body_pos_w"][:, 0]
    n_frames = ref_pos_usd.shape[0]

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    data = mujoco.MjData(model)
    model.opt.timestep = 1.0 / args.physics_hz  # 默认 500Hz 对齐 Isaac

    # 与 sim2sim 脚本相同的 PD patch:
    #   motor -> gain FIXED + bias AFFINE, gainprm[0]=kp, biasprm[1]=-kp, biasprm[2]=-kd
    # 实机 dance PD: 髋/膝 40/5, 踝 20/20 (2026-08-18 由 10/10 调高), 腰/臂 100/5
    KP_USD = np.array([
        40, 40, 100, 40, 40, 100, 100, 40, 40, 100, 100,
        40, 40, 100, 100, 20, 20, 100, 100, 20, 20,
    ], dtype=np.float64)
    KD_USD = np.array([
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        5, 5, 5, 5, 20, 20, 5, 5, 20, 20,
    ], dtype=np.float64)
    act_name_to_idx = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
        for i in range(model.nu)
    }
    for usd_i, name in enumerate(USD_JOINT_NAMES):
        aidx = act_name_to_idx[name]
        kp, kd = KP_USD[usd_i], KD_USD[usd_i]
        if args.ankle_kp is not None and name.endswith("_ankle_pitch_joint"):
            kp = args.ankle_kp
        if args.ankle_kd is not None and name.endswith("_ankle_pitch_joint"):
            kd = args.ankle_kd
        if args.ankle_kp is not None and name.endswith("_ankle_roll_joint"):
            kp = args.ankle_kp
        if args.ankle_kd is not None and name.endswith("_ankle_roll_joint"):
            kd = args.ankle_kd
        model.actuator_gaintype[aidx] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_biastype[aidx] = mujoco.mjtBias.mjBIAS_AFFINE
        model.actuator_gainprm[aidx, :] = 0.0
        model.actuator_gainprm[aidx, 0] = kp
        model.actuator_biasprm[aidx, :] = 0.0
        model.actuator_biasprm[aidx, 1] = -kp
        model.actuator_biasprm[aidx, 2] = -kd
    if args.ankle_kp is not None:
        print(f"[单因素] 踝 Kp -> {args.ankle_kp}")
    if args.ankle_kd is not None:
        print(f"[单因素] 踝 Kd -> {args.ankle_kd}")
    print("[zero-residual] 已应用实机 dance PD (motor->position PD)")
    if args.solref0 is not None:
        for gid in range(model.ngeom):
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            if gname in ("floor", "ground", "plane") or (gname and "collision" in gname):
                model.geom_solref[gid, 0] = args.solref0
        print(f"[单因素] solref[0] -> {args.solref0}")
    if args.condim is not None:
        for gid in range(model.ngeom):
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            if gname in ("floor", "ground", "plane") or (gname and "collision" in gname):
                model.geom_condim[gid] = args.condim
        print(f"[单因素] condim -> {args.condim}")
    if args.armature is not None:
        for jid in range(model.njnt):
            if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
                model.dof_armature[model.jnt_dofadr[jid]] = args.armature
        print(f"[单因素] armature -> {args.armature}")
    if args.pelvis_urdf_inertia:
        pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if pid < 0:
            raise RuntimeError("模型中没有 pelvis body")
        model.body_mass[pid] = 2.3981
        model.body_ipos[pid] = (-5.8492e-06, 8.4885e-07, -0.024029)
        model.body_iquat[pid] = (1.0, 0.0, 0.0, 0.0)  # 用完整张量, 不做主轴分解
        # URDF 非对角项 < 2e-6, 忽略; 主轴分量 (ixx, iyy, izz)
        model.body_inertia[pid] = (0.002725, 0.0024552, 0.002601)
        print("[单因素] pelvis 惯量 -> Isaac USD/URDF 值 (2.3981 kg)")
    if args.use_mesh_feet:
        for gid in range(model.ngeom):
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            if not gname:
                continue
            if gname.endswith("_collision"):
                # box 碰撞体改为纯视觉 (不参与碰撞)
                model.geom_contype[gid] = 0
                model.geom_conaffinity[gid] = 0
            elif gname in ("left_ankle_roll_link", "right_ankle_roll_link"):
                # 真实脚型 mesh 参与碰撞 (MuJoCo 自动使用凸包)
                model.geom_contype[gid] = 1
                model.geom_conaffinity[gid] = 1
        print("[单因素] 脚底碰撞: box -> 真实 ankle_roll mesh 凸包")
    if args.joint_damping is not None:
        for jid in range(model.njnt):
            if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
                model.dof_damping[model.jnt_dofadr[jid]] = args.joint_damping
        print(f"[单因素] 关节阻尼 -> {args.joint_damping}")
    if args.floor_friction is not None:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        model.geom_friction[gid, 0] = args.floor_friction
        print(f"[单因素] 地面摩擦 -> {args.floor_friction}")

    joint_qpos = {}
    joint_dof = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name and name not in ("root", "floating_base_joint"):
            joint_qpos[name] = model.jnt_qposadr[i]
            joint_dof[name] = model.jnt_dofadr[i]
    csv_qpos = np.array([joint_qpos[n] for n in CSV_JOINT_ORDER])
    csv_dof = np.array([joint_dof[n] for n in CSV_JOINT_ORDER])

    # 初始状态 = 参考第0帧 + 贴地
    data.qpos[:3] = ref_pos_w[0]
    data.qpos[3:7] = ref_quat[0]
    if args.upright_pitch:
        qw, qx, qy, qz = ref_quat[0]
        # 提取绕 z 的 yaw: 将 quat 的 x/y 分量置零并归一
        yaw_norm = np.hypot(qw, qz)
        data.qpos[3:7] = (qw / yaw_norm, 0.0, 0.0, qz / yaw_norm)
        print("[单因素] 初始 root 姿态 -> 站直 (只保留 yaw)")
    data.qpos[csv_qpos] = ref_pos_usd[0][CSV_TO_USD]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    low = 1e9
    for gname in ["left_ankle_roll_collision", "right_ankle_roll_collision",
                  "left_ankle_pitch_collision", "right_ankle_pitch_collision"]:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        if gid >= 0:
            p = data.geom_xpos[gid]
            mat = data.geom_xmat[gid].reshape(3, 3)
            sz = model.geom_size[gid]
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for szz in (-1, 1):
                        low = min(low, float((p + mat @ (sz * np.array([sx, sy, szz])))[2]))
    data.qpos[2] += args.ground_clearance - low
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    print(f"[zero-residual] 贴地修正后 root_z={data.qpos[2]:.4f} "
          f"脚底最低点离地间隙={args.ground_clearance:.4f}")

    # 足部碰撞 geom 名 (含 ankle_roll/pitch_collision)
    foot_geom_ids = []
    for gid in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if gname and ("ankle_roll_collision" in gname or "ankle_pitch_collision" in gname):
            foot_geom_ids.append(gid)
    contact_force = np.zeros(6, dtype=np.float64)

    # 零残差 (等价训练零动作):
    #   腿 q_des = 0*scale + default (default=参考帧0腿角)
    #   腰 q_des = 0*scale (不加 default!)
    #   臂 q_des = 参考帧臂角 (开环)
    default_q_csv = DEFAULT_Q_USD[CSV_TO_USD]
    frame = 0
    decim = int(round(1.0 / (100.0 * model.opt.timestep)))
    print(f"[zero-residual] timestep={model.opt.timestep}, decim={decim}, 初始 root_z={data.qpos[2]:.4f}")
    for step in range(decim * n_frames):
        if step % decim == 0:
            frame = min(frame, n_frames - 1)
            target = default_q_csv.copy()
            if args.torso_mode == "zero":
                target[12] = 0.0  # 腰: 官方零动作不加 default
            elif args.torso_mode == "ref0":
                target[12] = ref_pos_usd[0][CSV_TO_USD][12]
            elif args.torso_mode == "refmax":
                target[12] = float(ref_pos_usd[:, CSV_TO_USD[12]].max())
            else:  # ref: 随 motion 播放
                target[12] = ref_pos_usd[frame][CSV_TO_USD][12]
            if args.fix_frame0:
                # 固定第0帧: 腿/腰/臂都锁定参考第0帧 (不推进 motion)
                target[13:21] = ref_pos_usd[0][CSV_TO_USD][13:21]
            else:
                target[13:21] = ref_pos_usd[frame][CSV_TO_USD][13:21]  # 臂开环参考
            data.ctrl[:] = target
            if not args.fix_frame0:
                frame += 1
        mujoco.mj_step(model, data)
        t = (step + 1) * model.opt.timestep
        trace_points = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0)
        if args.trace and any(abs(t - tp) < 1e-9 for tp in trace_points):
            q = data.qpos[csv_qpos]
            dq = data.qvel[csv_dof]
            target = data.ctrl[csv_qpos - csv_qpos[0]] if model.nu == 21 else data.ctrl
            print(f"  [t={t:.2f}] root_z={data.qpos[2]:.4f} "
                  f"ank_p_L q={q[4]:+.4f}/target={data.ctrl[4]:+.4f} "
                  f"ank_p_R q={q[10]:+.4f}/target={data.ctrl[10]:+.4f} "
                  f"knee_L q={q[3]:+.4f}/target={data.ctrl[3]:+.4f}")
            print(f"          dq_ank_p_L={dq[4]:+.4f} | max|dq|={np.abs(dq).max():.4f}")
            if args.contact_trace:
                ncon = data.ncon
                max_f = 0.0
                foot_fn = 0.0
                foot_ft = 0.0
                foot_names = []
                for ci in range(ncon):
                    c = data.contact[ci]
                    mujoco.mj_contactForce(model, data, ci, contact_force)
                    f_mag = float(np.linalg.norm(contact_force[:3]))
                    if f_mag > max_f:
                        max_f = f_mag
                    g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
                    g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
                    is_foot = (c.geom1 in foot_geom_ids) or (c.geom2 in foot_geom_ids)
                    if is_foot:
                        fn = abs(float(contact_force[2]))
                        ft = float(np.linalg.norm(contact_force[:2]))
                        foot_fn += fn
                        foot_ft += ft
                        foot_names.append(f"{g1}<->{g2}:{fn:.1f}/{ft:.1f}")
                root_a = np.linalg.norm(data.qacc[0:3])
                max_tau = float(np.abs(data.qfrc_actuator[6:]).max())
                print(f"          ncon={ncon} max_contact_F={max_f:.1f} "
                      f"foot_Fn={foot_fn:.1f} foot_Ft={foot_ft:.1f} "
                      f"|root_a|={root_a:.2f} max_tau={max_tau:.1f}")
                if foot_names:
                    print(f"          feet: {', '.join(foot_names[:6])}")
        if t in (0.5, 1.0, 2.0, 5.0):
            maxq = float(np.abs(data.qvel[6:]).max())
            print(f"t={t:.1f}s root_z={data.qpos[2]:.4f} max_joint_vel={maxq:.3f}")
        if t >= 5.0:
            break
    print(f"[zero-residual] 完成 {t:.1f}s")


if __name__ == "__main__":
    main()
