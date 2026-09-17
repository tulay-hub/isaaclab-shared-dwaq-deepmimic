"""MuJoCo sim2sim (官方式 dance 模式): 用训练好的 ONNX 策略做闭环推理验证。

对齐 Lens110LabOfficialEnvCfg:
  - 策略 100Hz / 物理 500Hz (decimation 5)
  - 观测 126 维: state(45) + future(81)
      state : [13q + 13dq + 13prev_action + 3角速度(body) + 3gravity_b]
      future: 3 帧 (k/k+10/k+20) x (21 关节 CSV 顺序 + 6 root euler3+pos3)
  - 动作 13 维 (12腿 + 1腰):
      q_des[腿] = action[0:12] * 0.25 + default[腿]
      q_des[腰] = action[12] * 0.25
      q_des[臂] = 参考关节角 (开环, 不经策略)
  - PD: kp/kd 与训练一致 (腿120/4 髋yaw55/2 踝55/2 腰45/1.5 臂35/1.2)

用法 (gmr 环境, 需 mujoco + onnxruntime):
    python mujoco_sim2sim_official.py --onnx policy.onnx \
        --motion motion/lens110_amp_100hz_flat.npz \
        --mjcf assets/mjcf/lens110_21dof.xml
"""

import argparse
import os
import time

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort

# ---------------------------------------------------------------------------
# 关节顺序定义
# ---------------------------------------------------------------------------
# npz / USD 资产顺序 (21 关节, 与 lens110.py init_state / 旧导出脚本一致)
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

# 官方 CSV 顺序 = MuJoCo 模型关节顺序 (12 腿 + 1 腰 + 8 臂)
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

# usd_idx -> csv_idx (npz 顺序 -> MuJoCo/官方 CSV 顺序)
USD_TO_CSV = [CSV_JOINT_ORDER.index(name) for name in USD_JOINT_NAMES]
CSV_TO_USD = [USD_JOINT_NAMES.index(name) for name in CSV_JOINT_ORDER]

# ---------------------------------------------------------------------------
# 训练超参 (与资产配置 / 官方 yaml 一致)
# ---------------------------------------------------------------------------
ACTION_SCALE = 0.25
POLICY_HZ = 100.0
PHYSICS_HZ = 500.0
DECIMATION = 5  # 会被 main() 里按模型 timestep 动态覆盖
CLIP_OBS = 50.0
CLIP_ACT = 1.0

# 训练 kp/kd (USD 顺序) = 实机 dance PD:
#   髋/膝 40/5 | 踝 pitch/roll 20/20 (2026-08-18 由 10/10 调高)
#   | 腰 100/5 | 肩/肘 100/5
KP = np.array([
    40, 40, 100, 40, 40, 100, 100, 40, 40, 100, 100, 40, 40, 100, 100, 20, 20, 100, 100, 20, 20,
], dtype=np.float64)
KD = np.array([
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 20, 20, 5, 5, 20, 20,
], dtype=np.float64)

# stand 高 PD (训练分段 stand 段, 与 lens110_env.py _PD_STAND_* 一致)
STAND_KP = np.array([
    400, 400, 100, 500, 500, 100, 100, 300, 300, 100, 100, 400, 400, 100, 100, 200, 200, 100, 100, 200, 200,
], dtype=np.float64)
STAND_KD = np.array([
    5, 5, 5, 5, 5, 2, 2, 5, 5, 2, 2, 5, 5, 2, 2, 10, 10, 2, 2, 10, 10,
], dtype=np.float64)
# 分段 stand PD 帧区间 (与训练 env cfg 一致)
STAND_START_FRAMES = 650
STAND_END_FRAME = 3250
STAND_BLEND_FRAMES = 50
# 2026-08-18 咨询结果: 别人未使用分段 PD, 默认关闭 (代码留存)
STAND_SEGMENTS_ENABLED = False

# 默认关节角 (USD 顺序, 来自资产 init_state = npz 首帧)
DEFAULT_Q_USD = np.array([
    0.20553683, 0.21384868, -0.02090896, 0.1582034, -0.19115528, 0.6587707,
    0.62972623, 0.29305127, -0.30445874, 0.94194245, -1.1259768, -0.0530912,
    -0.06851172, -0.47967842, 0.5114301, 0.04094098, 0.05882506, -1.1318603,
    -1.114993, -0.1777105, 0.19954467,
], dtype=np.float64)


# ---------------------------------------------------------------------------
# 数学工具 (与 isaaclab 对齐)
# ---------------------------------------------------------------------------
def quat_inv_wxyz(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rotate_wxyz(q, v):
    """用 wxyz 四元数旋转向量 v (世界->body 用 inv)。"""
    w, x, y, z = q
    t2 = 2 * np.cross(np.array([x, y, z]), v)
    return v + w * t2 + np.cross(np.array([x, y, z]), t2)


def quat_mul_wxyz(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def euler_xyz_from_quat(q):
    """wxyz 四元数 -> XYZ extrinsic 欧拉角 (roll, pitch, yaw), 对齐 isaaclab。"""
    w, x, y, z = q
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = np.where(np.abs(sin_pitch) >= 1.0, np.sign(sin_pitch) * np.pi / 2.0, np.arcsin(sin_pitch))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=np.float64)


def quat_apply_inverse(q, v):
    """R^T @ v, 对齐 isaaclab quat_apply_inverse (wxyz)。"""
    w, x, y, z = q
    xyz = np.array([x, y, z])
    t = 2.0 * np.cross(xyz, v)
    return v - w * t + np.cross(xyz, t)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--motion", required=True)
    parser.add_argument(
        "--mjcf",
        default=os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                         "assets", "mjcf", "lens110_21dof_sim_flatfoot.xml")
        ),
        help="MJCF 模型路径 (默认 assets/mjcf/lens110_21dof_sim_flatfoot.xml)",
    )
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    motion = np.load(args.motion)
    fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    ref_pos_usd = motion["joint_pos"]   # (N,21) USD 顺序
    ref_vel_usd = motion["joint_vel"]
    ref_quat = motion["body_quat_w"][:, 0]  # (N,4) wxyz pelvis
    ref_pos_w = motion["body_pos_w"][:, 0]  # (N,3) pelvis
    ref_lin_vel_w = motion["body_lin_vel_w"][:, 0]
    ref_ang_vel_w = motion["body_ang_vel_w"][:, 0]
    n_frames = ref_pos_usd.shape[0]
    print(f"[sim2sim] 动作 {n_frames} 帧 @{fps:.0f}Hz ({n_frames/fps:.1f}s)")

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    data = mujoco.MjData(model)
    # 训练环境 500Hz 物理; 但 sim_flatfoot 等官方模型自带 timestep=0.001 (1000Hz)
    # 且 Newton 求解器参数是验证过的, 尊重模型默认, 只保证策略 100Hz
    # (decimation 按模型 timestep 计算, 见下)
    # 按模型 timestep 计算 decimation, 保证策略 100Hz:
    #   dt=0.002 (500Hz) -> decimation 5; dt=0.001 (1000Hz) -> decimation 10
    global DECIMATION
    DECIMATION = max(1, int(round(1.0 / (POLICY_HZ * model.opt.timestep))))
    print(f"[sim2sim] 模型 timestep={model.opt.timestep}, decimation={DECIMATION} "
          f"(物理 {1.0/model.opt.timestep:.0f}Hz / 策略 {POLICY_HZ:.0f}Hz)")
    # 模型自带 Newton 求解器 (sim_flatfoot: Newton 50, tol 1e-10) 则保留;
    # 播放器模型默认 PGS 100, 强 PD 需要更高精度 -> 提升迭代
    if model.opt.solver != mujoco.mjtSolver.mjSOL_NEWTON:
        model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        model.opt.iterations = 300
        model.opt.tolerance = 1e-9
    # 运行时把 position actuator 的 kp/kv 改成训练值
    # MuJoCo position actuator: gainprm = [kp, kv, 0], biasprm = [-kp, -kv, 0]
    act_name_to_idx = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
        for i in range(model.nu)
    }
    # ctrl 索引 (MuJoCo = CSV 顺序) -> USD 索引
    ctrl_to_usd = np.array([
        USD_JOINT_NAMES.index(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))
        for i in range(model.nu)
    ], dtype=int)

    # ---- 动力学 patch (参考 Lens110_ThreeAction 成功部署包) ----
    # 1) 脚底碰撞盒保持 MJCF 原始配置
    #    = 我们训练资产 (lens110_21dof.usd, 从 URDF 转换) 的脚底碰撞:
    #      URDF box origin (0.0353, ±0.001, -0.015), size 全长 (0.195, 0.091, 0.035)
    #      MuJoCo 半长 (0.0975, 0.0455, 0.0175)
    #    注意: 不能照抄成功包的 TRAINING_FOOT_BOXES (那是他们训练环境的脚底,
    #          与我们的训练资产不一致, 会导致接触点偏移、策略失稳)。
    foot_geom_names = [
        "left_ankle_roll_collision", "right_ankle_roll_collision",
        "left_ankle_pitch_collision", "right_ankle_pitch_collision",
    ]
    # 模型是否自带质心偏移 (sim_flatfoot/real_flatfoot 有, 播放器模型没有)
    _has_nonzero_ipos = any(np.abs(model.body_ipos[i]).max() > 1e-8 for i in range(1, model.nbody))

    # 2) 接触配置: 对齐 Isaac
    #    - Isaac contact_offset=0.04, MuJoCo 默认 margin=0.001 接触太晚
    #      -> 给地板和脚底盒加大 margin (接触检测距离)
    #    - 训练资产 enabled_self_collisions=True, 但 MuJoCo mesh 是凸包近似,
    #      保持 mesh 与地面碰撞 (contype=1/conaffinity=0), 关闭 mesh 自碰撞
    #      (与成功包一致, 避免凸包近似导致额外抖动)
    foot_geom_ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in foot_geom_names}
    foot_geom_ids.discard(-1)
    for gid in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if gname in {"floor", "ground", "plane"} or (gid in foot_geom_ids):
            # 地板/脚底: 参与碰撞 + 加大接触检测距离 (对齐 Isaac contact_offset)
            model.geom_contype[gid] = 1
            model.geom_conaffinity[gid] = 15
            model.geom_margin[gid] = 0.04
        else:
            # 其余 mesh: 与地面碰撞, 不参与自碰撞 (conaffinity=0)
            model.geom_contype[gid] = 1
            model.geom_conaffinity[gid] = 0
            model.geom_margin[gid] = 0.001
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id >= 0:
        # Isaac 地面材质: static=0.7, dynamic=0.7, combine=average (与训练一致)
        # MuJoCo friction = [sliding, torsional, rolling], 滑动摩擦用 0.7 对齐
        model.geom_friction[floor_id] = [0.7, 0.005, 0.0001]

    # 3) 关节: 播放器模型对齐 Isaac (无关节阻尼/摩擦/armature)
    #    - kd 由 actuator biasprm 提供, dof_damping 必须为 0 避免双重阻尼
    #    - 但 sim_flatfoot 等官方模型自带 damping/frictionloss/armature=0.01,
    #      对 MuJoCo Newton 求解稳定性是必须的 (清零会数值爆炸), 保留
    if not _has_nonzero_ipos and model.opt.solver != mujoco.mjtSolver.mjSOL_NEWTON:
        for jid in range(model.njnt):
            if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            did = model.jnt_dofadr[jid]
            model.dof_damping[did] = 0.0
            model.dof_frictionloss[did] = 0.0
            model.dof_armature[did] = 0.0

    # 5) 关节限位对齐 Isaac 训练资产 (lens110_21dof.usd)
    #    关键: sim_flatfoot 膝盖限位 [-0.087, 0.9] 远窄于 Isaac [-0.087, 2.443],
    #    舞蹈动作膝盖弯曲会被截断导致站不起/蹲下。其余关节限位两边一致。
    _joint_limit_isaac = {
        ".*_knee_joint": (-0.087, 2.443),
    }
    import re as _re
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        for pat, rng in _joint_limit_isaac.items():
            if _re.fullmatch(pat, name):
                model.jnt_range[jid] = rng
                print(f"[sim2sim] 关节限位对齐: {name} {model.jnt_range[jid]}")

    # 4) 质心对齐 Isaac: 若模型自带惯性偏移 (如 sim_flatfoot / real_flatfoot,
    #    从 URDF 转换时已写入), 则直接用; 若播放器模型 ipos 全 0 (质心在关节原点),
    #    则从 URDF 读取 inertial origin 写入 body_ipos。
    if _has_nonzero_ipos:
        print("[sim2sim] 模型已自带质心偏移, 直接使用")
    else:
        _urdf_path = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "..",
                         "资料", "lens110_real_robot_test_bundle_v29", "03_ROBOT_MODEL",
                         "lens110_21dof_current.urdf")
        )
        if os.path.isfile(_urdf_path):
            import xml.etree.ElementTree as ET
            _tree = ET.parse(_urdf_path)
            _ipos_map = {}
            for _link in _tree.getroot().iter("link"):
                _iner = _link.find("inertial")
                if _iner is not None:
                    _o = _iner.find("origin")
                    _xyz = np.array([float(v) for v in _o.get("xyz").split()]) if _o is not None else np.zeros(3)
                    _ipos_map[_link.get("name")] = _xyz
            for _i in range(1, model.nbody):
                _name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, _i)
                model.body_ipos[_i] = _ipos_map.get(_name, np.zeros(3))
            print("[sim2sim] 已从 URDF 写入质心偏移 (对齐 Isaac 训练资产)")
        else:
            print(f"[sim2sim] 未找到 URDF: {_urdf_path}, 保持 MuJoCo 默认质心")

    # 关节点名 -> qpos/qvel 索引
    joint_qpos = {}
    joint_dof = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name and name != "root":
            joint_qpos[name] = model.jnt_qposadr[i]
            joint_dof[name] = model.jnt_dofadr[i]
    # MuJoCo 关节顺序 == CSV_JOINT_ORDER
    csv_qpos = np.array([joint_qpos[n] for n in CSV_JOINT_ORDER])
    csv_dof = np.array([joint_dof[n] for n in CSV_JOINT_ORDER])

    # 4) actuator: gainprm[0]=kp, biasprm[1]=-kp, biasprm[2]=-kd
    def apply_pd(kp_arr: np.ndarray, kd_arr: np.ndarray) -> None:
        """把 kp/kd (USD 顺序) 写入 actuator, 与实机 effort 一致。"""
        for usd_i, name in enumerate(USD_JOINT_NAMES):
            aidx = act_name_to_idx[name]
            kp, kd = kp_arr[usd_i], kd_arr[usd_i]
            model.actuator_gaintype[aidx] = mujoco.mjtGain.mjGAIN_FIXED
            model.actuator_biastype[aidx] = mujoco.mjtBias.mjBIAS_AFFINE
            model.actuator_gainprm[aidx, :] = 0.0
            model.actuator_gainprm[aidx, 0] = kp
            model.actuator_biasprm[aidx, :] = 0.0
            model.actuator_biasprm[aidx, 1] = -kp
            model.actuator_biasprm[aidx, 2] = -kd
            effort = 80.0 if usd_i in (0, 1, 3, 4, 11, 12) else 36.0
            model.actuator_forcelimited[aidx] = 1
            model.actuator_forcerange[aidx, 0] = -effort
            model.actuator_forcerange[aidx, 1] = effort

    apply_pd(KP, KD)
    _last_pd_blend = -1.0

    def update_pd_for_frame(frame_idx: int) -> None:
        """按参考帧在 stand / 动态弱 PD 之间平滑切换 (与训练分段一致)。"""
        nonlocal _last_pd_blend
        if not STAND_SEGMENTS_ENABLED:
            if _last_pd_blend != 1.0:
                apply_pd(KP, KD)
                _last_pd_blend = 1.0
            return
        b = float(STAND_BLEND_FRAMES)
        start = float(STAND_START_FRAMES)
        end = float(STAND_END_FRAME)
        f = float(frame_idx)
        if f < start - b:
            blend = 0.0
        elif f < start:
            blend = (f - (start - b)) / b
        elif f < end:
            blend = 1.0
        elif f < end + b:
            blend = 1.0 - (f - end) / b
        else:
            blend = 0.0
        blend = min(1.0, max(0.0, blend))
        if abs(blend - _last_pd_blend) < 1e-6:
            return
        kp_arr = STAND_KP + (KP - STAND_KP) * blend
        kd_arr = STAND_KD + (KD - STAND_KD) * blend
        apply_pd(kp_arr, kd_arr)
        _last_pd_blend = blend

    # 初始状态: 参考第 0 帧 (位置/朝向/关节角/关节速度/根速度, 与训练 reset 一致)
    def _set_state_from_motion(frame_idx: int, set_vel: bool) -> None:
        data.qpos[:3] = ref_pos_w[frame_idx]
        data.qpos[3:7] = ref_quat[frame_idx]
        # MuJoCo qpos 关节顺序 = CSV 顺序, 直接按 CSV_TO_USD 取 npz(USD 顺序) 对应值
        data.qpos[csv_qpos] = ref_pos_usd[frame_idx][CSV_TO_USD]
        data.qvel[:] = 0.0
        if set_vel:
            data.qvel[:3] = ref_lin_vel_w[frame_idx]
            data.qvel[3:6] = ref_ang_vel_w[frame_idx]
            data.qvel[csv_dof] = ref_vel_usd[frame_idx][CSV_TO_USD]

    _set_state_from_motion(0, set_vel=True)
    mujoco.mj_forward(model, data)
    # 对齐脚底贴地: MuJoCo 模型几何与 Isaac 略有差异, 参考高度下脚会浮空,
    # 先把最低脚底放到地面 (与训练 reset 的"贴地"一致)。
    foot_geom_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in foot_geom_names]
    def _geom_lowest_z(geom_id: int) -> float:
        pos = data.geom_xpos[geom_id]
        mat = data.geom_xmat[geom_id].reshape(3, 3)
        size = model.geom_size[geom_id]
        gtype = model.geom_type[geom_id]
        if gtype == mujoco.mjtGeom.mjGEOM_BOX:
            return float(pos[2] - np.abs(mat[2, :]) @ size)
        if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            return float(pos[2] - size[0])
        if gtype in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
            return float(pos[2] - (abs(mat[2, 0]) * size[0] + abs(mat[2, 1]) * size[0] + abs(mat[2, 2]) * size[1]))
        return float(pos[2])
    lowest_z = min(_geom_lowest_z(g) for g in foot_geom_ids)
    data.qpos[2] += 0.002 - lowest_z
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    print(f"[sim2sim] 初始 root_z={data.qpos[2]:.3f} (脚底贴地, 原参考 {ref_pos_w[0,2]:.3f})")

    # 默认关节角 (MuJoCo/CSV 顺序)
    default_q_csv = DEFAULT_Q_USD[CSV_TO_USD]

    last_action = np.zeros(13, dtype=np.float32)
    frame = 0
    policy_step = 0
    prev_time = time.time()
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
        elif keycode == ord("r") or keycode == ord("R"):
            play_state["frame_reset"] = True
            print("[player] 从头重播")

    print(f"[player] 快捷键: Space 暂停 | [ / ] 减速/加速 | R 重播 | Esc 退出")

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        start_time = time.time()
        while viewer.is_running() and frame < n_frames:
            if play_state.get("frame_reset", False):
                play_state["frame_reset"] = False
                frame = 0
                policy_step = 0
                start_time = time.time()
                last_action[:] = 0.0
                _set_state_from_motion(0, set_vel=False)
                mujoco.mj_forward(model, data)
                # 贴地修正
                lowest_z = min(_geom_lowest_z(g) for g in foot_geom_ids)
                data.qpos[2] += 0.002 - lowest_z
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                continue

            if play_state["paused"]:
                time.sleep(0.02)
                continue

            # --- 策略推理 (100Hz) ---
            if policy_step % DECIMATION == 0:
                q_csv = data.qpos[csv_qpos].copy()          # 21 CSV 顺序
                qvel_csv = data.qvel[csv_dof].copy()
                robot_q_wxyz = data.qpos[3:7].copy()

                # 训练观测索引: official_state_obs 用 usd_to_csv = [csv_to_usd.index(i)]
                # 即 USD 第 j 个关节在 CSV 中的位置, 再取前 13。
                # MuJoCo q 是 CSV 顺序 -> 先转回 USD 顺序, 再按训练 usd_to_csv 索引。
                q_usd = q_csv[USD_TO_CSV]      # (21,) USD 顺序
                qvel_usd = qvel_csv[USD_TO_CSV]
                q_train = q_usd[USD_TO_CSV]    # 与 robot.data.joint_pos[:, usd_to_csv] 一致
                qvel_train = qvel_usd[USD_TO_CSV]

                # state: [13q + 13dq + 13prev + 3ω + 3gravity_b]
                base_ang_vel = quat_rotate_wxyz(quat_inv_wxyz(robot_q_wxyz), data.qvel[3:6])
                gravity_b = quat_apply_inverse(robot_q_wxyz, np.array([0.0, 0.0, -1.0]))
                state = np.concatenate([
                    q_train[:13].astype(np.float32),
                    qvel_train[:13].astype(np.float32),
                    last_action,
                    base_ang_vel.astype(np.float32),
                    gravity_b.astype(np.float32),
                ])

                # future: 3 帧 (k/k+10/k+20) x (21关节 CSV + 6 root)
                offsets = np.array([0, 10, 20])
                frames = np.clip(frame + offsets, 0, n_frames - 1)
                # 与训练 official_future_ref 一致: npz(USD) 按 usd_to_csv 索引
                joints = ref_pos_usd[frames][:, USD_TO_CSV]  # (3,21)
                roots = np.stack([
                    np.concatenate([euler_xyz_from_quat(ref_quat[f]), ref_pos_w[f]])
                    for f in frames
                ])                                          # (3,6)
                future = np.concatenate([joints.reshape(-1), roots.reshape(-1)]).astype(np.float32)

                obs = np.clip(np.concatenate([state, future]), -CLIP_OBS, CLIP_OBS)
                assert obs.shape[0] == 126, obs.shape
                out = sess.run(None, {"obs": obs.reshape(1, -1)})[0][0].astype(np.float32)
                out = np.clip(out, -CLIP_ACT, CLIP_ACT)
                last_action = out

                # 官方动作: 腿 = a*0.25 + default, 腰 = a*0.25, 臂 = 参考开环
                q_des_csv = default_q_csv.copy()
                q_des_csv[:12] = ACTION_SCALE * out[:12] + default_q_csv[:12]
                q_des_csv[12] = ACTION_SCALE * out[12]
                arm_usd = [USD_JOINT_NAMES.index(n) for n in CSV_JOINT_ORDER[13:21]]
                q_des_csv[13:21] = ref_pos_usd[frame][arm_usd]
                target_q = q_des_csv

            # --- 物理步 (500Hz, 5 次 = 1 策略步) ---
            for _ in range(DECIMATION):
                data.ctrl[:] = target_q
                mujoco.mj_step(model, data)

            frame += 1
            policy_step += 1
            update_pd_for_frame(frame)
            # 同步画面到 viewer (否则窗口不刷新, 看起来没动)
            viewer.sync()
            if frame % 100 == 0:
                print(f"[sim2sim] 帧 {frame}/{n_frames} ({frame/fps:.1f}s) "
                      f"根高度 {data.qpos[2]:.3f} 关节速度峰值 {np.abs(data.qvel[csv_dof]).max():.2f}")

            # 实时速率: 用绝对时间戳精确限速, 避免 sleep 累积误差导致播放过快
            if args.speed > 0:
                target_time = start_time + frame / (POLICY_HZ * play_state["speed"])
                sleep = target_time - time.time()
                if sleep > 0:
                    time.sleep(sleep)

    print(f"[sim2sim] 完成 {frame} 帧")


if __name__ == "__main__":
    main()
