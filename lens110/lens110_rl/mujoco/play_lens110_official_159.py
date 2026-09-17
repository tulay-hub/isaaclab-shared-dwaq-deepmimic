"""Lens110 159 维 URDF 顺序策略的官方播放器适配版。

基于 Lens110_Dance_Sim2Real_v2_20260825/replay/mujoco_sim2sim_official.py 的环境处理
(官方模型 lens110_21dof.xml, 保留官方脚底盒/接触/摩擦/膝盖限位/质心),
但观测换成新的 159 维契约, 动作换成 21 维 URDF 顺序参考中心残差:
    q_des = 参考当前帧关节角 + 0.25 * action (不 clip)
PD 默认从 deploy_config.yaml 读取 (训练 PD), 也可用官方强 PD。

Aligned 版额外做 Isaac 训练 URDF 动力学对齐:
- 写入 URDF link 质心偏移和主惯性矩;
- 用 URDF 关节限位覆盖 MJCF;
- 默认清除 MJCF 自带 passive joint damping/frictionloss, 避免与训练 PD 的 kd 重复计;
- 默认按训练 ImplicitActuator 的 armature 补 0.01 (torso_yaw 除外)。
"""

import argparse
import xml.etree.ElementTree as ET
import os
import time

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import yaml


POLICY_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "torso_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
]

# 动作数据集保持 Isaac/USD 交叉顺序不变。
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

USD_TO_POLICY = [USD_JOINT_NAMES.index(name) for name in POLICY_JOINT_NAMES]

# 官方强 PD (URDF 策略顺序), 仅在未提供 deploy 时使用
OFFICIAL_KP = np.array([
    40, 40, 40, 40, 55, 55,
    40, 40, 40, 40, 55, 55,
    100,
    20, 20, 20, 20,
    20, 20, 20, 20,
], dtype=np.float64)
OFFICIAL_KD = np.array([
    5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5,
    5,
    5, 5, 5, 5,
    5, 5, 5, 5,
], dtype=np.float64)

POLICY_HZ = 100.0
DEFAULT_URDF = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "..", "..", "legged_lab_lbot", "source", "legged_lab", "legged_lab", "data",
    "Robots", "model_humanoid_lens110", "lens110_21dof.urdf",
))
FOOT_BOX_NAMES = [
    "left_ankle_roll_collision", "right_ankle_roll_collision",
    "left_ankle_pitch_collision", "right_ankle_pitch_collision",
]
FOOT_ROLL_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]


def _vec3(s):
    return np.zeros(3) if not s else np.array([float(v) for v in s.split()], dtype=np.float64)


def apply_urdf_inertial_and_limits(model, urdf_path, armature=0.01, clear_passive_damping=True):
    """Align MuJoCo inertial/limit parameters with the Isaac training URDF."""
    root = ET.parse(urdf_path).getroot()
    link_inertial = {}
    for link in root.findall("link"):
        name = link.get("name")
        inertial = link.find("inertial")
        if not name or inertial is None:
            continue
        origin = inertial.find("origin")
        pos = _vec3(origin.get("xyz")) if origin is not None else np.zeros(3, dtype=np.float64)
        ii = inertial.find("inertia")
        if ii is None:
            inertia = np.zeros((3, 3), dtype=np.float64)
        else:
            ixx = float(ii.get("ixx", 0.0)); iyy = float(ii.get("iyy", 0.0)); izz = float(ii.get("izz", 0.0))
            ixy = float(ii.get("ixy", 0.0)); ixz = float(ii.get("ixz", 0.0)); iyz = float(ii.get("iyz", 0.0))
            inertia = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=np.float64)
        principal = np.linalg.eigvalsh(inertia)
        link_inertial[name] = (pos, np.maximum(principal, 1e-12))

    joint_limits = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        limit = joint.find("limit")
        if name and limit is not None:
            joint_limits[name] = (float(limit.get("lower")), float(limit.get("upper")))

    for body_id in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name in link_inertial:
            pos, principal = link_inertial[name]
            model.body_ipos[body_id] = pos
            model.body_inertia[body_id] = principal

    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name in joint_limits and model.jnt_limited[joint_id]:
            model.jnt_range[joint_id] = joint_limits[name]

    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name in USD_JOINT_NAMES:
            dof_id = int(model.jnt_dofadr[joint_id])
            if clear_passive_damping:
                model.dof_damping[dof_id] = 0.0
                model.dof_frictionloss[dof_id] = 0.0
            # Training URDF does not expose passive damping; Isaac's damping is in the
            # implicit actuator PD. Armature is explicit in lens110_dance.py for
            # hips/knees/feet/arms, while torso_yaw has none.
            if name != "torso_yaw_joint":
                model.dof_armature[dof_id] = armature
    print(f"[player] URDF inertial/limits aligned: {os.path.basename(urdf_path)}, armature={armature}, clear_passive_damping={clear_passive_damping}")


def collect_mesh_foot_geoms(model):
    """Geoms attached to ankle pitch/roll links, matching Isaac URDF foot meshes."""
    ids = []
    for gid in range(model.ngeom):
        body_id = int(model.geom_bodyid[gid])
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if "ankle_pitch_link" in body or "ankle_roll_link" in body:
            if int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_MESH):
                ids.append(gid)
    return ids


def collect_box_foot_geoms(model):
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in FOOT_BOX_NAMES]
    return [int(g) for g in ids if int(g) >= 0]


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
    """Contact flags for left/right ankle_roll link, independent of box vs mesh geoms."""
    flags = np.zeros(2, dtype=np.float32)
    force = np.zeros(6, dtype=np.float64)
    roll_body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        for n in FOOT_ROLL_BODY_NAMES
    ]
    for i in range(data.ncon):
        c = data.contact[i]
        body_ids = {int(model.geom_bodyid[int(c.geom1)]), int(model.geom_bodyid[int(c.geom2)])}
        for side, roll_id in enumerate(roll_body_ids):
            if roll_id in body_ids:
                mujoco.mj_contactForce(model, data, i, force)
                if force[0] > 1.0:
                    flags[side] = 1.0
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


def exact_roll_bottoms(model, data, roll_geoms):
    bottoms = []
    for gid in roll_geoms:
        sx, sy, sz = model.geom_size[gid]
        corners = np.array([
            [x, y, z]
            for x in (-sx, sx)
            for y in (-sy, sy)
            for z in (-sz, sz)
        ], dtype=np.float64)
        rmat = data.geom_xmat[gid].reshape(3, 3)
        world_corners = data.geom_xpos[gid] + corners @ rmat.T
        bottoms.append(float(world_corners[:, 2].min()))
    return np.array(bottoms, dtype=np.float64)


def settle_feet_init(model, data, joint_qpos, target_clearance=0.0005):
    """Exact 双脚贴地: 交替校平脚底 + 按碰撞盒真实最低点调 root_z。"""
    roll_geoms = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
        for n in ("left_ankle_roll_collision", "right_ankle_roll_collision")
    ]
    roll_geoms = [int(g) for g in roll_geoms if int(g) >= 0]
    if not roll_geoms:
        return []
    for _ in range(12):
        flatten_feet_init(model, data, joint_qpos)
        bottoms = exact_roll_bottoms(model, data, roll_geoms)
        # Match Isaac's two-foot contact signal: raise/lower the root until the
        # higher foot box bottom reaches target_clearance. The lower foot may
        # start with mm-level penetration, which MuJoCo solves immediately.
        data.qpos[2] += target_clearance - float(np.max(bottoms))
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
    return roll_geoms


def build_obs_159(model, data, policy_qpos, policy_dof, ref_pos_policy, ref_quat, frame, n_frames):
    q = data.qpos[policy_qpos].copy()
    qvel = data.qvel[policy_dof].copy()
    root_q = data.qpos[3:7].copy()
    root_ang_vel_b = data.qvel[3:6].copy()
    parts = [
        root_rot_tan_norm(root_q),          # 6
        root_ang_vel_b,                     # 3 机身系角速度
        q,                                  # 21 关节角
        qvel,                               # 21 关节速度
    ]
    for i in range(4):
        parts.append(root_rot_tan_norm(ref_quat[min(frame + i, n_frames - 1)]))  # 24
    for i in range(4):
        parts.append(ref_pos_policy[min(frame + i, n_frames - 1)])                # 84
    obs = np.concatenate(parts).astype(np.float32)
    assert obs.shape[0] == 159, obs.shape
    return obs.reshape(1, -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--motion", required=True)
    parser.add_argument("--mjcf", required=True)
    parser.add_argument("--deploy", default=None, help="deploy_config.yaml (训练 PD)")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--action_output_order",
        choices=["urdf", "usd"],
        default="urdf",
        help="checkpoint 动作输出顺序; 旧 5100 checkpoint 实际为 usd",
    )
    parser.add_argument("--urdf", default=DEFAULT_URDF, help="用于对齐质心/惯量/关节限位的训练 URDF")
    parser.add_argument("--no_urdf_align", action="store_true", default=False, help="关闭 URDF 对齐")
    parser.add_argument("--armature", type=float, default=0.01, help="按训练 ImplicitActuator 补 armature")
    parser.add_argument("--keep_joint_damping", action="store_true", default=False, help="保留 MJCF passive damping，不推荐")
    parser.add_argument(
        "--foot_collision",
        choices=["isaac", "box", "both"],
        default="box",
        help="脚底碰撞: isaac=使用 URDF 训练里的踝部 mesh collision; box=官方 MJCF 简化脚底盒; both=同时开启",
    )
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
    if args.urdf and not args.no_urdf_align:
        apply_urdf_inertial_and_limits(
            model,
            args.urdf,
            armature=args.armature,
            clear_passive_damping=not args.keep_joint_damping,
        )
    DECIMATION = max(1, int(round(1.0 / (POLICY_HZ * model.opt.timestep))))
    print(f"[sim2sim] 模型 timestep={model.opt.timestep}, decimation={DECIMATION} "
          f"(物理 {1.0/model.opt.timestep:.0f}Hz / 策略 {POLICY_HZ:.0f}Hz)")

    # ---- 官方播放器的环境处理 (脚底盒保留模型原始, 碰撞/摩擦/膝盖限位/质心) ----
    box_foot_ids = collect_box_foot_geoms(model)
    mesh_foot_ids = collect_mesh_foot_geoms(model)
    if args.foot_collision == "box":
        support_foot_ids = box_foot_ids
    elif args.foot_collision == "both":
        support_foot_ids = sorted(set(box_foot_ids) | set(mesh_foot_ids))
    else:
        support_foot_ids = sorted(set(mesh_foot_ids) | set(box_foot_ids))
    if args.foot_collision == "isaac":
        # Isaac uses the ankle URDF mesh collision. Keep boxes disabled during play,
        # but their IDs are only used for initial foot flattening/height settling.
        support_foot_ids = sorted(set(mesh_foot_ids))
    foot_geom_ids = set(support_foot_ids)
    print(f"[sim2sim] 脚底碰撞模式={args.foot_collision}, support geoms={sorted(foot_geom_ids)}")
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
    policy_qpos = np.array([joint_qpos[n] for n in POLICY_JOINT_NAMES])
    policy_dof = np.array([joint_dof[n] for n in POLICY_JOINT_NAMES])
    ctrl_to_policy = np.array([POLICY_JOINT_NAMES.index(
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
    for policy_i, name in enumerate(POLICY_JOINT_NAMES):
        aidx = act_name_to_idx[name]
        model.actuator_gaintype[aidx] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_biastype[aidx] = mujoco.mjtBias.mjBIAS_AFFINE
        model.actuator_gainprm[aidx, :] = 0.0
        model.actuator_gainprm[aidx, 0] = kp[policy_i]
        model.actuator_biasprm[aidx, :] = 0.0
        model.actuator_biasprm[aidx, 1] = -kp[policy_i]
        model.actuator_biasprm[aidx, 2] = -kd[policy_i]
        effort = 80.0 if name in {
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
            "torso_yaw_joint",
        } else 36.0
        model.actuator_forcelimited[aidx] = 1
        model.actuator_forcerange[aidx, 0] = -effort
        model.actuator_forcerange[aidx, 1] = effort

    # 初始状态: 参考第 0 帧 + 速度 + 脚底贴地 (与官方一致)
    data.qpos[:3] = ref_pos_w[0]
    data.qpos[3:7] = ref_quat[0]
    ref_pos_policy = ref_pos[:, USD_TO_POLICY]
    data.qpos[policy_qpos] = ref_pos_policy[0]
    data.qvel[:] = 0.0
    data.qvel[:3] = ref_lin_vel_w[0]
    data.qvel[3:6] = quat_to_rotmat_wxyz(ref_quat[0]).T @ ref_ang_vel_w[0]
    data.qvel[policy_dof] = motion["joint_vel"][0][USD_TO_POLICY]
    mujoco.mj_forward(model, data)
    # 让两只脚的主支撑盒 (ankle_roll_collision) 都贴地: 按两只脚 roll 盒底部的最大值抬升,
    # 允许较低那只轻微压入 (几毫米), 物理接触会自动化解 -> 初始脚触地 obs=[1,1]。
    roll_geoms = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
        for n in ("left_ankle_roll_collision", "right_ankle_roll_collision")
    ]
    joint_qpos = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): int(model.jnt_qposadr[i])
                  for i in range(model.njnt) if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) not in (None, "root")}
    settle_roll_geoms = settle_feet_init(model, data, joint_qpos)
    print(f"[sim2sim] 初始 root_ang_vel_b = {data.qvel[3:6].copy()}")
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
                data.qpos[policy_qpos] = ref_pos_policy[0]
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                settle_feet_init(model, data, joint_qpos)
                continue
            if play_state["paused"]:
                time.sleep(0.02)
                continue

            # 策略推理 (100Hz)
            obs = build_obs_159(model, data, policy_qpos, policy_dof, ref_pos_policy, ref_quat, frame, n_frames)
            out = sess.run(None, {"obs": obs})[0][0].astype(np.float32)
            if action_clip is not None:
                out = np.clip(out, -float(action_clip), float(action_clip))
            if args.action_output_order == "usd":
                # 老训练动作项实际按 Isaac/USD 默认顺序输出, 转回策略的 URDF 顺序。
                out = out[USD_TO_POLICY]
            # 21 维参考中心残差: q_des = 参考 + 0.25 * action (不 clip)
            q_des_policy = ref_pos_policy[frame] + 0.25 * out
            # 目标限位钳制: Isaac 驱动器会把目标硬钳在关节范围内, MuJoCo 关节限位是软约束,
            # 不钳制会导致踝 roll 等关节冲出限位 (脚侧翻 50°)。这里手动对齐。
            for j, name in enumerate(POLICY_JOINT_NAMES):
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid >= 0 and model.jnt_limited[jid]:
                    lo, hi = model.jnt_range[jid]
                    q_des_policy[j] = min(max(q_des_policy[j], lo), hi)
            target_q = q_des_policy[ctrl_to_policy]

            for _ in range(DECIMATION):
                data.ctrl[:] = target_q
                mujoco.mj_step(model, data)
            frame += 1
            viewer.sync()
            if frame % 100 == 0:
                print(f"[sim2sim] 帧 {frame}/{n_frames} ({frame/fps:.1f}s) "
                      f"根高度 {data.qpos[2]:.3f} 关节速度峰值 {np.abs(data.qvel[policy_dof]).max():.2f}")
            if play_state["speed"] > 0:
                target_time = start_time + frame / (POLICY_HZ * play_state["speed"])
                sleep = target_time - time.time()
                if sleep > 0:
                    time.sleep(sleep)
    print(f"[sim2sim] 完成 {frame} 帧")


if __name__ == "__main__":
    main()
