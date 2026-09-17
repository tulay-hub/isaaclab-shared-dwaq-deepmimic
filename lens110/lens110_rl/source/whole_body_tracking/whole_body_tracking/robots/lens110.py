"""Lens110 21-DOF 机器人资产配置 (对齐本机训练资产)。

URDF 路径指向本地 lens110 工程 (含 meshes/), PD 用强 PD:
  髋p/r/y + 膝 120/4, 踝 55/2, 腰 45/1.5, 肩/肘 35/1.2
初始姿态 = 舞蹈参考第 0 帧 (与训练 npz 对齐)。
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

LENS110_URDF_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "..", "..", "assets", "lens110_21dof.urdf",
    )
)
LENS110_ACTION_SCALE = 0.25

LENS110_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=False,
        merge_fixed_joints=False,
        asset_path=LENS110_URDF_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=8,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # 舞蹈参考第 0 帧 (lens110_amp_100hz_flat.npz)
        pos=(0.024599, -0.269941, 0.659900),
        rot=(0.699323, -0.022836, -0.026560, -0.713947),
        joint_pos={
            "left_hip_pitch_joint": 0.205537,
            "right_hip_pitch_joint": 0.213849,
            "torso_yaw_joint": -0.020909,
            "left_hip_roll_joint": 0.158203,
            "right_hip_roll_joint": -0.191155,
            "left_shoulder_pitch_joint": 0.658771,
            "right_shoulder_pitch_joint": 0.629726,
            "left_hip_yaw_joint": 0.293051,
            "right_hip_yaw_joint": -0.304459,
            "left_shoulder_roll_joint": 0.941942,
            "right_shoulder_roll_joint": -1.125977,
            "left_knee_joint": -0.053091,
            "right_knee_joint": -0.068512,
            "left_shoulder_yaw_joint": -0.479678,
            "right_shoulder_yaw_joint": 0.511430,
            "left_ankle_pitch_joint": 0.058394,
            "right_ankle_pitch_joint": 0.076278,
            "left_elbow_joint": -1.131860,
            "right_elbow_joint": -1.114993,
            "left_ankle_roll_joint": -0.177711,
            "right_ankle_roll_joint": 0.199545,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*knee_joint",
                "torso_yaw_joint",
            ],
            effort_limit_sim={
                ".*_hip_pitch_joint": 80.0,
                ".*_hip_roll_joint": 80.0,
                ".*_hip_yaw_joint": 36.0,
                ".*knee_joint": 80.0,
                "torso_yaw_joint": 36.0,
            },
            stiffness={
                ".*_hip_pitch_joint": 120.0,
                ".*_hip_roll_joint": 120.0,
                ".*_hip_yaw_joint": 120.0,
                ".*knee_joint": 120.0,
                "torso_yaw_joint": 45.0,
            },
            damping={
                ".*_hip_pitch_joint": 4.0,
                ".*_hip_roll_joint": 4.0,
                ".*_hip_yaw_joint": 4.0,
                ".*knee_joint": 4.0,
                "torso_yaw_joint": 1.5,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*ankle_pitch_joint", ".*ankle_roll_joint"],
            effort_limit_sim=36.0,
            stiffness=55.0,
            damping=2.0,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*shoulder_pitch_joint",
                ".*shoulder_roll_joint",
                ".*shoulder_yaw_joint",
                ".*elbow_joint",
            ],
            effort_limit_sim=36.0,
            stiffness=35.0,
            damping=1.2,
        ),
    },
)
