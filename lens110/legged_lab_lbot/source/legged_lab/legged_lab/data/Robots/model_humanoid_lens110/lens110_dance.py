import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent


LENS110_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=str(ASSET_DIR / "lens110_21dof.urdf"),
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
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.68),
        joint_pos={
            "left_hip_pitch_joint": -0.14,
            "left_hip_roll_joint": -0.01,
            "left_hip_yaw_joint": -0.1,
            "left_knee_joint": 0.36,
            "left_ankle_pitch_joint": -0.2575,
            "left_ankle_roll_joint": -0.0,
            "right_hip_pitch_joint": -0.14,
            "right_hip_roll_joint": -0.01,
            "right_hip_yaw_joint": 0.1,
            "right_knee_joint": 0.36,
            "right_ankle_pitch_joint": -0.2575,
            "right_ankle_roll_joint": -0.0,
            "torso_yaw_joint": 0.0,
            "left_shoulder_pitch_joint": 0.4,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": -0.8,
            "right_shoulder_pitch_joint": 0.4,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": -0.8,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*knee_joint",
                "torso_yaw_joint",
            ],
            effort_limit_sim=80.0,
            stiffness={
                ".*_hip_yaw_joint": 40.0,
                ".*_hip_roll_joint": 40.0,
                ".*_hip_pitch_joint": 40.0,
                ".*knee_joint": 40.0,
                "torso_yaw_joint": 100.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*knee_joint": 5.0,
                "torso_yaw_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*knee_joint": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*ankle_pitch_joint", ".*ankle_roll_joint"],
            effort_limit_sim=36.0,
            # 脚踝 PD 太低 (10/10) 时, 参考第 0 帧零动作会前倾倒地。
            # 训练定版: 55/5 (阻尼足、不积累晃动; 稳态前倾由策略用踝动作修正)。
            stiffness=55.0,
            damping=5.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*shoulder_pitch_joint",
                ".*shoulder_roll_joint",
                ".*shoulder_yaw_joint",
                ".*elbow_joint",
            ],
            effort_limit_sim=36.0,
            stiffness=20.0,
            # 手臂阻尼 20/1 -> 20/5: 减少跟随快速甩臂时的抖动
            damping=5.0,
            armature={
                ".*shoulder.*": 0.01,
                ".*elbow_joint": 0.01,
            },
        ),
    },
)
