from pathlib import Path

import isaaclab.sim as sim_utils
from isaacsim.core.utils.extensions import enable_extension
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim import converters
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.utils import clone
from pxr import PhysxSchema, Usd, UsdPhysics


ASSET_DIR = Path(__file__).resolve().parent
enable_extension("isaacsim.asset.importer.mjcf")


def _remove_worldbody_articulation_roots(usd_path: str):
    usd_root = Path(usd_path)
    usd_files = [usd_root, *sorted((usd_root.parent / "configuration").glob("*.usd"))]
    for usd_file in usd_files:
        stage = Usd.Stage.Open(str(usd_file))
        if stage is None:
            continue
        changed = False
        for prim in stage.Traverse():
            if prim.GetName() != "worldBody":
                continue
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                changed = True
            if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
                changed = True
        if changed:
            stage.GetRootLayer().Save()


@clone
def spawn_lens110_mjcf(
    prim_path: str,
    cfg: sim_utils.MjcfFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    mjcf_loader = converters.MjcfConverter(cfg)
    _remove_worldbody_articulation_roots(mjcf_loader.usd_path)
    return _spawn_from_usd_file(prim_path, mjcf_loader.usd_path, cfg, translation, orientation)

LENS110_MJCF_ACTIVE_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_upper_joint",
    "left_ankle_lower_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_upper_joint",
    "right_ankle_lower_joint",
    "torso_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]

LENS110_MJCF_DEFAULT_JOINT_POS = {
    "left_hip_pitch_joint": -0.14,
    "left_hip_roll_joint": 0.01,
    "left_hip_yaw_joint": -0.1,
    "left_knee_joint": 0.30,
    "left_ankle_pitch_joint": -0.15,
    "left_ankle_roll_joint": 0.0,
    "left_ankle_upper_joint": 0.161341,
    "left_ankle_lower_joint": 0.162273,
    "right_hip_pitch_joint": -0.14,
    "right_hip_roll_joint": -0.01,
    "right_hip_yaw_joint": 0.1,
    "right_knee_joint": 0.30,
    "right_ankle_pitch_joint": -0.15,
    "right_ankle_roll_joint": 0.0,
    "right_ankle_upper_joint": 0.161341,
    "right_ankle_lower_joint": 0.162273,
    "torso_yaw_joint": 0.0,
    "left_shoulder_pitch_joint": 0.4,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": -0.8,
    "right_shoulder_pitch_joint": 0.4,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.8,
}


LENS110_MJCF_CFG = ArticulationCfg(
    spawn=sim_utils.MjcfFileCfg(
        func=spawn_lens110_mjcf,
        fix_base=False,
        asset_path=str(ASSET_DIR / "mjcf" / "lens110_train.xml"),
        import_sites=True,
        self_collision=True,
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
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.68),
        joint_pos=LENS110_MJCF_DEFAULT_JOINT_POS,
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
        "ankles": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_upper_joint", ".*_ankle_lower_joint"],
            effort_limit_sim=36.0,
            stiffness=5.0,
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
            damping=1.0,
            armature={
                ".*shoulder.*": 0.01,
                ".*elbow_joint": 0.01,
            },
        ),
    },
)
