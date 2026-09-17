# SPDX-License-Identifier: BSD-3-Clause
"""Compact MuJoCo sim2sim runner for the Lens110 AMP policy.

The policy outputs joint-position targets.  This script patches the configured
MuJoCo actuators into the same affine position-PD form used by training, then
writes target joint positions to ``data.ctrl``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass

if "--headless" in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

import mujoco
try:
    import cv2
except ImportError:  # Video output is optional for headless replay.
    cv2 = None
try:
    import mujoco_viewer
except ImportError:  # Window rendering is optional for headless replay.
    mujoco_viewer = None
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

try:
    from pynput import keyboard
except ImportError:
    keyboard = None


DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "pr.json")


def repo_root() -> str:
    current = os.path.abspath(os.path.dirname(__file__))
    while True:
        if os.path.isdir(os.path.join(current, "source", "legged_lab")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        current = parent


def resolve_path(path: str, *base_dirs: str) -> str:
    if os.path.isabs(path):
        return path
    candidates = [os.path.abspath(path)]
    candidates.extend(os.path.join(base_dir, path) for base_dir in base_dirs)
    candidates.append(os.path.join(repo_root(), path))
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(os.path.join(repo_root(), path))


@dataclass
class Sim2SimConfig:
    torchscript_policy: str
    onnx_policy: str
    policy_backend: str
    model_path: str
    dt: float
    decimation: int
    num_actions: int
    num_obs: int
    mujoco_joint_names: list[str]
    policy_joint_names: list[str]
    default_pos: np.ndarray
    action_scale: np.ndarray
    init_height: float | None
    fallback_init_height: float
    action_clip: float | None
    kp: np.ndarray
    kd: np.ndarray
    tau_limit: np.ndarray
    armature: np.ndarray
    foot_contact_geoms: list[str]
    static_friction: float | None
    dynamic_friction: float | None


@dataclass
class RobotRuntime:
    policy_to_mujoco: list[int]
    default_pos: np.ndarray
    action_scale: np.ndarray
    base_pos: np.ndarray
    action_clip: float | None
    static_friction: float | None
    dynamic_friction: float | None
    qpos_ids: np.ndarray | None = None
    qvel_ids: np.ndarray | None = None
    actuator_ids: np.ndarray | None = None
    target_min: np.ndarray | None = None
    target_max: np.ndarray | None = None


def as_float_array(config: dict, key: str, expected_len: int) -> np.ndarray:
    values = np.array(config[key], dtype=np.float64)
    if values.shape != (expected_len,):
        raise ValueError(f"Config key '{key}' must contain {expected_len} values, got shape {values.shape}")
    return values


def load_config(path: str) -> Sim2SimConfig:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    config_dir = os.path.dirname(os.path.abspath(path))
    robot = raw["robot"]
    num_actions = int(robot["num_actions"])
    mujoco_joint_names = list(robot["mujoco_joint_names"])
    policy_joint_names = list(robot["policy_joint_names"])
    if len(mujoco_joint_names) != num_actions:
        raise ValueError("mujoco_joint_names length must match robot.num_actions")
    if len(policy_joint_names) != num_actions:
        raise ValueError("policy_joint_names length must match robot.num_actions")

    physics = raw.get("physics", {})
    return Sim2SimConfig(
        torchscript_policy=resolve_path(raw["policy"]["torchscript"], config_dir),
        onnx_policy=resolve_path(raw["policy"]["onnx"], config_dir),
        policy_backend=raw["policy"].get("backend", "torchscript"),
        model_path=resolve_path(raw["model"]["path"], config_dir),
        dt=float(raw["simulation"]["dt"]),
        decimation=int(raw["simulation"]["decimation"]),
        num_actions=num_actions,
        num_obs=9 + 3 * num_actions,
        mujoco_joint_names=mujoco_joint_names,
        policy_joint_names=policy_joint_names,
        default_pos=as_float_array(robot, "default_pos", num_actions),
        action_scale=as_float_array(robot, "action_scale", num_actions),
        init_height=None if robot.get("init_height") is None else float(robot["init_height"]),
        fallback_init_height=float(robot.get("fallback_init_height", 0.68)),
        action_clip=None if robot.get("action_clip") is None else float(robot["action_clip"]),
        kp=as_float_array(robot, "kp", num_actions),
        kd=as_float_array(robot, "kd", num_actions),
        tau_limit=as_float_array(robot, "tau_limit", num_actions),
        armature=as_float_array(robot, "armature", num_actions),
        foot_contact_geoms=list(robot["foot_contact_geoms"]),
        static_friction=None if physics.get("static_friction") is None else float(physics["static_friction"]),
        dynamic_friction=None if physics.get("dynamic_friction") is None else float(physics["dynamic_friction"]),
    )


class TorchScriptPolicy:
    def __init__(self, path: str):
        self.policy = torch.jit.load(path, map_location="cpu")
        self.policy.eval()

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return self.policy(torch.from_numpy(obs.astype(np.float32, copy=False)))[0].detach().cpu().numpy()


class OnnxPolicy:
    def __init__(self, path: str):
        try:
            import onnxruntime as ort

            self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.backend = "onnxruntime"
            self._run = self._run_onnxruntime
        except ImportError:
            import onnx
            from onnx.reference import ReferenceEvaluator

            model = onnx.load(path)
            onnx.checker.check_model(model)
            self.session = ReferenceEvaluator(model)
            self.input_name = model.graph.input[0].name
            self.output_name = model.graph.output[0].name
            self.backend = "onnx-reference"
            self._run = self._run_reference

    def _run_onnxruntime(self, obs: np.ndarray) -> np.ndarray:
        return self.session.run([self.output_name], {self.input_name: obs})[0]

    def _run_reference(self, obs: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: obs})[0]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self._run(obs.astype(np.float32, copy=False))[0]


class CommandState:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0
    camera_follow = True
    reset_requested = False

    @classmethod
    def set_initial(cls, command: tuple[float, float, float]) -> None:
        cls.vx, cls.vy, cls.dyaw = command

    @classmethod
    def update(cls, dvx: float = 0.0, dvy: float = 0.0, ddyaw: float = 0.0) -> None:
        cls.vx = float(np.clip(cls.vx + dvx, -0.8, 2.0))
        cls.vy = float(np.clip(cls.vy + dvy, -0.8, 0.8))
        cls.dyaw = float(np.clip(cls.dyaw + ddyaw, -1.5, 1.5))
        print(f"cmd vx={cls.vx:.2f}, vy={cls.vy:.2f}, dyaw={cls.dyaw:.2f}")

    @classmethod
    def zero(cls) -> None:
        cls.vx = cls.vy = cls.dyaw = 0.0


def on_press(key) -> None:
    try:
        char = key.char.lower() if hasattr(key, "char") and key.char is not None else None
        if char == "8":
            CommandState.update(dvx=0.1)
        elif char == "2":
            CommandState.update(dvx=-0.1)
        elif char == "4":
            CommandState.update(dvy=0.1)
        elif char == "6":
            CommandState.update(dvy=-0.1)
        elif char == "7":
            CommandState.update(ddyaw=0.1)
        elif char == "9":
            CommandState.update(ddyaw=-0.1)
        elif char == "f":
            CommandState.camera_follow = not CommandState.camera_follow
            print(f"camera_follow={CommandState.camera_follow}")
        elif char == "0":
            CommandState.reset_requested = True
    except AttributeError:
        if key == keyboard.Key.up:
            CommandState.update(dvx=0.1)
        elif key == keyboard.Key.down:
            CommandState.update(dvx=-0.1)
        elif key == keyboard.Key.left:
            CommandState.update(ddyaw=0.1)
        elif key == keyboard.Key.right:
            CommandState.update(ddyaw=-0.1)


def start_keyboard_listener():
    if keyboard is None:
        return None
    listener = keyboard.Listener(on_press=on_press, on_release=lambda _: None)
    listener.start()
    return listener


def make_index_mapping(source_names: list[str], target_names: list[str]) -> list[int]:
    target_index = {name: i for i, name in enumerate(target_names)}
    return [target_index[name] for name in source_names]


def name_to_id(model: mujoco.MjModel, obj_type, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise ValueError(f"Missing {obj_type.name}: {name}")
    return obj_id


def build_joint_data(model: mujoco.MjModel, cfg: Sim2SimConfig):
    joint_ids = np.array(
        [name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in cfg.mujoco_joint_names],
        dtype=np.int32,
    )
    qpos_ids = model.jnt_qposadr[joint_ids].astype(np.int32)
    qvel_ids = model.jnt_dofadr[joint_ids].astype(np.int32)

    actuator_ids = np.full(len(joint_ids), -1, dtype=np.int32)
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        matches = np.flatnonzero(joint_ids == joint_id)
        if len(matches):
            actuator_ids[matches[0]] = actuator_id
    if np.any(actuator_ids < 0):
        missing = [name for name, actuator_id in zip(cfg.mujoco_joint_names, actuator_ids) if actuator_id < 0]
        raise ValueError(f"Missing actuators for joints: {missing}")

    target_min = np.full(len(joint_ids), -np.inf, dtype=np.float64)
    target_max = np.full(len(joint_ids), np.inf, dtype=np.float64)
    for i, joint_id in enumerate(joint_ids):
        if model.jnt_limited[joint_id]:
            target_min[i], target_max[i] = model.jnt_range[joint_id]

    return qpos_ids, qvel_ids, actuator_ids, target_min, target_max


def geom_min_z(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float:
    if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
        sx, sy, sz = model.geom_size[geom_id]
        corners = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)], dtype=np.float64)
        points = data.geom_xpos[geom_id] + corners @ data.geom_xmat[geom_id].reshape(3, 3).T
        return float(points[:, 2].min())
    return float(data.geom_xpos[geom_id, 2] - model.geom_rbound[geom_id])


def grounded_base_height(
    model: mujoco.MjModel, data: mujoco.MjData, robot: RobotRuntime, cfg: Sim2SimConfig, clearance: float = 0.002
) -> float:
    saved_qpos = data.qpos.copy()
    saved_qvel = data.qvel.copy()
    saved_ctrl = data.ctrl.copy()

    data.qpos[:3] = robot.base_pos
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[robot.qpos_ids] = robot.default_pos
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    min_z = []
    for name in cfg.foot_contact_geoms:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id >= 0:
            min_z.append(geom_min_z(model, data, geom_id))

    data.qpos[:], data.qvel[:], data.ctrl[:] = saved_qpos, saved_qvel, saved_ctrl
    mujoco.mj_forward(model, data)
    return float(robot.base_pos[2] if not min_z else robot.base_pos[2] - min(min_z) + clearance)


def patch_model(model: mujoco.MjModel, robot: RobotRuntime, cfg: Sim2SimConfig) -> None:
    foot_geom_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in cfg.foot_contact_geoms
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0
    }
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if geom_name == "floor" or geom_id in foot_geom_ids:
            model.geom_contype[geom_id] = 1
            model.geom_conaffinity[geom_id] = 1
        else:
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0
    print(f"[INFO] active contacts: floor + {len(foot_geom_ids)} foot geoms")

    if robot.static_friction is not None:
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_id >= 0:
            original = model.geom_friction[floor_id].copy()
            slide = robot.dynamic_friction if robot.dynamic_friction is not None else robot.static_friction
            model.geom_friction[floor_id] = [slide, original[1], original[2]]
            print(
                "[INFO] floor friction "
                f"{np.array2string(original, precision=3)} -> "
                f"{np.array2string(model.geom_friction[floor_id], precision=3)}"
            )

    model.dof_damping[:] = 0.0
    model.dof_armature[robot.qvel_ids] = cfg.armature
    for mujoco_id, actuator_id in enumerate(robot.actuator_ids):
        kp = float(cfg.kp[mujoco_id])
        kd = float(cfg.kd[mujoco_id])
        effort = float(cfg.tau_limit[mujoco_id])
        model.actuator_gaintype[actuator_id] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_biastype[actuator_id] = mujoco.mjtBias.mjBIAS_AFFINE
        model.actuator_gainprm[actuator_id, :] = 0.0
        model.actuator_gainprm[actuator_id, 0] = kp
        model.actuator_biasprm[actuator_id, :] = 0.0
        model.actuator_biasprm[actuator_id, 1] = -kp
        model.actuator_biasprm[actuator_id, 2] = -kd
        model.actuator_ctrllimited[actuator_id] = 0
        model.actuator_forcelimited[actuator_id] = 1
        model.actuator_forcerange[actuator_id] = [-effort, effort]
    print("[INFO] position-PD actuators patched with training gains and torque limits")


def hold_default_stand(data: mujoco.MjData, robot: RobotRuntime) -> None:
    data.qpos[:3] = robot.base_pos
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[robot.qpos_ids] = robot.default_pos
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.ctrl[robot.actuator_ids] = robot.default_pos


def get_base_obs(data: mujoco.MjData):
    qpos = data.qpos.astype(np.float64)
    qvel = data.qvel.astype(np.float64)
    rot = R.from_quat(qpos[3:7][[1, 2, 3, 0]])
    lin_vel_b = rot.apply(qvel[:3], inverse=True).astype(np.float64)
    ang_vel_b = data.sensor("angular-velocity").data.astype(np.float64)
    gravity_b = rot.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.float64)
    return qpos, qvel, lin_vel_b, ang_vel_b, gravity_b


def make_observation(q, dq, omega, gravity, action, robot: RobotRuntime, cfg: Sim2SimConfig):
    joint_pos_rel = q - robot.default_pos
    q_obs = np.zeros(cfg.num_actions, dtype=np.float64)
    dq_obs = np.zeros(cfg.num_actions, dtype=np.float64)
    for policy_id, mujoco_id in enumerate(robot.policy_to_mujoco):
        q_obs[policy_id] = joint_pos_rel[mujoco_id]
        dq_obs[policy_id] = dq[mujoco_id]

    obs = np.zeros((1, cfg.num_obs), dtype=np.float32)
    obs[0, 0:3] = omega
    obs[0, 3:6] = gravity
    obs[0, 6:9] = [CommandState.vx, CommandState.vy, CommandState.dyaw]
    obs[0, 9 : 9 + cfg.num_actions] = q_obs
    obs[0, 9 + cfg.num_actions : 9 + 2 * cfg.num_actions] = dq_obs
    obs[0, 9 + 2 * cfg.num_actions : 9 + 3 * cfg.num_actions] = action
    return obs


def action_to_target(action: np.ndarray, robot: RobotRuntime) -> np.ndarray:
    target_pos = robot.default_pos.copy()
    target_delta_policy = action * robot.action_scale
    for policy_id, mujoco_id in enumerate(robot.policy_to_mujoco):
        target_pos[mujoco_id] += target_delta_policy[policy_id]
    np.clip(target_pos, robot.target_min, robot.target_max, out=target_pos)
    return target_pos


def init_render(model: mujoco.MjModel, data: mujoco.MjData, args: argparse.Namespace, cfg: Sim2SimConfig):
    if args.headless:
        if cv2 is None:
            raise ImportError("Video rendering requires OpenCV; install opencv-python in the active local environment.")
        model.vis.global_.offwidth = args.viewer_width
        model.vis.global_.offheight = args.viewer_height
        renderer = mujoco.Renderer(model, width=args.viewer_width, height=args.viewer_height)
        camera = mujoco.MjvCamera()
        camera.distance = 4.0
        camera.azimuth = 45.0
        camera.elevation = -20.0
        camera.lookat = [0.0, 0.0, 1.0]
        writer = cv2.VideoWriter(
            args.output,
            cv2.VideoWriter_fourcc(*"mp4v"),
            1.0 / cfg.dt / cfg.decimation,
            (args.viewer_width, args.viewer_height),
        )
        return renderer, camera, writer, None

    if mujoco_viewer is None:
        raise ImportError("Window rendering requires mujoco-python-viewer in the active local environment.")
    viewer = mujoco_viewer.MujocoViewer(model, data, mode="window", width=args.viewer_width, height=args.viewer_height)
    viewer.cam.distance = 4.0
    viewer.cam.azimuth = 45.0
    viewer.cam.elevation = -20.0
    viewer.cam.lookat = [0.0, 0.0, 1.0]
    return None, None, None, viewer


def render(data: mujoco.MjData, renderer, camera, writer, viewer, headless: bool) -> bool:
    if CommandState.camera_follow:
        base_pos = [float(x) for x in data.qpos[:3]]
        if headless:
            camera.lookat = base_pos
        else:
            viewer.cam.lookat = base_pos
    if headless:
        renderer.update_scene(data, camera=camera)
        writer.write(renderer.render())
        return True
    try:
        viewer.render()
    except Exception as exc:
        if "GLFW window does not exist" in str(exc):
            return False
        raise
    return True


def summarize_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, int]:
    counts: dict[str, int] = {}
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or f"geom_{contact.geom1}"
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or f"geom_{contact.geom2}"
        pair = " / ".join(sorted((name1, name2)))
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def make_robot(args: argparse.Namespace, cfg: Sim2SimConfig) -> RobotRuntime:
    return RobotRuntime(
        policy_to_mujoco=make_index_mapping(cfg.policy_joint_names, cfg.mujoco_joint_names),
        default_pos=cfg.default_pos.copy(),
        action_scale=cfg.action_scale.copy(),
        base_pos=np.array([0.0, 0.0, args.init_height], dtype=np.float64),
        action_clip=cfg.action_clip,
        static_friction=cfg.static_friction,
        dynamic_friction=cfg.dynamic_friction,
    )


def run(policy, args: argparse.Namespace, cfg: Sim2SimConfig) -> None:
    print("Keyboard: 8/2 vx, 4/6 vy, 7/9 yaw, arrows vx/yaw, 0 reset, F camera follow")
    listener = start_keyboard_listener()

    model = mujoco.MjModel.from_xml_path(args.model_path)
    model.opt.timestep = cfg.dt
    data = mujoco.MjData(model)

    robot = make_robot(args, cfg)
    robot.qpos_ids, robot.qvel_ids, robot.actuator_ids, robot.target_min, robot.target_max = build_joint_data(model, cfg)
    patch_model(model, robot, cfg)

    if not args.no_auto_base_height:
        robot.base_pos[2] = grounded_base_height(model, data, robot, cfg)
        print(f"[INFO] auto base height: {robot.base_pos[2]:.3f}")

    hold_default_stand(data, robot)
    mujoco.mj_forward(model, data)
    initial_qpos = data.qpos.copy()
    initial_qvel = data.qvel.copy()

    renderer, camera, writer, viewer = init_render(model, data, args, cfg)
    action = np.zeros(cfg.num_actions, dtype=np.float64)
    target_pos = robot.default_pos.copy()
    min_root_z = float("inf")
    max_abs_yaw_rate = 0.0
    max_abs_vy = 0.0
    start_time = time.time()

    total_steps = int(args.sim_duration / cfg.dt)
    for step in tqdm(range(total_steps), desc="Simulating"):
        if CommandState.reset_requested:
            data.qpos[:] = initial_qpos
            data.qvel[:] = initial_qvel
            data.ctrl[:] = 0.0
            data.ctrl[robot.actuator_ids] = robot.default_pos
            action[:] = 0.0
            target_pos[:] = robot.default_pos
            CommandState.zero()
            CommandState.reset_requested = False
            mujoco.mj_forward(model, data)

        qpos, qvel, lin_vel_b, omega_b, gravity_b = get_base_obs(data)
        q = qpos[robot.qpos_ids]
        dq = qvel[robot.qvel_ids]
        min_root_z = min(min_root_z, float(qpos[2]))
        max_abs_yaw_rate = max(max_abs_yaw_rate, abs(float(omega_b[2])))
        max_abs_vy = max(max_abs_vy, abs(float(lin_vel_b[1])))

        if step % cfg.decimation == 0:
            obs = make_observation(q, dq, omega_b, gravity_b, action, robot, cfg)
            action[:] = policy(obs)
            if robot.action_clip is not None:
                np.clip(action, -robot.action_clip, robot.action_clip, out=action)
            target_pos = action_to_target(action, robot)

            policy_step = step // cfg.decimation
            if policy_step % max(1, args.print_every) == 0:
                print(
                    f"cmd=({CommandState.vx:.2f}, {CommandState.vy:.2f}, {CommandState.dyaw:.2f}) "
                    f"vel=({lin_vel_b[0]:.2f}, {lin_vel_b[1]:.2f}, {omega_b[2]:.2f})"
                )
            if not render(data, renderer, camera, writer, viewer, args.headless):
                break

        data.ctrl[robot.actuator_ids] = target_pos
        mujoco.mj_step(model, data)

        elapsed = time.time() - start_time
        target_time = (step + 1) * cfg.dt
        if elapsed < target_time:
            time.sleep(target_time - elapsed)

    print(
        "[SUMMARY] "
        f"min_root_z={min_root_z:.3f}, "
        f"max_abs_yaw_rate={max_abs_yaw_rate:.3f}, "
        f"max_abs_vy={max_abs_vy:.3f}, "
        f"final_contacts={summarize_contacts(model, data)}"
    )

    if args.headless:
        writer.release()
        print(f"[INFO] saved video: {args.output}")
    else:
        viewer.close()
    if listener is not None:
        listener.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compact Lens110 MuJoCo sim2sim runner.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Sim2sim JSON config path.")
    parser.add_argument("--load_model", default=None, help="Override TorchScript policy path from config.")
    parser.add_argument("--load_onnx", default=None, help="Override ONNX policy path from config.")
    parser.add_argument("--policy_backend", choices=("torchscript", "onnx"), default=None)
    parser.add_argument("--model_path", default=None, help="Override MuJoCo XML model path from config.")
    parser.add_argument("--cmd_vel", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="Initial command: vx vy yaw_rate.")
    parser.add_argument("--sim_duration", type=float, default=10000.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", default="simulation_simple.mp4")
    parser.add_argument("--viewer_width", type=int, default=1280)
    parser.add_argument("--viewer_height", type=int, default=720)
    parser.add_argument("--init_height", type=float, default=None)
    parser.add_argument("--no_auto_base_height", action="store_true")
    parser.add_argument("--print_every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    args.load_model = resolve_path(args.load_model) if args.load_model else cfg.torchscript_policy
    args.load_onnx = resolve_path(args.load_onnx) if args.load_onnx else cfg.onnx_policy
    args.policy_backend = args.policy_backend or cfg.policy_backend
    args.model_path = resolve_path(args.model_path) if args.model_path else cfg.model_path
    args.init_height = args.init_height if args.init_height is not None else (cfg.init_height or cfg.fallback_init_height)

    CommandState.set_initial(tuple(args.cmd_vel))
    if args.policy_backend == "onnx":
        policy = OnnxPolicy(args.load_onnx)
        print(f"[INFO] loaded ONNX policy with {policy.backend}: {args.load_onnx}")
    else:
        policy = TorchScriptPolicy(args.load_model)
        print(f"[INFO] loaded TorchScript policy: {args.load_model}")
    run(policy, args, cfg)


if __name__ == "__main__":
    main()
