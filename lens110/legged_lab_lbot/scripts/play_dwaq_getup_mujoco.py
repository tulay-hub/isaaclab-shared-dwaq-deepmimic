"""LENS110 DWAQ walking + automatic get-up MuJoCo player.

The walking policy remains the 76-dim DWAQ Torch policy.  When the robot is
detected as fallen, the existing 288-dim AMP get-up ONNX policy takes over.
Both policies run at 100 Hz while MuJoCo advances at 500 Hz.
"""

from __future__ import annotations

import argparse
import collections
import os
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import torch
import yaml

from mujoco_terrain import TERRAIN_NAMES, load_model_with_terrain

from play_dwaq_mujoco import (
    ACTION_SCALE,
    ARMATURE_USD,
    DEFAULT_QPOS_USD,
    DECIMATION,
    EFFORT_USD,
    GAIT_PERIOD,
    HISTORY_LEN,
    KD_USD,
    KP_USD,
    KEY_DOWN,
    KEY_KP_2,
    KEY_KP_4,
    KEY_KP_6,
    KEY_KP_8,
    KEY_LEFT,
    KEY_RIGHT,
    KEY_UP,
    SIM_DT,
    USD_JOINT_NAMES,
    compute_gait_phase,
    flatten_history_term_major,
    load_policy,
)


def _repository_root() -> Path:
    """Find the reorganized repository root without relying on old depth."""
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        if (parent / "projects").is_dir() and (parent / "frameworks").is_dir():
            return parent
    # Keep standalone copies of the old framework usable as a fallback.
    return script_path.parents[4]


REPOSITORY_ROOT = _repository_root()
GETUP_ROOT = (
    REPOSITORY_ROOT
    / "projects"
    / "04_fall_to_stand"
    / "exports"
    / "versions"
    / "Lens110_GetUp_Sim2Real_v2_20260908"
)
DEFAULT_GETUP_ONNX = GETUP_ROOT / "policy" / "Lens110-AMP-GetUp_model_72500.onnx"
DEFAULT_GETUP_CFG = GETUP_ROOT / "config" / "deploy_config.yaml"


def quat_to_rotmat_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * w), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--xml", required=True)
    parser.add_argument("--getup-onnx", default=str(DEFAULT_GETUP_ONNX))
    parser.add_argument("--getup-cfg", default=str(DEFAULT_GETUP_CFG))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--terrain", choices=TERRAIN_NAMES, default="flat")
    parser.add_argument("--terrain-seed", type=int, default=7)
    parser.add_argument("--stair-scale", type=float, default=1.0)
    args = parser.parse_args()

    actor, vae = load_policy(args.checkpoint, torch.device(args.device))
    getup_cfg = yaml.safe_load(Path(args.getup_cfg).read_text(encoding="utf-8"))
    getup_joint_names = list(getup_cfg["joint_names_mjcf"])
    getup_default = np.asarray(getup_cfg["default_joint_pos_rad"], dtype=np.float64)
    getup_scale = np.asarray(getup_cfg["action_scale"], dtype=np.float64)
    getup_kp = np.asarray(getup_cfg["joint_stiffness_kp"], dtype=np.float64)
    getup_kd = np.asarray(getup_cfg["joint_damping_kd"], dtype=np.float64)
    getup_effort = np.asarray(getup_cfg["effort_limit_nm"], dtype=np.float64)
    getup_dt = float(getup_cfg["frequencies"]["timestep"])
    getup_decimation = int(getup_cfg["frequencies"]["decimation"])
    if abs(getup_dt - SIM_DT) > 1e-9 or getup_decimation != DECIMATION:
        raise RuntimeError(
            f"get-up frequency mismatch: cfg={1/getup_dt:.0f}Hz/{getup_decimation}, "
            f"player={1/SIM_DT:.0f}Hz/{DECIMATION}"
        )

    getup_session = ort.InferenceSession(args.getup_onnx, providers=["CPUExecutionProvider"])
    getup_input_name = getup_session.get_inputs()[0].name

    model = load_model_with_terrain(args.xml, args.terrain, args.terrain_seed, args.stair_scale)
    data = mujoco.MjData(model)
    model.opt.timestep = SIM_DT
    print(f"[sim] terrain={args.terrain} seed={args.terrain_seed} stair_scale={args.stair_scale}")
    print(
        f"[sim] physics={1/SIM_DT:.0f}Hz, policy={1/(SIM_DT * DECIMATION):.0f}Hz, "
        f"getup={os.path.basename(args.getup_onnx)}"
    )

    joint_qpos: dict[str, int] = {}
    joint_dof: dict[str, int] = {}
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name and name != "root":
            joint_qpos[name] = model.jnt_qposadr[joint_id]
            joint_dof[name] = model.jnt_dofadr[joint_id]

    missing = sorted((set(USD_JOINT_NAMES) | set(getup_joint_names)) - set(joint_qpos))
    if missing:
        raise RuntimeError(f"XML missing joints: {missing}")
    usd_qpos = np.asarray([joint_qpos[n] for n in USD_JOINT_NAMES])
    usd_dof = np.asarray([joint_dof[n] for n in USD_JOINT_NAMES])
    getup_qpos = np.asarray([joint_qpos[n] for n in getup_joint_names])
    getup_dof = np.asarray([joint_dof[n] for n in getup_joint_names])

    act_name_to_idx = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(model.nu)
    }
    if any(n not in act_name_to_idx for n in USD_JOINT_NAMES):
        raise RuntimeError("DWAQ player requires one position actuator per named joint")

    # Keep every body collision active so a fallen robot can support itself on
    # its hands/torso.  This matches the Isaac asset's enabled self-collisions.
    original_contype = model.geom_contype.copy()
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    for geom_id in range(model.ngeom):
        if geom_id == floor_id:
            model.geom_contype[geom_id] = 1
            model.geom_conaffinity[geom_id] = 15
        else:
            model.geom_contype[geom_id] = original_contype[geom_id]
            model.geom_conaffinity[geom_id] = 1 if original_contype[geom_id] else 0
    if floor_id >= 0:
        model.geom_friction[floor_id] = [0.7, 0.005, 0.0001]

    dwaq_kp_by_name = dict(zip(USD_JOINT_NAMES, KP_USD))
    dwaq_kd_by_name = dict(zip(USD_JOINT_NAMES, KD_USD))
    dwaq_effort_by_name = dict(zip(USD_JOINT_NAMES, EFFORT_USD))
    getup_kp_by_name = dict(zip(getup_joint_names, getup_kp))
    getup_kd_by_name = dict(zip(getup_joint_names, getup_kd))
    getup_effort_by_name = dict(zip(getup_joint_names, getup_effort))

    def set_dynamics(damping: float, frictionloss: float, armature: np.ndarray) -> None:
        model.dof_damping[usd_dof] = damping
        model.dof_frictionloss[usd_dof] = frictionloss
        model.dof_armature[usd_dof] = armature

    def enable_dwaq_position_pd() -> None:
        for name in USD_JOINT_NAMES:
            actuator_id = act_name_to_idx[name]
            kp = float(dwaq_kp_by_name[name])
            effort = float(dwaq_effort_by_name[name])
            model.actuator_gaintype[actuator_id] = mujoco.mjtGain.mjGAIN_FIXED
            model.actuator_biastype[actuator_id] = mujoco.mjtBias.mjBIAS_AFFINE
            model.actuator_gainprm[actuator_id, :] = 0.0
            model.actuator_gainprm[actuator_id, 0] = kp
            model.actuator_biasprm[actuator_id, :] = 0.0
            model.actuator_biasprm[actuator_id, 1] = -kp
            model.actuator_biasprm[actuator_id, 2] = -float(dwaq_kd_by_name[name])
            model.actuator_forcelimited[actuator_id] = 1
            model.actuator_forcerange[actuator_id] = [-effort, effort]

    def disable_position_pd() -> None:
        for actuator_id in act_name_to_idx.values():
            model.actuator_gainprm[actuator_id, :] = 0.0
            model.actuator_biasprm[actuator_id, :] = 0.0
            model.actuator_forcelimited[actuator_id] = 0

    def root_frames() -> tuple[np.ndarray, np.ndarray, float]:
        rotation = quat_to_rotmat_wxyz(data.qpos[3:7])
        gravity_b = rotation.T @ np.array([0.0, 0.0, -1.0])
        ang_vel_b = rotation.T @ data.qvel[3:6]
        tilt_deg = float(np.degrees(np.arccos(np.clip(-gravity_b[2], -1.0, 1.0))))
        return gravity_b, ang_vel_b, tilt_deg

    def getup_obs_block(prev_action: np.ndarray) -> np.ndarray:
        gravity_b, ang_vel_b, _ = root_frames()
        return np.concatenate(
            [ang_vel_b, gravity_b, np.zeros(3), data.qpos[getup_qpos] - getup_default,
             data.qvel[getup_dof], prev_action]
        ).astype(np.float32)

    state = {
        "mode": "walk",
        "reset_requested": False,
        "push_requested": False,
        "force_getup": False,
        "running": True,
        "fall_count": 0,
        "upright_count": 0,
        "walk_history_ready": False,
        "walk_time": 0.0,
    }
    walk_history = np.zeros((HISTORY_LEN, 76), dtype=np.float32)
    walk_last_action = np.zeros(21, dtype=np.float32)
    getup_history: collections.deque[np.ndarray] = collections.deque(maxlen=4)
    getup_last_action = np.zeros(21, dtype=np.float64)
    command = np.zeros(3, dtype=np.float32)

    def switch_to_getup(reason: str) -> None:
        if state["mode"] == "getup":
            return
        state["mode"] = "getup"
        state["fall_count"] = 0
        state["upright_count"] = 0
        command[:] = 0.0
        getup_last_action.fill(0.0)
        getup_history.clear()
        block = getup_obs_block(getup_last_action)
        for _ in range(getup_history.maxlen):
            getup_history.append(block.copy())
        set_dynamics(0.01, 0.01, np.full(21, 0.01, dtype=np.float64))
        disable_position_pd()
        data.ctrl[:] = 0.0
        data.qfrc_applied[:] = 0.0
        print(f"[state] WALK -> GETUP ({reason})")

    def switch_to_walk() -> None:
        state["mode"] = "walk"
        state["upright_count"] = 0
        state["fall_count"] = 0
        state["walk_history_ready"] = False
        state["walk_time"] = 0.0
        walk_last_action.fill(0.0)
        set_dynamics(0.0, 0.0, ARMATURE_USD)
        enable_dwaq_position_pd()
        data.qfrc_applied[:] = 0.0
        print("[state] GETUP -> WALK")

    # Initial standing pose, matching the DWAQ environment.
    enable_dwaq_position_pd()
    set_dynamics(0.0, 0.0, ARMATURE_USD)
    data.qpos[:3] = [0.0, 0.0, 0.68]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[usd_qpos] = DEFAULT_QPOS_USD
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    initial_qpos = data.qpos.copy()
    initial_qvel = data.qvel.copy()

    def reset_simulation() -> None:
        mujoco.mj_resetData(model, data)
        data.qpos[:] = initial_qpos
        data.qvel[:] = initial_qvel
        mujoco.mj_forward(model, data)
        state["mode"] = "walk"
        state["fall_count"] = 0
        state["upright_count"] = 0
        state["walk_history_ready"] = False
        state["walk_time"] = 0.0
        walk_history.fill(0.0)
        walk_last_action.fill(0.0)
        getup_history.clear()
        getup_last_action.fill(0.0)
        command[:] = 0.0
        set_dynamics(0.0, 0.0, ARMATURE_USD)
        enable_dwaq_position_pd()

    def push_once() -> None:
        data.qvel[0:3] += np.random.uniform([-0.6, -0.4, -0.2], [0.6, 0.4, 0.2])
        data.qvel[3:6] += np.random.uniform(-0.4, 0.4, 3)

    def on_key(keycode: int) -> None:
        if keycode in (ord("8"), KEY_KP_8, KEY_UP):
            command[0] = np.clip(command[0] + 0.05, -0.6, 1.0)
        elif keycode in (ord("2"), KEY_KP_2, KEY_DOWN):
            command[0] = np.clip(command[0] - 0.05, -0.6, 1.0)
        elif keycode in (ord("4"), KEY_KP_4, KEY_LEFT):
            command[1] = np.clip(command[1] + 0.05, -0.5, 0.5)
        elif keycode in (ord("6"), KEY_KP_6, KEY_RIGHT):
            command[1] = np.clip(command[1] - 0.05, -0.5, 0.5)
        elif keycode == ord("q"):
            command[2] = np.clip(command[2] + 0.05, -1.57, 1.57)
        elif keycode == ord("e"):
            command[2] = np.clip(command[2] - 0.05, -1.57, 1.57)
        elif keycode == ord("0"):
            command[:] = 0.0
        elif keycode in (ord("r"), ord("R")):
            state["reset_requested"] = True
        elif keycode in (ord("f"), ord("F")):
            state["push_requested"] = True
        elif keycode in (ord("g"), ord("G")):
            state["force_getup"] = True
        elif keycode == 27:
            state["running"] = False
        print(
            f"mode={state['mode']} cmd: vx={command[0]:.2f} "
            f"vy={command[1]:.2f} yaw={command[2]:.2f}"
        )

    policy_dt = SIM_DT * DECIMATION
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        print("Keys: keypad 8/2/4/6 or arrows | f push | g getup | r reset | q/e yaw | Esc quit")
        while viewer.is_running() and state["running"]:
            loop_start = time.time()
            if state["reset_requested"]:
                state["reset_requested"] = False
                reset_simulation()
                print("[reset] restored standing DWAQ state")
                viewer.sync()
                continue
            if state["push_requested"]:
                state["push_requested"] = False
                push_once()
            if state["force_getup"]:
                state["force_getup"] = False
                switch_to_getup("manual")

            gravity_b, ang_vel_b, tilt_deg = root_frames()
            if state["mode"] == "walk":
                fallen = data.qpos[2] < 0.44 or tilt_deg > 70.0
                state["fall_count"] = state["fall_count"] + 1 if fallen else 0
                if state["fall_count"] >= 3:
                    switch_to_getup("height/tilt")

            if state["mode"] == "walk":
                q = data.qpos[usd_qpos].copy()
                qvel = data.qvel[usd_dof].copy()
                gait = compute_gait_phase(state["walk_time"])
                obs = np.concatenate(
                    [
                        ang_vel_b,
                        gravity_b,
                        command,
                        q - DEFAULT_QPOS_USD,
                        qvel,
                        walk_last_action,
                        gait,
                    ]
                ).astype(np.float32)
                if not state["walk_history_ready"]:
                    walk_history[:] = obs
                    state["walk_history_ready"] = True
                else:
                    walk_history[:-1] = walk_history[1:]
                    walk_history[-1] = obs
                with torch.no_grad():
                    hist = torch.from_numpy(flatten_history_term_major(walk_history)[None]).to(torch.device(args.device))
                    obs_t = torch.from_numpy(obs[None]).to(torch.device(args.device))
                    code = vae.encode_mean(hist)
                    action = actor(torch.cat([code, obs_t], dim=-1))[0].cpu().numpy()
                target = DEFAULT_QPOS_USD + ACTION_SCALE * action
                for i, name in enumerate(USD_JOINT_NAMES):
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    if model.jnt_limited[jid]:
                        target[i] = np.clip(target[i], *model.jnt_range[jid])
                walk_last_action[:] = action
            else:
                getup_obs = np.concatenate(list(getup_history)).astype(np.float32)
                getup_action = getup_session.run(None, {getup_input_name: getup_obs[None]})[0][0]
                getup_last_action[:] = getup_action
                target = getup_default + getup_scale * getup_action

            for _ in range(DECIMATION):
                if state["mode"] == "walk":
                    data.qfrc_applied[:] = 0.0
                    for i, name in enumerate(USD_JOINT_NAMES):
                        data.ctrl[act_name_to_idx[name]] = target[i]
                else:
                    data.ctrl[:] = 0.0
                    tau = getup_kp * (target - data.qpos[getup_qpos]) - getup_kd * data.qvel[getup_dof]
                    tau = np.clip(tau, -getup_effort, getup_effort)
                    data.qfrc_applied[:] = 0.0
                    data.qfrc_applied[getup_dof] = tau
                mujoco.mj_step(model, data)

            if state["mode"] == "getup":
                getup_history.append(getup_obs_block(getup_last_action))
                _, _, tilt_deg = root_frames()
                standing = data.qpos[2] > 0.58 and tilt_deg < 25.0
                state["upright_count"] = state["upright_count"] + 1 if standing else 0
                if state["upright_count"] >= 25:
                    switch_to_walk()
            else:
                state["walk_time"] += policy_dt

            viewer.sync()
            time.sleep(max(0.0, policy_dt - (time.time() - loop_start)))


if __name__ == "__main__":
    main()
