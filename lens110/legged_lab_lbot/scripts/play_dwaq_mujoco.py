"""LENS110 DWAQ blind-walking MuJoCo player.

Follows the same actuator/joint-order pattern as play_lens110_official_161.py.
"""

import argparse
import math
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn

from mujoco_terrain import TERRAIN_NAMES, load_model_with_terrain

# Isaac Lab (USD) joint order -- the order the policy was trained with
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

# Default joint positions in USD order (from lens110_dance.py init_state)
DEFAULT_QPOS_USD = np.array([
    -0.14, -0.14, 0.0,        # left/right hip_pitch, torso
    -0.01, -0.01,             # left/right hip_roll
    0.4, 0.4,                 # left/right shoulder_pitch
    -0.10, 0.10,              # left/right hip_yaw
    0.2, -0.2,                # left/right shoulder_roll
    0.36, 0.36,               # left/right knee
    0.0, 0.0,                 # left/right shoulder_yaw
    -0.2575, -0.2575,         # left/right ankle_pitch
    -0.8, -0.8,               # left/right elbow
    0.0, 0.0,                 # left/right ankle_roll
], dtype=np.float64)

# Training PD gains in USD order (from lens110_dance.py actuators)
KP_USD = np.array([
    40, 40, 100,             # hip_pitch L/R, torso
    40, 40,                   # hip_roll L/R
    20, 20,                   # shoulder_pitch L/R
    40, 40,                   # hip_yaw L/R
    20, 20,                   # shoulder_roll L/R
    40, 40,                   # knee L/R
    20, 20,                   # shoulder_yaw L/R
    55, 55,                   # ankle_pitch L/R
    20, 20,                   # elbow L/R
    55, 55,                   # ankle_roll L/R
], dtype=np.float64)
KD_USD = np.full(21, 5.0, dtype=np.float64)

# Isaac Lab LENS110_CFG armature: all actuated joints except torso_yaw_joint.
ARMATURE_USD = np.array([
    0.01, 0.01, 0.0,
    0.01, 0.01,
    0.01, 0.01,
    0.01, 0.01,
    0.01, 0.01,
    0.01, 0.01,
    0.01, 0.01,
    0.01, 0.01,
    0.01, 0.01,
    0.01, 0.01,
], dtype=np.float64)

EFFORT_USD = np.array([
    80, 80, 80,
    80, 80,
    36, 36,
    80, 80,
    36, 36,
    80, 80,
    36, 36,
    36, 36,
    36, 36,
    36, 36,
], dtype=np.float64)

ACTION_SCALE = 0.25
GAIT_PERIOD = 0.6
HISTORY_LEN = 5
POLICY_HZ = 100.0
SIM_DT = 0.002  # match LENS110 training: sim.dt=0.002, decimation=5 -> 100Hz policy
DECIMATION = 5
HEADING_CONTROL_STIFFNESS = 1.0
MAX_HEADING_YAW_COMMAND = 0.6

# GLFW key codes used by MuJoCo viewer.
KEY_UP = 265
KEY_DOWN = 264
KEY_LEFT = 263
KEY_RIGHT = 262
KEY_KP_2 = 322
KEY_KP_4 = 324
KEY_KP_6 = 326
KEY_KP_8 = 328

FOOT_GEOM_NAMES = [
    "left_ankle_roll_collision", "right_ankle_roll_collision",
    "left_ankle_pitch_collision", "right_ankle_pitch_collision",
]


class ActorMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden):
        super().__init__()
        layers = []
        curr = in_dim
        for h in hidden:
            layers += [nn.Linear(curr, h), nn.ELU()]
            curr = h
        layers.append(nn.Linear(curr, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ContextVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, velocity_dim, code_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, latent_dim), nn.ELU())
        self.mean_vel = nn.Linear(latent_dim, velocity_dim)
        self.logvar_vel = nn.Linear(latent_dim, velocity_dim)
        self.mean_latent = nn.Linear(latent_dim, code_dim - velocity_dim)
        self.logvar_latent = nn.Linear(latent_dim, code_dim - velocity_dim)

    def encode_mean(self, x):
        h = self.encoder(x)
        return torch.cat([self.mean_vel(h), self.mean_latent(h)], dim=-1)


def load_policy(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    policy_obs_dim = 76
    actor = ActorMLP(19 + policy_obs_dim, 21, [512, 256, 128])
    actor.load_state_dict({k.replace("actor.", "net."): v for k, v in sd.items() if k.startswith("actor.")})
    vae = ContextVAE(policy_obs_dim * HISTORY_LEN, 128, 64, 3, 19)
    vae.load_state_dict({k.replace("context_vae.", ""): v for k, v in sd.items() if k.startswith("context_vae.")}, strict=False)
    actor.to(device).eval()
    vae.to(device).eval()
    print(f"[OK] checkpoint iter={ckpt.get('iter','?')}")
    return actor, vae


def compute_gait_phase(t):
    phase = (t % GAIT_PERIOD) / GAIT_PERIOD
    s = [math.sin(2 * math.pi * ((phase + off) % 1.0)) for off in (0.0, 0.5)]
    c = [math.cos(2 * math.pi * ((phase + off) % 1.0)) for off in (0.0, 0.5)]
    return np.array(s + c, dtype=np.float64)


def flatten_history_term_major(history: np.ndarray) -> np.ndarray:
    """Flatten policy history like Isaac Lab's per-term history buffers.

    Isaac Lab returns each term's five-frame history contiguously, then
    concatenates terms in observation declaration order.  This is different
    from flattening five complete 76-D frames.
    """
    term_slices = (
        slice(0, 3),
        slice(3, 6),
        slice(6, 9),
        slice(9, 30),
        slice(30, 51),
        slice(51, 72),
        slice(72, 76),
    )
    return np.concatenate([history[:, term_slice].reshape(-1) for term_slice in term_slices])


def demo_command(t: float, duration: float, legacy: bool = False) -> np.ndarray:
    """Deterministic command sequence for observing the policy response."""
    duration = max(float(duration), 20.0)
    if legacy:
        ramp_speed = 0.20
        walk_speed = 0.40
    else:
        ramp_speed = 0.25
        walk_speed = 0.30
    if t < 2.0:
        vx = 0.0
    elif t < 5.0:
        vx = ramp_speed
    elif t < duration - 8.0:
        vx = walk_speed
    elif t < duration - 3.0:
        vx = ramp_speed
    else:
        vx = 0.0
    return np.array([vx, 0.0, 0.0], dtype=np.float32)


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--xml", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--terrain", choices=TERRAIN_NAMES, default="flat")
    parser.add_argument("--terrain-seed", type=int, default=7)
    parser.add_argument("--stair-scale", type=float, default=1.0)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run an automatic long walking command demo.",
    )
    parser.add_argument("--demo-duration", type=float, default=60.0)
    parser.add_argument(
        "--legacy-command",
        action="store_true",
        help="Use the pre-heading-lock command semantics saved by older checkpoints.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    actor, vae = load_policy(args.checkpoint, device)

    model = load_model_with_terrain(args.xml, args.terrain, args.terrain_seed, args.stair_scale)
    data = mujoco.MjData(model)
    model.opt.timestep = SIM_DT
    print(f"[sim] terrain={args.terrain} seed={args.terrain_seed} stair_scale={args.stair_scale}")
    print(f"[sim] timestep={model.opt.timestep}, decimation={DECIMATION} (physics {1/SIM_DT:.0f}Hz, policy {1/(SIM_DT*DECIMATION):.0f}Hz)")

    # -- Collision: only feet and floor collide (matches official player) --
    foot_geom_ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in FOOT_GEOM_NAMES}
    foot_geom_ids.discard(-1)
    for gid in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if gname in {"floor", "ground", "plane"} or gid in foot_geom_ids:
            model.geom_contype[gid] = 1
            model.geom_conaffinity[gid] = 15
        else:
            model.geom_contype[gid] = 1
            model.geom_conaffinity[gid] = 0
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id >= 0:
        model.geom_friction[floor_id] = [0.7, 0.005, 0.0001]

    # Index mapping: name -> MuJoCo internal address
    act_name_to_idx = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(model.nu)}
    joint_qpos, joint_dof = {}, {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name and name != "root":
            joint_qpos[name] = model.jnt_qposadr[i]
            joint_dof[name] = model.jnt_dofadr[i]

    usd_qpos = np.array([joint_qpos[n] for n in USD_JOINT_NAMES])
    usd_dof = np.array([joint_dof[n] for n in USD_JOINT_NAMES])

    # Configure position actuators (same as official player)
    for usd_i, name in enumerate(USD_JOINT_NAMES):
        aidx = act_name_to_idx[name]
        model.actuator_gaintype[aidx] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_biastype[aidx] = mujoco.mjtBias.mjBIAS_AFFINE
        model.actuator_gainprm[aidx, :] = 0.0
        model.actuator_gainprm[aidx, 0] = KP_USD[usd_i]
        model.actuator_biasprm[aidx, :] = 0.0
        model.actuator_biasprm[aidx, 1] = -KP_USD[usd_i]
        model.actuator_biasprm[aidx, 2] = -KD_USD[usd_i]
        model.actuator_forcelimited[aidx] = 1
        model.actuator_forcerange[aidx, 0] = -EFFORT_USD[usd_i]
        model.actuator_forcerange[aidx, 1] = EFFORT_USD[usd_i]

    # Match the Isaac Lab URDF asset: no physical joint damping/friction is
    # specified there; damping comes from the actuator PD term.  The MuJoCo
    # XML contains extra joint damping values, so clear them explicitly.
    model.dof_damping[usd_dof] = 0.0
    model.dof_frictionloss[usd_dof] = 0.0
    model.dof_armature[usd_dof] = ARMATURE_USD
    print("[sim] PD matched training: physical damping=0, armature=0.01 (torso=0)")

    # Initial state
    data.qpos[:3] = [0.0, 0.0, 0.68]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[usd_qpos] = DEFAULT_QPOS_USD
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    policy_obs_dim = 76
    obs_history = np.zeros((HISTORY_LEN, policy_obs_dim), dtype=np.float32)
    last_action = np.zeros(21, dtype=np.float32)
    cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    sim_time = 0.0
    heading_target = 0.0
    straight_command_active = False
    running = [True]
    reset_requested = [False]
    history_initialized = [False]

    initial_qpos = data.qpos.copy()
    initial_qvel = data.qvel.copy()

    def reset_simulation():
        nonlocal sim_time, heading_target, straight_command_active
        mujoco.mj_resetData(model, data)
        data.qpos[:] = initial_qpos
        data.qvel[:] = initial_qvel
        mujoco.mj_forward(model, data)
        obs_history.fill(0.0)
        last_action.fill(0.0)
        cmd[:] = 0.0
        sim_time = 0.0
        heading_target = 0.0
        straight_command_active = False
        history_initialized[0] = False

    def on_key(keycode):
        if keycode in (ord("8"), KEY_KP_8, KEY_UP):
            cmd[0] = np.clip(cmd[0] + 0.05, -0.6, 1.0)
        elif keycode in (ord("2"), KEY_KP_2, KEY_DOWN):
            cmd[0] = np.clip(cmd[0] - 0.05, -0.6, 1.0)
        elif keycode in (ord("4"), KEY_KP_4, KEY_LEFT):
            cmd[1] = np.clip(cmd[1] + 0.05, -0.5, 0.5)
        elif keycode in (ord("6"), KEY_KP_6, KEY_RIGHT):
            cmd[1] = np.clip(cmd[1] - 0.05, -0.5, 0.5)
        elif keycode == ord("q"):
            cmd[2] = np.clip(cmd[2] + 0.05, -1.57, 1.57)
        elif keycode == ord("e"):
            cmd[2] = np.clip(cmd[2] - 0.05, -1.57, 1.57)
        elif keycode == ord("0"):
            cmd[:] = 0.0
        elif keycode in (ord("r"), ord("R")):
            reset_requested[0] = True
            return
        elif keycode == 27:
            running[0] = False
        print(f"cmd: vx={cmd[0]:.2f} vy={cmd[1]:.2f} yaw={cmd[2]:.2f}")

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        print("Keys: keypad 8/2/4/6 or arrows | q/e yaw | 0 stop | r reset | Esc quit")
        if args.demo:
            speed_text = "0.20m/s 3s -> 0.40m/s long walk -> 0.20m/s 5s" if args.legacy_command else "0.25m/s 3s -> 0.30m/s long walk -> 0.25m/s 5s"
            print(f"[demo] duration={args.demo_duration:.1f}s: 0m/s 2s -> {speed_text} -> 0m/s")
        if args.legacy_command:
            print("[sim] legacy command semantics: heading lock disabled")
        else:
            print(
                f"[sim] straight heading lock: kp={HEADING_CONTROL_STIFFNESS:.1f}, "
                f"yaw_limit={MAX_HEADING_YAW_COMMAND:.1f}rad/s"
            )
        last_demo_cmd = cmd.copy()
        while viewer.is_running() and running[0]:
            if reset_requested[0]:
                reset_requested[0] = False
                reset_simulation()
                print("[reset] restored initial pose, command, action and VAE history")
                viewer.sync()
                continue

            if args.demo:
                cmd[:] = demo_command(sim_time, args.demo_duration, args.legacy_command)
                if not np.array_equal(cmd, last_demo_cmd):
                    print(f"[demo] t={sim_time:.2f}s -> vx={cmd[0]:.2f} vy={cmd[1]:.2f} yaw={cmd[2]:.2f}")
                    last_demo_cmd = cmd.copy()

            # -- build observation (USD order) --
            q = data.qpos[usd_qpos].copy()
            qvel = data.qvel[usd_dof].copy()
            root_q = data.qpos[3:7].copy()
            w, x, y, z = root_q
            R = np.array([
                [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
            proj_g = R.T @ np.array([0.0, 0.0, -1.0])
            ang_vel_world = data.qvel[3:6].copy()
            ang_vel_b = R.T @ ang_vel_world
            gait = compute_gait_phase(sim_time)

            # Match the Isaac command generator: straight forward/backward
            # segments hold the heading sampled at the segment start and
            # expose the corrective yaw command to the policy observation.
            current_heading = math.atan2(R[1, 0], R[0, 0])
            straight_command = (
                abs(float(cmd[0])) > 0.15
                and abs(float(cmd[1])) < 0.1
                and abs(float(cmd[2])) < 0.1
            )
            if args.legacy_command:
                policy_cmd = cmd.copy()
            else:
                if straight_command and not straight_command_active:
                    heading_target = current_heading
                if straight_command:
                    policy_cmd = cmd.copy()
                    policy_cmd[2] = np.clip(
                        HEADING_CONTROL_STIFFNESS * wrap_to_pi(heading_target - current_heading),
                        -MAX_HEADING_YAW_COMMAND,
                        MAX_HEADING_YAW_COMMAND,
                    )
                else:
                    policy_cmd = cmd.copy()
            straight_command_active = straight_command

            obs = np.concatenate([
                ang_vel_b, proj_g, policy_cmd,
                q - DEFAULT_QPOS_USD, qvel,
                last_action, gait
            ]).astype(np.float32)
            assert obs.shape[0] == 76, obs.shape

            if not history_initialized[0]:
                # Isaac Lab CircularBuffer duplicates the first observation
                # across all history slots after reset.
                obs_history[:] = obs
                history_initialized[0] = True
            else:
                obs_history[:-1] = obs_history[1:]
                obs_history[-1] = obs

            with torch.no_grad():
                hist = torch.from_numpy(flatten_history_term_major(obs_history)).unsqueeze(0).to(device)
                code = vae.encode_mean(hist)
                obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
                inp = torch.cat([code, obs_t], dim=-1)
                action = actor(inp).squeeze(0).cpu().numpy()

            joint_targets_usd = DEFAULT_QPOS_USD + ACTION_SCALE * action
            # The Isaac action term has no action clip.  Clamp only the final
            # position target to the MuJoCo joint limits for stable playback.
            for usd_i, name in enumerate(USD_JOINT_NAMES):
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid >= 0 and model.jnt_limited[jid]:
                    lo, hi = model.jnt_range[jid]
                    joint_targets_usd[usd_i] = np.clip(joint_targets_usd[usd_i], lo, hi)
            last_action = action.astype(np.float32)

            # -- step simulation --
            for _ in range(DECIMATION):
                for usd_i, name in enumerate(USD_JOINT_NAMES):
                    aidx = act_name_to_idx[name]
                    data.ctrl[aidx] = joint_targets_usd[usd_i]
                mujoco.mj_step(model, data)
                sim_time += model.opt.timestep
            viewer.sync()


if __name__ == "__main__":
    main()
