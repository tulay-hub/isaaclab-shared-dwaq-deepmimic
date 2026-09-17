# SPDX-License-Identifier: BSD-3-Clause
"""Lean Lens110 upper/lower MuJoCo sim2sim runner.

By default this runs the physical upper/lower MJCF path from ``ul.json`` while
preserving the configured upper/lower policy semantics:
pitch/roll observations converted to upper/lower, and the training
upper/lower-to-pitch/roll action replay before commanding the physical ankle.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time

if "--headless" in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

import mujoco
import numpy as np
from tqdm import tqdm

import ul_full as full


DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "ul.json")


def quiet_call(func, *args, verbose: bool = False, **kwargs):
    if verbose:
        return func(*args, **kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def infer_params_dir(policy_path: str) -> str | None:
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(policy_path)))
    params_dir = os.path.join(run_dir, "params")
    return params_dir if os.path.isdir(params_dir) else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lean Lens110 upper/lower sim2sim.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--load_model", default=None)
    parser.add_argument("--params_dir", default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--ankle_model_path", default=None)
    parser.add_argument("--policy_ankle_model_path", default=None)
    parser.add_argument("--cmd_vel", type=float, nargs=3, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--decimation", type=int, default=None)
    parser.add_argument("--sim_duration", type=float, default=None)
    parser.add_argument("--print_every", type=int, default=None)
    parser.add_argument("--action_scale", type=float, default=1.0)
    parser.add_argument("--ankle_action_scale", type=float, default=1.0)
    parser.add_argument(
        "--arm_action_scale",
        type=float,
        default=1.0,
        help="Scale policy arm actions before applying them. Use 0 to keep arms at the training default pose.",
    )
    parser.add_argument("--ankle_command_rel_clip", type=float, default=None)
    parser.add_argument("--ankle_command_rate_limit", type=float, default=None)
    parser.add_argument("--ankle_motor_kp_scale", type=float, default=None)
    parser.add_argument("--ankle_motor_kd_scale", type=float, default=None)
    parser.add_argument("--ankle_motor_tau_scale", type=float, default=None)
    parser.add_argument(
        "--ankle_control",
        choices=("pr_joint_torque", "model_pr_task_torque", "exact_pr_task_torque", "pr_task_torque", "direct_pd"),
        default=None,
        help=(
            "Ankle motor control. pr_joint_torque computes pitch/roll PD torque like training and applies it "
            "directly on the passive pitch/roll DoFs. model_pr_task_torque maps that torque to upper/lower motors "
            "with the fitted ankle-model Jacobian. exact_pr_task_torque uses the MuJoCo tendon Jacobian. "
            "direct_pd uses upper/lower position PD."
        ),
    )
    parser.add_argument("--stand_warmup", type=float, default=0.0)
    parser.add_argument("--command_ramp", type=float, default=1.0)
    parser.add_argument(
        "--startup_freeze",
        type=float,
        default=1.0,
        help="Keep the floating base fixed at startup while contacts and the ankle linkage settle.",
    )
    parser.add_argument(
        "--use_stand_default",
        action="store_true",
        default=True,
        help="Initialize and warm up with robot.stand_default_pos from the JSON instead of the training default pose.",
    )
    parser.add_argument(
        "--use_training_default",
        dest="use_stand_default",
        action="store_false",
        help="Initialize and warm up with the training default pose instead of robot.stand_default_pos.",
    )
    parser.add_argument(
        "--policy_default_from_stand",
        action="store_true",
        default=False,
        help="Use robot.stand_default_pos as the policy/action default for the physical upper/lower plant.",
    )
    parser.add_argument(
        "--policy_default_from_training",
        dest="policy_default_from_stand",
        action="store_false",
        help="Keep the training default pose as the policy/action default.",
    )
    parser.add_argument(
        "--training_action",
        dest="upper_lower_action_mode",
        action="store_const",
        const="through_pitch_roll",
        default=None,
        help="Replay the training action term: upper/lower policy target -> pitch/roll target -> upper/lower motor target.",
    )
    parser.add_argument(
        "--direct_ul",
        dest="upper_lower_action_mode",
        action="store_const",
        const="direct",
        help="Send policy upper/lower targets directly to upper/lower motors.",
    )
    parser.add_argument("--hold_default", action="store_true")
    parser.add_argument("--freeze_base", action="store_true")
    parser.add_argument("--debug_first_obs", action="store_true")
    parser.add_argument("--debug_first_action", action="store_true")
    parser.add_argument("--init_height", type=float, default=None)
    parser.add_argument("--no_auto_base_height", action="store_true")
    parser.add_argument("--base_ang_vel_from_qvel", dest="base_ang_vel_from_qvel", action="store_true", default=False)
    parser.add_argument("--base_ang_vel_from_sensor", dest="base_ang_vel_from_qvel", action="store_false")
    parser.add_argument("--sensor_orientation", dest="sensor_orientation", action="store_true", default=True)
    parser.add_argument("--root_orientation_from_qpos", dest="sensor_orientation", action="store_false")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", default="lens110_ul.mp4")
    parser.add_argument("--viewer_width", type=int, default=1280)
    parser.add_argument("--viewer_height", type=int, default=720)
    parser.add_argument("--show_menus", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_runtime(args: argparse.Namespace):
    cfg = full.load_config(args.config)
    cfg.policy_io = "upper_lower"
    if args.upper_lower_action_mode is not None:
        cfg.upper_lower_action_mode = args.upper_lower_action_mode
    full.validate_policy_joint_order(cfg)

    args.load_model = full.resolve_path(args.load_model) if args.load_model else cfg.torchscript_policy
    if not args.load_model and not args.hold_default:
        raise FileNotFoundError(
            "No local upper/lower TorchScript policy is configured. "
            "Provide --load_model with a local policy.pt file."
        )
    if args.params_dir:
        args.params_dir = full.resolve_path(args.params_dir, os.path.dirname(os.path.abspath(args.config)))
    else:
        args.params_dir = cfg.source_params_dir or (infer_params_dir(args.load_model) if args.load_model else None)
    args.model_path = full.resolve_path(args.model_path) if args.model_path else cfg.model_path
    args.ankle_model_path = full.resolve_path(args.ankle_model_path) if args.ankle_model_path else cfg.ankle_model_path
    policy_ankle_model_overridden = args.policy_ankle_model_path is not None
    args.policy_ankle_model_path = (
        full.resolve_path(args.policy_ankle_model_path)
        if args.policy_ankle_model_path
        else cfg.policy_ankle_model_path
    )

    quiet_call(full.apply_run_params, cfg, args.params_dir, verbose=args.verbose)
    if not policy_ankle_model_overridden:
        args.policy_ankle_model_path = cfg.policy_ankle_model_path
    if args.ankle_control is None:
        args.ankle_control = "pr_joint_torque" if cfg.ankle_pd_space == "pitch_roll" else "direct_pd"
    if args.ankle_command_rel_clip is not None:
        cfg.ankle_command_rel_clip = None if args.ankle_command_rel_clip < 0.0 else args.ankle_command_rel_clip
    if args.ankle_command_rate_limit is not None:
        cfg.ankle_command_rate_limit = (
            None if args.ankle_command_rate_limit < 0.0 else args.ankle_command_rate_limit
        )
    if args.ankle_motor_kp_scale is not None:
        cfg.ankle_motor_kp_scale = args.ankle_motor_kp_scale
    if args.ankle_motor_kd_scale is not None:
        cfg.ankle_motor_kd_scale = args.ankle_motor_kd_scale
    if args.ankle_motor_tau_scale is not None:
        cfg.ankle_motor_tau_scale = args.ankle_motor_tau_scale
    if args.dt is not None:
        cfg.dt = args.dt
    if args.decimation is not None:
        cfg.decimation = args.decimation
    if args.policy_default_from_stand and cfg.stand_default_pos is not None:
        cfg.default_pos = cfg.stand_default_pos.copy()

    args.init_height = args.init_height if args.init_height is not None else (cfg.init_height or cfg.fallback_init_height)
    args.cmd_vel = tuple(args.cmd_vel) if args.cmd_vel is not None else cfg.default_cmd_vel
    args.sim_duration = args.sim_duration if args.sim_duration is not None else cfg.sim_duration
    args.print_every = args.print_every if args.print_every is not None else cfg.print_every
    return cfg


def build_sim(args: argparse.Namespace, cfg):
    model = mujoco.MjModel.from_xml_path(args.model_path)
    model.opt.timestep = cfg.dt
    data = mujoco.MjData(model)
    quiet_call(full.align_default_pos_to_model_ranges, model, cfg, verbose=args.verbose)

    robot = full.make_robot(args, cfg)
    (
        robot.qpos_ids,
        robot.qvel_ids,
        robot.control_qpos_ids,
        robot.control_qvel_ids,
        robot.actuator_ids,
        robot.observe_target_min,
        robot.observe_target_max,
        robot.control_target_min,
        robot.control_target_max,
    ) = full.build_joint_data(model, cfg)
    robot.passive_default_qpos_ids, robot.passive_default_values = full.build_passive_joint_defaults(model, cfg)

    control_index = {name: i for i, name in enumerate(cfg.control_joint_names)}
    has_upper_lower = all(name in control_index for name in full.UPPER_LOWER_JOINT_NAMES)
    if not has_upper_lower:
        raise RuntimeError("This lean runner expects lens110.xml with upper/lower ankle motors.")

    policy_ankle_model = full.AnkleCommandModel(args.policy_ankle_model_path)
    if args.ankle_control == "exact_pr_task_torque":
        ankle_mapper = full.ExactAnkleMapper(model, robot, cfg)
    else:
        ankle_mapper = full.PolynomialAnkleMapper(args.ankle_model_path, robot, cfg)
    robot.control_default_pos = ankle_mapper.observe_to_control_target(robot.default_pos)
    if cfg.upper_lower_action_mode == "direct":
        full.apply_upper_lower_command(
            robot.control_default_pos,
            full.policy_default_upper_lower(robot, cfg, policy_ankle_model),
            cfg,
        )
    robot.stand_pos = (
        cfg.stand_default_pos.copy()
        if args.use_stand_default and cfg.stand_default_pos is not None
        else robot.default_pos.copy()
    )
    robot.stand_control_pos = ankle_mapper.observe_to_control_target(robot.stand_pos)

    if cfg.use_training_passive_ankle_limits:
        quiet_call(full.patch_training_passive_ankle_limits, model, verbose=args.verbose)
    torque_control_ids = set(range(len(cfg.control_joint_names)))
    quiet_call(full.patch_model, model, robot, cfg, torque_control_ids=torque_control_ids, verbose=args.verbose)

    if not args.no_auto_base_height:
        robot.base_pos[2] = full.grounded_base_height(model, data, robot, cfg)

    full.hold_default_stand(data, robot, cfg)
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    return model, data, robot, ankle_mapper, policy_ankle_model


def has_physical_upper_lower_plant(cfg) -> bool:
    return all(name in cfg.control_joint_names for name in full.UPPER_LOWER_JOINT_NAMES)


def run_training_plant(args: argparse.Namespace, cfg) -> None:
    args.debug_actions = args.debug_first_action
    args.debug_obs = args.debug_first_obs
    args.debug_ankle = False
    args.ankle_control = "position_model"
    full.CommandState.set_initial(tuple(args.cmd_vel))
    policy = None if args.hold_default else full.TorchScriptPolicy(args.load_model)
    full.run(policy, args, cfg)


def pd_torque(control_q: np.ndarray, control_dq: np.ndarray, target: np.ndarray, cfg) -> np.ndarray:
    kp = cfg.kp.astype(np.float64, copy=True)
    kd = cfg.kd.astype(np.float64, copy=True)
    tau_limit = cfg.tau_limit.astype(np.float64, copy=True)
    for control_id, joint_name in enumerate(cfg.control_joint_names):
        if joint_name in full.UPPER_LOWER_JOINT_NAMES:
            kp[control_id] *= cfg.ankle_motor_kp_scale
            kd[control_id] *= cfg.ankle_motor_kd_scale
            tau_limit[control_id] *= cfg.ankle_motor_tau_scale
    tau = kp * (target - control_q) - kd * control_dq
    return np.clip(tau, -tau_limit, tau_limit)


def arm_policy_action_ids(cfg) -> np.ndarray:
    return np.array(
        [
            i
            for i, name in enumerate(cfg.policy_joint_names)
            if "shoulder" in name or "elbow" in name
        ],
        dtype=np.int32,
    )


def model_ankle_jacobian_pr2ul(ankle_model, ankle_pr: np.ndarray, side: str, eps: float = 1e-5) -> np.ndarray:
    """Numerically evaluate d(upper, lower) / d(pitch, roll) from the fitted training model."""
    ankle_pr = np.asarray(ankle_pr, dtype=np.float64).reshape(4)
    if side == "left":
        cols = (0, 1)
        rows = (0, 1)
    else:
        cols = (2, 3)
        rows = (2, 3)

    jac = np.zeros((2, 2), dtype=np.float64)
    for col_id, pr_id in enumerate(cols):
        plus = ankle_pr.copy()
        minus = ankle_pr.copy()
        plus[pr_id] += eps
        minus[pr_id] -= eps
        ul_plus = ankle_model.pos_pr2ul(plus.reshape(1, 4))[0]
        ul_minus = ankle_model.pos_pr2ul(minus.reshape(1, 4))[0]
        jac[:, col_id] = (ul_plus[list(rows)] - ul_minus[list(rows)]) / (2.0 * eps)
    return jac


def model_ankle_torque_control(
    side: str,
    q: np.ndarray,
    dq: np.ndarray,
    target_pos: np.ndarray,
    robot,
    cfg,
    ankle_model,
) -> np.ndarray:
    observe_index = {name: i for i, name in enumerate(cfg.mujoco_joint_names)}
    control_index = {name: i for i, name in enumerate(cfg.control_joint_names)}
    if side == "left":
        pitch_name, roll_name = "left_ankle_pitch_joint", "left_ankle_roll_joint"
        upper_name, lower_name = "left_ankle_upper_joint", "left_ankle_lower_joint"
    else:
        pitch_name, roll_name = "right_ankle_pitch_joint", "right_ankle_roll_joint"
        upper_name, lower_name = "right_ankle_upper_joint", "right_ankle_lower_joint"

    task_ids = np.array([observe_index[pitch_name], observe_index[roll_name]], dtype=np.int32)
    motor_ids = np.array([control_index[upper_name], control_index[lower_name]], dtype=np.int32)
    ankle_pr = q[
        [
            observe_index["left_ankle_pitch_joint"],
            observe_index["left_ankle_roll_joint"],
            observe_index["right_ankle_pitch_joint"],
            observe_index["right_ankle_roll_joint"],
        ]
    ]
    tau_pr = cfg.kp[task_ids] * (target_pos[task_ids] - q[task_ids]) - cfg.kd[task_ids] * dq[task_ids]
    tau_pr = np.clip(tau_pr, -cfg.tau_limit[task_ids], cfg.tau_limit[task_ids])
    jacobian = model_ankle_jacobian_pr2ul(ankle_model, ankle_pr, side)
    try:
        tau_ul = np.linalg.solve(jacobian.T, tau_pr)
    except np.linalg.LinAlgError:
        tau_ul = np.linalg.pinv(jacobian.T) @ tau_pr
    tau_limit = cfg.tau_limit[motor_ids] * cfg.ankle_motor_tau_scale
    return np.clip(tau_ul, -tau_limit, tau_limit)


def pitch_roll_task_torque(side: str, q: np.ndarray, dq: np.ndarray, target_pos: np.ndarray, cfg) -> tuple[np.ndarray, np.ndarray]:
    observe_index = {name: i for i, name in enumerate(cfg.mujoco_joint_names)}
    if side == "left":
        pitch_name, roll_name = "left_ankle_pitch_joint", "left_ankle_roll_joint"
    else:
        pitch_name, roll_name = "right_ankle_pitch_joint", "right_ankle_roll_joint"
    task_ids = np.array([observe_index[pitch_name], observe_index[roll_name]], dtype=np.int32)
    tau_pr = cfg.kp[task_ids] * (target_pos[task_ids] - q[task_ids]) - cfg.kd[task_ids] * dq[task_ids]
    return task_ids, np.clip(tau_pr, -cfg.tau_limit[task_ids], cfg.tau_limit[task_ids])


def reset_state(data, initial_qpos, initial_qvel, robot, cfg, action, real_ankle_default):
    if not full.CommandState.reset_requested:
        return
    data.qpos[:] = initial_qpos
    data.qvel[:] = initial_qvel
    data.ctrl[:] = 0.0
    action[:] = 0.0
    full.CommandState.zero()
    full.CommandState.reset_requested = False
    return real_ankle_default.copy(), robot.stand_pos.copy(), robot.stand_control_pos.copy()


def run(args: argparse.Namespace, cfg) -> None:
    policy = full.TorchScriptPolicy(args.load_model)
    model, data, robot, ankle_mapper, policy_ankle_model = build_sim(args, cfg)

    print(f"[INFO] policy: {args.load_model}")
    print(f"[INFO] XML: {args.model_path}")
    print("[INFO] plant: upper/lower physical motors")
    if cfg.upper_lower_action_mode == "direct":
        print("[INFO] policy IO: upper/lower action direct to upper/lower motors")
    else:
        print("[INFO] policy IO: upper/lower action replayed through training UL->PR conversion")
    print("[INFO] obs: gyro=angular-velocity sensor, gravity=orientation sensor quaternion")
    if cfg.upper_lower_observation_source == "control":
        print("[INFO] ankle obs: upper/lower control joint position/velocity")
    else:
        print("[INFO] ankle obs: passive pitch/roll converted to upper/lower like training")
    if args.ankle_control == "pr_joint_torque":
        print("[INFO] ankle control: pitch/roll task-space PD torque -> passive pitch/roll qfrc_applied")
    elif args.ankle_control in {"model_pr_task_torque", "pr_task_torque"}:
        print("[INFO] ankle control: pitch/roll task-space PD torque -> fitted-model inv(J.T) -> upper/lower motor torque")
    elif args.ankle_control == "exact_pr_task_torque":
        print("[INFO] ankle control: pitch/roll task-space PD torque -> tendon-geometry inv(J.T) -> upper/lower motor torque")
    else:
        print("[INFO] ankle control: direct upper/lower position PD")
    print(f"[INFO] training ankle_pd_space: {cfg.ankle_pd_space}")
    print("[INFO] control: explicit torque PD")
    print(f"[INFO] policy action clip: {cfg.action_clip}")
    print(f"[INFO] arm action scale: {args.arm_action_scale:g}")
    print(f"[INFO] ankle model: {policy_ankle_model.path}")
    if args.startup_freeze > 0.0:
        print(f"[INFO] startup settle: freeze base and hold default for {args.startup_freeze:.2f}s")
    print(f"[INFO] cmd: ({args.cmd_vel[0]:.2f}, {args.cmd_vel[1]:.2f}, {args.cmd_vel[2]:.2f})")
    print("Keyboard: 8/2 vx, 4/6 vy, 7/9 yaw, arrows vx/yaw, 0 reset, F follow")

    listener = None if args.headless else full.start_keyboard_listener()
    full.CommandState.set_initial(tuple(args.cmd_vel))

    renderer, camera, writer, viewer = full.init_render(model, data, args, cfg)

    last_action = np.zeros(cfg.num_actions, dtype=np.float64)
    scaled_action = np.zeros(cfg.num_actions, dtype=np.float64)
    target_pos = robot.stand_pos.copy()
    control_target = robot.stand_control_pos.copy()
    data.ctrl[:] = 0.0

    ul_ids = full.upper_lower_control_ids(cfg)
    real_ankle_default = robot.stand_control_pos[ul_ids].copy()
    real_ankle_target = real_ankle_default.copy()
    ankle_action_ids = np.array(
        [
            i
            for i, name in enumerate(cfg.policy_joint_names)
            if full.POLICY_ANKLE_TO_CONTROL_JOINT.get(name, name) in full.UPPER_LOWER_JOINT_NAMES
        ],
        dtype=np.int32,
    )
    arm_action_ids = arm_policy_action_ids(cfg)
    warmup_steps = int(round(max(0.0, args.stand_warmup) / cfg.dt))
    startup_freeze_steps = int(round(max(0.0, args.startup_freeze) / cfg.dt))
    policy_start_steps = max(warmup_steps, startup_freeze_steps)
    command_ramp_steps = int(round(max(0.0, args.command_ramp) / cfg.dt))

    initial_qpos = data.qpos.copy()
    initial_qvel = data.qvel.copy()
    frozen_root_qpos = initial_qpos[:7].copy()
    min_root_z = float("inf")
    max_abs_yaw_rate = 0.0
    max_abs_vy = 0.0
    start_time = time.time()
    printed_first_action = False
    printed_first_obs = False

    total_steps = int(args.sim_duration / cfg.dt)
    for step in tqdm(range(total_steps), desc="Simulating"):
        reset_result = reset_state(data, initial_qpos, initial_qvel, robot, cfg, last_action, real_ankle_default)
        if reset_result is not None:
            real_ankle_target, target_pos, control_target = reset_result
            scaled_action[:] = 0.0
            mujoco.mj_forward(model, data)

        qpos, qvel, lin_vel_b, omega_b, gravity_b = full.get_base_obs(
            data,
            use_root_ang_vel=args.base_ang_vel_from_qvel,
            use_sensor_orientation=args.sensor_orientation,
        )
        q = qpos[robot.qpos_ids]
        dq = qvel[robot.qvel_ids]
        control_q = qpos[robot.control_qpos_ids]
        control_dq = qvel[robot.control_qvel_ids]

        min_root_z = min(min_root_z, float(qpos[2]))
        max_abs_yaw_rate = max(max_abs_yaw_rate, abs(float(omega_b[2])))
        max_abs_vy = max(max_abs_vy, abs(float(lin_vel_b[1])))

        if step % cfg.decimation == 0:
            settling = step < policy_start_steps
            if args.hold_default or settling:
                last_action[:] = 0.0
                scaled_action[:] = 0.0
                target_pos = robot.stand_pos.copy()
                control_target = robot.stand_control_pos.copy()
            else:
                obs = full.make_observation(
                    q,
                    dq,
                    control_q,
                    control_dq,
                    omega_b,
                    gravity_b,
                    last_action,
                    robot,
                    cfg,
                    policy_ankle_model,
                )
                if command_ramp_steps > 0:
                    alpha = np.clip((step - policy_start_steps) / command_ramp_steps, 0.0, 1.0)
                    obs[0, 6:9] = np.array([full.CommandState.vx, full.CommandState.vy, full.CommandState.dyaw]) * alpha
                if warmup_steps > 0 and step < warmup_steps:
                    obs[0, 6:9] = 0.0

                if step < policy_start_steps:
                    last_action[:] = 0.0
                    scaled_action[:] = 0.0
                else:
                    if args.debug_first_obs and not printed_first_obs:
                        full.print_first_obs_debug(obs, q, dq, control_q, last_action, robot, cfg)
                        printed_first_obs = True
                    raw_action = policy(obs).astype(np.float64, copy=False)
                    last_action[:] = raw_action
                    if cfg.action_clip is not None:
                        np.clip(last_action, -cfg.action_clip, cfg.action_clip, out=last_action)
                    scaled_action[:] = last_action
                    if len(ankle_action_ids):
                        scaled_action[ankle_action_ids] *= args.ankle_action_scale
                    if len(arm_action_ids):
                        scaled_action[arm_action_ids] *= args.arm_action_scale
                    scaled_action[:] *= args.action_scale

                if cfg.upper_lower_action_mode == "direct":
                    control_target = full.upper_lower_action_to_control_target(scaled_action, robot, cfg, policy_ankle_model)
                    target_pos = robot.default_pos.copy()
                else:
                    target_pos = full.upper_lower_policy_action_to_pitch_roll_target(
                        scaled_action,
                        robot,
                        cfg,
                        policy_ankle_model,
                    )
                    control_target = ankle_mapper.observe_to_control_target(target_pos)

                if args.debug_first_action and not printed_first_action and step >= warmup_steps:
                    print(
                        "[DEBUG] first raw action unclipped="
                        f"{np.array2string(raw_action, precision=3, suppress_small=True)}"
                    )
                    print(
                        "[DEBUG] first raw action clipped="
                        f"{np.array2string(last_action, precision=3, suppress_small=True)}"
                    )
                    print(
                        "[DEBUG] first scaled action="
                        f"{np.array2string(scaled_action, precision=3, suppress_small=True)}"
                    )
                    full.print_first_action_debug(scaled_action, target_pos, control_target, robot, cfg)
                    printed_first_action = True

            if step < policy_start_steps:
                target_pos = robot.stand_pos.copy()
                control_target = robot.stand_control_pos.copy()
            desired_ankle = control_target[ul_ids].copy()
            real_ankle_target = full.limit_real_ankle_command(desired_ankle, real_ankle_target, real_ankle_default, cfg)
            control_target[ul_ids] = real_ankle_target

            policy_step = step // cfg.decimation
            if policy_step % max(1, args.print_every) == 0:
                ankle_rel = real_ankle_target - real_ankle_default
                print(
                    f"t={step * cfg.dt:5.2f}s "
                    f"vel=({lin_vel_b[0]:+.2f},{lin_vel_b[1]:+.2f},{omega_b[2]:+.2f}) "
                    f"z={qpos[2]:+.3f} "
                    "ankle_rel="
                    f"({ankle_rel[0]:+.3f},{ankle_rel[1]:+.3f},{ankle_rel[2]:+.3f},{ankle_rel[3]:+.3f})"
                )

            if not full.render(data, renderer, camera, writer, viewer, args.headless):
                break

        data.qfrc_applied[:] = 0.0
        torque = pd_torque(control_q, control_dq, control_target, cfg)
        if args.ankle_control == "pr_joint_torque" and cfg.upper_lower_action_mode != "direct":
            torque[ul_ids] = 0.0
            left_task_ids, left_tau_pr = pitch_roll_task_torque("left", q, dq, target_pos, cfg)
            right_task_ids, right_tau_pr = pitch_roll_task_torque("right", q, dq, target_pos, cfg)
            data.qfrc_applied[robot.qvel_ids[left_task_ids]] += left_tau_pr
            data.qfrc_applied[robot.qvel_ids[right_task_ids]] += right_tau_pr
        elif args.ankle_control != "direct_pd" and cfg.upper_lower_action_mode != "direct":
            if args.ankle_control == "exact_pr_task_torque":
                torque[ul_ids[:2]] = full.ankle_torque_control("left", q, dq, target_pos, robot, cfg, ankle_mapper)
                torque[ul_ids[2:]] = full.ankle_torque_control("right", q, dq, target_pos, robot, cfg, ankle_mapper)
            else:
                torque[ul_ids[:2]] = model_ankle_torque_control(
                    "left", q, dq, target_pos, robot, cfg, policy_ankle_model
                )
                torque[ul_ids[2:]] = model_ankle_torque_control(
                    "right", q, dq, target_pos, robot, cfg, policy_ankle_model
                )
        data.ctrl[robot.actuator_ids] = torque
        mujoco.mj_step(model, data)
        if args.freeze_base or step < startup_freeze_steps:
            data.qpos[:7] = frozen_root_qpos
            data.qvel[:6] = 0.0
            mujoco.mj_forward(model, data)
        elapsed = time.time() - start_time
        target_time = (step + 1) * cfg.dt
        if elapsed < target_time:
            time.sleep(target_time - elapsed)

    print(
        "[SUMMARY] "
        f"min_root_z={min_root_z:.3f}, "
        f"max_abs_yaw_rate={max_abs_yaw_rate:.3f}, "
        f"max_abs_vy={max_abs_vy:.3f}, "
        f"final_contacts={full.summarize_contacts(model, data)}"
    )

    if args.headless:
        writer.release()
        print(f"[INFO] saved video: {args.output}")
    else:
        viewer.close()
    if listener is not None:
        listener.stop()


def main() -> None:
    args = parse_args()
    cfg = load_runtime(args)
    if has_physical_upper_lower_plant(cfg):
        run(args, cfg)
    else:
        run_training_plant(args, cfg)


if __name__ == "__main__":
    main()
