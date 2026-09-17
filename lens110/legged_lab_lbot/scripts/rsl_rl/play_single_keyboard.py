# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-robot play script with arrow-key command control."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL and keyboard speed control.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--export_onnx", action="store_true", default=False, help="Export policy as ONNX before playing.")
parser.add_argument("--follow_camera", action="store_true", default=False, help="Enable camera follow the robot.")
parser.add_argument("--follow_camera_distance", type=float, default=2.5, help="Camera distance behind the robot.")
parser.add_argument("--follow_camera_height", type=float, default=1.2, help="Camera height above the robot base.")
parser.add_argument("--cmd_vx_step", type=float, default=0.10, help="Incremental x velocity per UP/DOWN key press.")
parser.add_argument("--cmd_yaw_step", type=float, default=0.20, help="Incremental yaw rate per LEFT/RIGHT key press.")
parser.add_argument(
    "--cmd_vx_limit",
    type=float,
    nargs=2,
    default=(-2.5, 3.0),
    metavar=("MIN", "MAX"),
    help="Keyboard x velocity limits. Use a negative MIN to walk backwards.",
)
parser.add_argument(
    "--cmd_yaw_limit",
    type=float,
    nargs=2,
    default=(-1.5, 1.5),
    metavar=("MIN", "MAX"),
    help="Keyboard yaw-rate limits for turning.",
)
parser.add_argument(
    "--command_name",
    type=str,
    default=None,
    help="Command term name to control (auto-detected from task config if not provided).",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# single-robot mode
args_cli.num_envs = 1

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import inspect
import os
import time

import torch

from rsl_rl.algorithms import PPO, PPOAMP
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

try:
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ImportError:
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

# Import extensions to set up environment tasks
import legged_lab.tasks  # noqa: F401

# PLACEHOLDER: Extension template (do not remove this comment)


class KeyboardVelocityController:
    """Handle arrow-key velocity control for one simulated robot."""

    def __init__(self, command_step: torch.Tensor, command_limits: torch.Tensor, num_envs: int, device: torch.device):
        self.command_step = command_step.to(device=device)
        self.command_limits = command_limits.to(device=device)
        self.command = torch.zeros((num_envs, 3), device=device, dtype=torch.float)
        self.current_command_name: str | None = None
        self.enabled = False
        self.stop_requested = False
        self._sub = None
        self._input = None
        self._keyboard = None

        if command_limits.ndim != 2 or command_limits.shape != (3, 2):
            raise ValueError("command_limits must have shape (3, 2)")

        try:
            import carb
            import omni

            self._input = carb.input.acquire_input_interface()
            self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
            self._sub = self._input.subscribe_to_keyboard_events(
                self._keyboard,
                self._on_keyboard_event,
            )
            self.enabled = True
            self._show_controls()
        except Exception as exc:  # pragma: no cover - depends on Isaac headless/input availability
            print(f"[WARN] Keyboard control unavailable: {exc}")

    def close(self) -> None:
        if self._input is not None and self._keyboard is not None and self._sub is not None:
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._sub)
            self._sub = None

    def _show_controls(self) -> None:
        print("[INFO] Keyboard control: Up/Down: +/- vx, Left/Right: +/- yaw, Space: zero, Esc: quit, R: reset")
        limits = self.command_limits.detach().cpu().tolist()
        print(
            "[INFO] Command limits: "
            f"vx=[{limits[0][0]:+.2f}, {limits[0][1]:+.2f}], "
            f"vy=[{limits[1][0]:+.2f}, {limits[1][1]:+.2f}], "
            f"yaw=[{limits[2][0]:+.2f}, {limits[2][1]:+.2f}]"
        )

    def _clamp_command(self) -> None:
        self.command[:, 0].clamp_(self.command_limits[0, 0], self.command_limits[0, 1])
        self.command[:, 1].clamp_(self.command_limits[1, 0], self.command_limits[1, 1])
        self.command[:, 2].clamp_(self.command_limits[2, 0], self.command_limits[2, 1])

    def _print_command(self) -> None:
        vx, vy, yaw = self.command[0].tolist()
        print(f"[CMD] vx={vx:+.2f}, vy={vy:+.2f}, yaw={yaw:+.2f}")

    def _on_keyboard_event(self, event, *args, **kwargs):
        from carb.input import KeyboardEventType

        if event.type != KeyboardEventType.KEY_PRESS:
            return True

        key = event.input.name.upper()
        if key in {"UP", "UP_ARROW", "ARROW_UP", "KEY_UP"}:
            self.command[:, 0] += self.command_step[0]
        elif key in {"DOWN", "DOWN_ARROW", "ARROW_DOWN", "KEY_DOWN"}:
            self.command[:, 0] -= self.command_step[0]
        elif key in {"LEFT", "LEFT_ARROW", "ARROW_LEFT", "KEY_LEFT"}:
            self.command[:, 2] += self.command_step[2]
        elif key in {"RIGHT", "RIGHT_ARROW", "ARROW_RIGHT", "KEY_RIGHT"}:
            self.command[:, 2] -= self.command_step[2]
        elif key in {"SPACE", "SPACEBAR"}:
            self.command.zero_()
        elif key in {"ESC", "ESCAPE"}:
            self.stop_requested = True
            return False
        elif key == "R":
            self._request_reset = True
            return True
        else:
            return True

        self._clamp_command()
        self._print_command()
        return True

    @property
    def request_reset(self) -> bool:
        return bool(getattr(self, "_request_reset", False))

    def consume_reset(self) -> bool:
        requested = self.request_reset
        self._request_reset = False
        return requested


def _strip_unsupported_ppo_optimizer(agent_cfg: RslRlBaseRunnerCfg) -> RslRlBaseRunnerCfg:
    """Drop PPO optimizer field for older local rsl_rl builds."""
    algorithm_cfg = getattr(agent_cfg, "algorithm", None)
    if algorithm_cfg is None or not hasattr(algorithm_cfg, "optimizer"):
        return agent_cfg
    if getattr(algorithm_cfg, "class_name", None) not in {"PPO", "PPOAMP"}:
        return agent_cfg
    optimizer_name = getattr(algorithm_cfg, "optimizer")
    if optimizer_name != "adam":
        print(
            "[WARNING]: The `optimizer` parameter for PPO/PPOAMP is not supported by the current local rsl_rl. "
            "Defaulting to `adam`."
        )
    del algorithm_cfg.optimizer
    return agent_cfg


def _sanitize_runner_cfg(train_cfg: dict) -> dict:
    """Drop algorithm kwargs unsupported by the local rsl_rl version."""
    algorithm_cfg = train_cfg.get("algorithm")
    if not isinstance(algorithm_cfg, dict):
        return train_cfg
    class_name = algorithm_cfg.get("class_name")
    algorithm_classes = {
        "PPO": PPO,
        "PPOAMP": PPOAMP,
    }
    algorithm_class = algorithm_classes.get(class_name)
    if algorithm_class is None:
        return train_cfg
    valid_params = set(inspect.signature(algorithm_class.__init__).parameters.keys())
    valid_params.discard("self")
    filtered_algorithm_cfg = {
        key: value for key, value in algorithm_cfg.items() if key == "class_name" or key in valid_params
    }
    dropped_keys = sorted(set(algorithm_cfg.keys()) - set(filtered_algorithm_cfg.keys()))
    if dropped_keys:
        print(f"[INFO]: Dropping unsupported {class_name} config keys for local rsl_rl: {dropped_keys}")
    train_cfg["algorithm"] = filtered_algorithm_cfg
    return train_cfg


def _unwrap_env(env):
    current = env
    while hasattr(current, "unwrapped") and current.unwrapped is not current:
        current = current.unwrapped
    return current


def _extract_simulation_context(env):
    if hasattr(env, "sim"):
        return env.sim
    scene = getattr(env, "scene", None)
    return getattr(scene, "sim", None) if scene is not None else None


def _extract_robot_position(env):
    robot = getattr(env, "robot", None)
    if robot is None:
        scene = getattr(env, "scene", None)
        if scene is not None and hasattr(scene, "__getitem__"):
            try:
                robot = scene["robot"]
            except Exception:  # pragma: no cover - scene implementation can vary
                robot = None

    if robot is None or not hasattr(robot, "data"):
        return None

    data = robot.data
    if hasattr(data, "root_state_w"):
        return data.root_state_w[:, :3]
    if hasattr(data, "root_pos_w"):
        return data.root_pos_w[:, :3]
    if hasattr(data, "root_pose_w"):
        return data.root_pose_w[:, :3]
    return None


def _follow_camera_step(sim_ctx, robot_pos, args) -> bool:
    if sim_ctx is None or robot_pos is None:
        return False

    pos = robot_pos[0] if getattr(robot_pos, "ndim", 1) == 2 else robot_pos
    if not torch.is_tensor(pos):
        return False

    pos_cpu = pos.detach().to(device="cpu", dtype=torch.float).tolist()
    eye = [
        pos_cpu[0] + args.follow_camera_distance,
        pos_cpu[1] + args.follow_camera_distance,
        pos_cpu[2] + args.follow_camera_height,
    ]
    target = [pos_cpu[0], pos_cpu[1], pos_cpu[2]]

    try:
        sim_ctx.set_camera_view(eye=eye, target=target)
        return True
    except Exception:
        return False


def _discover_command_term_name(cfg) -> str | None:
    commands_cfg = getattr(cfg, "commands", None)
    if commands_cfg is None:
        return None
    for name, value in vars(commands_cfg).items():
        if name.startswith("_"):
            continue
        if value is None:
            continue
        if hasattr(value, "ranges"):
            return name
    return None


def _resolve_limits_from_cfg(cmd_cfg) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    # return (vx_min, vx_max), (vy_min,vy_max), (yaw_min,yaw_max)
    if cmd_cfg is None:
        return (-1.0, 1.0), (0.0, 0.0), (-1.0, 1.0)
    ranges = getattr(cmd_cfg, "ranges", None)
    if ranges is None:
        return (-1.0, 1.0), (0.0, 0.0), (-1.0, 1.0)

    vx = getattr(ranges, "lin_vel_x", (-1.0, 1.0))
    vy = getattr(ranges, "lin_vel_y", (0.0, 0.0))
    yaw = getattr(ranges, "ang_vel_z", (-1.0, 1.0))
    return (tuple(float(v) for v in vx), tuple(float(v) for v in vy), tuple(float(v) for v in yaw))


def _apply_command(command_env, command: torch.Tensor, command_name: str | None = None) -> bool:
    if command is None or command.numel() == 0:
        return False

    if command.shape[-1] != 3:
        raise ValueError("Command must have shape (..., 3).")

    target = command

    command_manager = getattr(command_env, "command_manager", None)
    if command_manager is not None:
        # preferred direct write path for ManagerBased envs
        if command_name is not None and hasattr(command_manager, "get_command"):
            try:
                running_command = command_manager.get_command(command_name)
                if torch.is_tensor(running_command):
                    running_command[:, :3] = target
                    return True
            except Exception:
                pass

        if command_name is not None and hasattr(command_manager, "set_command"):
            try:
                command_manager.set_command(command_name, target)
                return True
            except Exception:
                pass

        if command_name is not None and hasattr(command_manager, "get_term"):
            try:
                term = command_manager.get_term(command_name)
                for attr in ("command", "commands", "_command"):
                    value = getattr(term, attr, None)
                    if torch.is_tensor(value) and value.shape[:2] == target.shape[:2]:
                        value[:, :3] = target
                        return True
            except Exception:
                pass

        # fallback: walk manager terms
        term_map = getattr(command_manager, "_terms", {})
        if isinstance(term_map, dict):
            # only try requested term first, then any velocity-like term
            term_candidates = []
            if command_name is not None and command_name in term_map:
                term_candidates.append(term_map[command_name])
            for name, term in term_map.items():
                if term not in term_candidates:
                    term_candidates.append(term)

            for term in term_candidates:
                if term is None:
                    continue
                for attr in ("command", "commands", "_command"):
                    value = getattr(term, attr, None)
                    if torch.is_tensor(value) and value.shape[:2] == target.shape[:2]:
                        value[:, :3] = target
                        return True

    command_generator = getattr(command_env, "command_generator", None)
    if command_generator is not None:
        for attr in ("command", "commands", "_command"):
            value = getattr(command_generator, attr, None)
            if torch.is_tensor(value) and value.shape[:2] == target.shape[:2]:
                value[:, :3] = target
                return True

    return False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent and keyboard velocity control."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = _strip_unsupported_ppo_optimizer(agent_cfg)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    base_env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(base_env.unwrapped, DirectMARLEnv):
        base_env = multi_agent_to_single_agent(base_env)

    sim_ctx = _extract_simulation_context(_unwrap_env(base_env))

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        base_env = gym.wrappers.RecordVideo(base_env, **video_kwargs)

    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner_cfg = _sanitize_runner_cfg(agent_cfg.to_dict())

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, runner_cfg, log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "AMPRunner":
        from rsl_rl.runners import AMPRunner

        runner = AMPRunner(env, runner_cfg, log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, runner_cfg, log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path, map_location=agent_cfg.device)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to jit, and optionally onnx
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    if args_cli.export_onnx:
        try:
            export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")
        except Exception as exc:
            print(f"[WARN] Failed to export ONNX policy; continuing play without ONNX export: {exc}")

    command_env = _unwrap_env(env)

    command_name = args_cli.command_name
    detected_name = _discover_command_term_name(env_cfg)
    if command_name is None:
        command_name = detected_name
    if command_name is None:
        print("[WARN] Unable to infer command term name; command control may not work.")

    cmd_cfg = None
    if command_name is not None:
        cmd_cfg = getattr(getattr(env_cfg, "commands", None), command_name, None)
    _, vy_limits, _ = _resolve_limits_from_cfg(cmd_cfg)
    vx_limits = tuple(args_cli.cmd_vx_limit)
    yaw_limits = tuple(args_cli.cmd_yaw_limit)
    command_limits = torch.tensor([vx_limits, vy_limits, yaw_limits], device=env.unwrapped.device, dtype=torch.float)

    controller = KeyboardVelocityController(
        command_step=torch.tensor([args_cli.cmd_vx_step, 0.0, args_cli.cmd_yaw_step], device=env.unwrapped.device),
        command_limits=command_limits,
        num_envs=1,
        device=env.unwrapped.device,
    )
    controller.current_command_name = command_name

    dt = env.unwrapped.step_dt

    follow_warned = False
    if args_cli.follow_camera:
        if sim_ctx is None:
            print("[WARN] Camera follow requested, but simulation context was not found.")
            follow_warned = True
        elif not _follow_camera_step(sim_ctx, _extract_robot_position(_unwrap_env(env)), args_cli):
            print("[WARN] Camera follow could not be initialized in this environment.")
            follow_warned = True

    # reset then force one-step standing command
    obs = env.get_observations()
    if not _apply_command(command_env, controller.command, command_name=command_name):
        print("[WARN] Could not bind command tensor to environment. Keyboard control may have no effect.")

    timestep = 0
    # simulate environment
    while simulation_app.is_running() and not controller.stop_requested:
        start_time = time.time()

        if controller.consume_reset():
            env.reset()

        _apply_command(command_env, controller.command, command_name=command_name)

        # run everything in inference mode
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)

        if args_cli.follow_camera and not follow_warned:
            if not _follow_camera_step(sim_ctx, _extract_robot_position(_unwrap_env(env)), args_cli):
                print("[WARN] Camera follow failed. Continue with last camera view.")
                follow_warned = True

        timestep += 1
        if args_cli.video and timestep >= args_cli.video_length:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    controller.close()
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
