#!/usr/bin/env python3
"""Convenience launcher for Lens110 12-DoF AMP training.

This wrapper keeps all real training logic in scripts/rsl_rl/train.py and only
sets safe defaults for the LegPitchRoll task used by sim2sim_lens110_12dof.py.

Examples:
    python scripts/train_lens110_12dof.py
    python scripts/train_lens110_12dof.py --max_iterations 80000 --run_name straight_gait
    python scripts/train_lens110_12dof.py --resume --load_run 2026-07-15_17-39-11 --checkpoint model_2750.pt --learning_rate 3e-5

Any unknown arguments are forwarded to Hydra, for example:
    python scripts/train_lens110_12dof.py agent.algorithm.entropy_coef=0.006
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path


DEFAULT_TASK = "LeggedLab-Isaac-AMP-Lens110-LegPitchRoll-v0"
DEFAULT_NUM_ENVS = 2048
DEFAULT_MAX_ITERATIONS = 50000
DEFAULT_RUN_NAME = "12dof_gait_tune"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Lens110 12-DoF leg pitch/roll AMP policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", default=DEFAULT_TASK, help="Gym task id.")
    parser.add_argument("--num_envs", type=int, default=DEFAULT_NUM_ENVS, help="Number of Isaac Lab envs.")
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help="Learning iterations for this training run.",
    )
    parser.add_argument("--run_name", default=DEFAULT_RUN_NAME, help="Suffix for the log directory.")
    parser.add_argument("--seed", type=int, default=None, help="Training seed. Use -1 for random seed.")
    parser.add_argument("--device", default=None, help="Isaac/RSL device, for example cuda:0 or cpu.")

    parser.add_argument("--resume", action="store_true", help="Resume from a previous checkpoint.")
    parser.add_argument("--load_run", default=None, help="Run directory under logs/rsl_rl/lens110_amp_leg_pitch_roll.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint filename, for example model_2750.pt.")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Optional PPO/AMP learning-rate override. For resume tuning, 3e-5 is a conservative start.",
    )

    parser.add_argument("--logger", choices=("tensorboard", "wandb", "neptune"), default=None)
    parser.add_argument("--log_project_name", default=None)
    parser.add_argument("--video", action="store_true", help="Enable periodic training video recording.")
    parser.add_argument("--video_length", type=int, default=None)
    parser.add_argument("--video_interval", type=int, default=None)

    parser.add_argument("--render", action="store_true", help="Run with viewer instead of headless mode.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without launching training.")
    return parser


def _append_if_value(cmd: list[str], option: str, value: object | None) -> None:
    if value is not None:
        cmd.extend([option, str(value)])


def build_train_command(args: argparse.Namespace, hydra_overrides: list[str]) -> list[str]:
    root = _repo_root()
    train_py = root / "scripts" / "rsl_rl" / "train.py"
    if not train_py.is_file():
        raise FileNotFoundError(f"Cannot find training entry point: {train_py}")

    cmd = [
        sys.executable,
        str(train_py),
        "--task",
        args.task,
        "--num_envs",
        str(args.num_envs),
        "--max_iterations",
        str(args.max_iterations),
    ]
    if not args.render:
        cmd.append("--headless")

    _append_if_value(cmd, "--run_name", args.run_name)
    _append_if_value(cmd, "--seed", args.seed)
    _append_if_value(cmd, "--device", args.device)

    if args.resume:
        cmd.append("--resume")
    _append_if_value(cmd, "--load_run", args.load_run)
    _append_if_value(cmd, "--checkpoint", args.checkpoint)

    _append_if_value(cmd, "--logger", args.logger)
    _append_if_value(cmd, "--log_project_name", args.log_project_name)
    if args.video:
        cmd.append("--video")
    _append_if_value(cmd, "--video_length", args.video_length)
    _append_if_value(cmd, "--video_interval", args.video_interval)

    if args.learning_rate is not None:
        cmd.append(f"agent.algorithm.learning_rate={args.learning_rate:g}")
    cmd.extend(hydra_overrides)
    return cmd


def main() -> None:
    parser = _build_parser()
    args, hydra_overrides = parser.parse_known_args()
    cmd = build_train_command(args, hydra_overrides)

    root = _repo_root()
    print("[INFO] Lens110 12-DoF training launcher")
    print(f"[INFO] repo: {root}")
    print(f"[INFO] task: {args.task}")
    print("[INFO] command:")
    print("  " + " ".join(shlex.quote(part) for part in cmd))
    if args.dry_run:
        return

    os.chdir(root)
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
