"""从 Lens110 rsl_rl checkpoint 导出 ONNX + deploy_config.yaml (无 GUI)。

用法:
    python scripts/rsl_rl/export_onnx.py \
        --task Tracking-Flat-Lens110-v0 --headless \
        --checkpoint logs/rsl_rl/lens110_flat/<run>/model_XXXX.pt

输出: <checkpoint 所在目录>/exported/policy.onnx 和 deploy_config.yaml
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

import cli_args

parser = argparse.ArgumentParser(description="Export Lens110 rsl_rl checkpoint to ONNX.")
parser.add_argument("--task", type=str, default="Tracking-Flat-Lens110-v0")
parser.add_argument("--num_envs", type=int, default=1)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    if not os.path.isfile(args_cli.checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {args_cli.checkpoint}")
    resume_path = os.path.abspath(args_cli.checkpoint)
    runner.load(resume_path)

    policy_nn = runner.alg.policy
    normalizer = getattr(policy_nn, "actor_obs_normalizer", None) or getattr(policy_nn, "student_obs_normalizer", None)
    export_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_motion_policy_as_onnx(
        env.unwrapped, policy_nn, normalizer=normalizer, path=export_dir, filename="policy.onnx"
    )
    attach_onnx_metadata(env.unwrapped, "lens110", export_dir)
    print(f"[export] done: {export_dir}")
    env.close()


main()
simulation_app.close()
