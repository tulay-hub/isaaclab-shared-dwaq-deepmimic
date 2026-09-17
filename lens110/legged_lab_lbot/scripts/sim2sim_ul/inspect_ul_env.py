# SPDX-License-Identifier: BSD-3-Clause
"""Print the real IsaacLab joint/action/observation order for Lens110 upper/lower."""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect Lens110 upper/lower play environment ordering.")
parser.add_argument("--task", default="LeggedLab-Isaac-AMP-Lens110-UpperLower-Play-v0")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--checkpoint", default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import legged_lab.tasks  # noqa: F401


def _print_names(title: str, names) -> None:
    print(title)
    for index, name in enumerate(names):
        print(f"  {index:02d}: {name}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.observations.policy.enable_corruption = False

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    action_term = base_env.action_manager.get_term("joint_pos")

    _print_names("[INSPECT] robot.joint_names", robot.joint_names)
    _print_names("[INSPECT] action_term._joint_names", action_term._joint_names)
    _print_names("[INSPECT] action_term._action_names", action_term._action_names)
    print("[INSPECT] action scale:", action_term._scale[0].detach().cpu().numpy())
    if getattr(action_term, "_clip", None) is not None:
        print("[INSPECT] action clip:", action_term._clip[0].detach().cpu().numpy())

    obs, _ = env.reset()
    policy_obs = obs["policy"] if isinstance(obs, dict) else obs
    print("[INSPECT] policy obs shape:", tuple(policy_obs.shape))
    print("[INSPECT] policy obs[0]:", policy_obs[0].detach().cpu().numpy())

    if args_cli.checkpoint:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import AMPRunner, OnPolicyRunner

        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner_cls = AMPRunner if agent_cfg.class_name == "AMPRunner" else OnPolicyRunner
        runner = runner_cls(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args_cli.checkpoint, map_location=agent_cfg.device)
        policy = runner.get_inference_policy(device=base_env.device)
        with torch.inference_mode():
            action = policy(wrapped.get_observations())
        print("[INSPECT] first policy action:", action[0].detach().cpu().numpy())
        wrapped.close()
    else:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
