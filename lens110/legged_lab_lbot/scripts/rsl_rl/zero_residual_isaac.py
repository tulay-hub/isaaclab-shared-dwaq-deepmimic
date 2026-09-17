"""Isaac 零动作测试: 不用策略, 把 21 关节目标固定在 frame0, 跑 5 秒看是否站住。"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="LeggedLab-Isaac--Deepmimic-Lens110-Play-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import gymnasium as gym
import numpy as np
import torch
from isaaclab_tasks.utils.hydra import hydra_task_config

import legged_lab.tasks  # noqa: F401


def _repository_root() -> Path:
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        if (parent / "projects").is_dir() and (parent / "frameworks").is_dir():
            return parent
    raise RuntimeError("Cannot find the reorganized Lens110 repository root")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = 1
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.unwrapped.reset()

    motion = np.load(
        _repository_root()
        / "projects"
        / "01_dance_whole_body"
        / "data"
        / "processed"
        / "retargeted_actions"
        / "tangbohushuo_dance"
        / "training"
        / "tangbohushuoDJ_v2_100hz.npz"
    )
    frame0 = motion["joint_pos"][0]
    robot = env.unwrapped.scene["robot"]
    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    default_q = robot.data.default_joint_pos[0].cpu().numpy()
    # target = default + 0.25*action -> action 使 target=frame0
    action = (frame0 - default_q) / 0.25

    for s in range(2500):  # 500Hz * 5s
        env.unwrapped.step(torch.from_numpy(action[None, :]).float().to(env.unwrapped.device))
        if s % 250 == 0:
            print(f"step {s} ({s/500:.1f}s): root_z={float(robot.data.root_pos_w[0,2]):.3f}")
    env.close()


main()
simulation_app.close()
