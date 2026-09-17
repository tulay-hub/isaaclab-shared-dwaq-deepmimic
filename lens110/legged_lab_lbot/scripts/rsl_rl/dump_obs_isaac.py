"""在 Isaac 里用 34600 策略从第 0 帧跑 10 步, 打印 obs/action 前 20 维。

用法:
    python scripts/rsl_rl/dump_obs_isaac.py --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

import cli_args

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="LeggedLab-Isaac--Deepmimic-Lens110-Play-v0")
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
from rsl_rl.runners import OnPolicyRunner

import legged_lab.tasks  # noqa: F401


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = 1
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    resume_path = os.path.abspath(args_cli.checkpoint)
    runner.load(resume_path)
    policy = runner.alg.policy
    policy.eval()

    obs = env.get_observations()
    with torch.inference_mode():
        for step in range(10):
            o = obs["policy"].squeeze(0).cpu().numpy()
            print(f"step {step}: obs.shape={o.shape}")
            print(f"  root_rot_tan_norm[0:6]      ={o[0:6]}")
            print(f"  root_ang_vel_w[6:9]         ={o[6:9]}")
            print(f"  joint_pos[9:30]             ={o[9:30]}")
            print(f"  joint_vel[30:51]            ={o[30:51]}")
            print(f"  ref_root_rot_tan_norm[51:75]={o[51:75]}")
            print(f"  ref_joint_pos[75:96]        ={o[75:96]}")
            print(f"  ref_joint_pos[96:117]       ={o[96:117]}")
            print(f"  foot_contact[159:161]       ={o[159:161]}")
            action = policy.act_inference(obs)
            a = action.squeeze(0).cpu().numpy()
            print(f"  action min/max=({a.min():.3f},{a.max():.3f}) a={a}")
            obs, _, _, _ = env.step(action)
    env.close()


main()
simulation_app.close()
