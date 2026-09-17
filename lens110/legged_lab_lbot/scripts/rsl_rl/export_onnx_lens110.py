"""Lens110 DeepMimic checkpoint -> ONNX + deploy_config.yaml (无 GUI)。

用法:
    python scripts/rsl_rl/export_onnx_lens110.py --headless \
        --checkpoint logs/rsl_rl/lens110_deepmimic/<run>/model_XXXX.pt

输出: <checkpoint 目录>/exported/policy.onnx + deploy_config.yaml
"""

import argparse
import copy
import os
import sys

import torch
from isaaclab.app import AppLauncher

import cli_args

parser = argparse.ArgumentParser(description="Export Lens110 DeepMimic checkpoint to ONNX.")
parser.add_argument("--task", type=str, default="LeggedLab-Isaac--Deepmimic-Lens110-v0")
parser.add_argument("--num_envs", type=int, default=1)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import yaml
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import legged_lab.tasks  # noqa: F401


class _OnnxPolicyExporter(torch.nn.Module):
    """与 legged_lab play.py 相同的 ONNX 导出器。"""

    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.is_recurrent = policy.is_recurrent
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
        else:
            raise ValueError("Policy does not have an actor/student module.")
        self.normalizer = copy.deepcopy(normalizer) if normalizer else torch.nn.Identity()

    def forward(self, x):
        return self.actor(self.normalizer(x))

    def export(self, path, filename):
        self.to("cpu")
        input_size = getattr(self.actor, "input_dim", None)
        if input_size is None:
            input_size = self.actor[0].in_features
        obs = torch.zeros(1, input_size)
        torch.onnx.export(
            self,
            obs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={},
        )


def export_policy_as_onnx(policy, path, normalizer=None, filename="policy.onnx", verbose=False):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    exporter = _OnnxPolicyExporter(policy, normalizer, verbose)
    exporter.export(path, filename)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    if not os.path.isfile(args_cli.checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {args_cli.checkpoint}")
    resume_path = os.path.abspath(args_cli.checkpoint)
    runner.load(resume_path)

    policy_nn = runner.alg.policy
    normalizer = getattr(policy_nn, "actor_obs_normalizer", None) or getattr(
        policy_nn, "student_obs_normalizer", None
    )
    export_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_onnx(policy_nn, export_dir, normalizer=normalizer, filename="policy.onnx")

    # deploy_config.yaml
    env_u = env.unwrapped
    robot = env_u.scene["robot"]
    action_term = env_u.action_manager.get_term("joint_pos")
    joint_names = list(action_term._joint_names)
    raw_names = list(robot.data.joint_names)
    idx = [raw_names.index(n) for n in joint_names]
    deploy = {
        "default_joint_pos": robot.data.default_joint_pos[0][idx].cpu().tolist(),
        "joint_names": joint_names,
        "joint_stiffness": robot.data.default_joint_stiffness[0][idx].cpu().tolist(),
        "joint_damping": robot.data.default_joint_damping[0][idx].cpu().tolist(),
        "observation_names": list(env_u.observation_manager.active_terms["policy"]),
        "observation_history_lengths": [1] * len(env_u.observation_manager.active_terms["policy"]),
        "action_scale": float(action_term._scale),
    }
    with open(os.path.join(export_dir, "deploy_config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(deploy, f, allow_unicode=True, default_flow_style=None)
    print(f"[export] done: {export_dir}")
    env.close()


main()
simulation_app.close()
