"""不启动 Isaac, 直接把 Lens110 DeepMimic checkpoint 转成 ONNX + deploy_config。"""

import copy
import os
import sys

import torch
import yaml
from tensordict import TensorDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "rsl_rl"))
from rsl_rl.modules import ActorCritic  # noqa: E402


OBS_DIM = 161
NUM_ACTIONS = 21
JOINT_NAMES = [
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
DEFAULT_Q = [
    -0.14, -0.14, 0.0, -0.01, -0.01, 0.4, 0.4, -0.1, 0.1, 0.2, -0.2,
    0.36, 0.36, 0.0, 0.0, -0.2575, -0.2575, -0.8, -0.8, 0.0, 0.0,
]
# 与训练 lens110_dance.py 一致: 髋/膝 40/5, 腰 100/5, 肩/肘 20/5, 踝 55/5
STIFF = [40, 40, 100, 40, 40, 20, 20, 40, 40, 20, 20, 40, 40, 20, 20, 55, 55, 20, 20, 55, 55]
DAMP = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
OBS_NAMES = [
    "root_rot_tan_norm", "root_ang_vel_w", "joint_pos", "joint_vel",
    "ref_root_rot_tan_norm", "ref_joint_pos", "foot_contact",
]


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    if not ckpt or not os.path.isfile(ckpt):
        raise SystemExit(
            "usage: convert_pt_to_onnx_lens110.py <model_XXXX.pt> [arm_damping] [action_clip|None]"
        )
    # 可选覆盖: 旧 checkpoint 按各自训练配置导出 (默认: 手臂 20/5, clip=2.0)
    arm_damping = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    action_clip = 2.0
    if len(sys.argv) > 3:
        if sys.argv[3].lower() in {"none", "null"}:
            action_clip = None
        else:
            action_clip = float(sys.argv[3])
    damp = list(DAMP)
    for j, name in enumerate(JOINT_NAMES):
        if name.startswith(("left_", "right_")) and "shoulder" in name or "elbow" in name:
            damp[j] = arm_damping

    obs = TensorDict({"obs": torch.zeros(1, OBS_DIM)}, batch_size=[1])
    obs_groups = {"policy": ["obs"], "critic": ["obs"]}
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    # checkpoint 是否带观测归一化参数 (T800 风格开启 / 旧版关闭)
    has_norm = any("normalizer" in k for k in state["model_state_dict"])
    policy = ActorCritic(
        obs=obs,
        obs_groups=obs_groups,
        num_actions=NUM_ACTIONS,
        actor_obs_normalization=has_norm,
        critic_obs_normalization=has_norm,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
    )
    print(f"[convert] checkpoint 观测归一化: {has_norm}")
    policy.load_state_dict(state["model_state_dict"])
    policy.eval()

    export_dir = os.path.join(os.path.dirname(os.path.abspath(ckpt)), "exported")
    os.makedirs(export_dir, exist_ok=True)

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = copy.deepcopy(policy.actor)
            self.normalizer = copy.deepcopy(policy.actor_obs_normalizer)

        def forward(self, x):
            return self.actor(self.normalizer(x))

    wrapper = Wrapper().eval()
    torch.onnx.export(
        wrapper,
        torch.zeros(1, OBS_DIM),
        os.path.join(export_dir, "policy.onnx"),
        export_params=True,
        opset_version=11,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={},
    )
    deploy = {
        "default_joint_pos": DEFAULT_Q,
        "joint_names": JOINT_NAMES,
        "joint_stiffness": STIFF,
        "joint_damping": damp,
        "observation_names": OBS_NAMES,
        "observation_history_lengths": [1] * len(OBS_NAMES),
        "action_scale": 0.25,
        # reference: q_des = 参考当前帧关节角 + action_scale * clip(action, -1, 1)
        "action_mode": "reference",
        # T800 风格默认不 clip; 旧 checkpoint 导出时可传 1.0
        "action_clip": action_clip,
    }
    with open(os.path.join(export_dir, "deploy_config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(deploy, f, allow_unicode=True, default_flow_style=None)
    print("exported:", os.path.join(export_dir, "policy.onnx"))


if __name__ == "__main__":
    main()
