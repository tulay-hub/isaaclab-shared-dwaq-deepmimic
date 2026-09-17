"""Export DWAQ checkpoint to ONNX with URDF-ordered input/output.

The policy was trained with Isaac Lab's USD joint order (interleaved).
This wrapper accepts observations in URDF order, permutes internally to
USD for the network, and returns actions in URDF order -- so any consumer
(MuJoCo player, real robot SDK) can use URDF joint order directly.

Usage:
    python scripts/export_dwaq_onnx_urdf.py \
        --checkpoint logs/rsl_rl/lens110_dwaq/<run>/model_XXXX.pt \
        --output policy_dwaq_urdf.onnx
"""

import argparse

import torch
import torch.nn as nn


# URDF joint order (from lens110_21dof.urdf, top to bottom)
URDF_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "torso_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
]

# Isaac Lab USD joint order (from play_lens110_official_161.py)
USD_JOINTS = [
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

# For each USD index i, usd_to_urdf[i] = index in URDF order of the same joint
usd_to_urdf = torch.tensor([URDF_JOINTS.index(n) for n in USD_JOINTS], dtype=torch.long)
# For each URDF index j, urdf_to_usd[j] = index in USD order of the same joint
urdf_to_usd = torch.tensor([USD_JOINTS.index(n) for n in URDF_JOINTS], dtype=torch.long)

N_JOINTS = 21
OBS_DIM = 76
HISTORY_LEN = 5
OBS_HISTORY_DIM = OBS_DIM * HISTORY_LEN
CODE_DIM = 19

# Current observation layout: [ang_vel(0:3), proj_g(3:6), cmd(6:9), joint_pos(9:30), joint_vel(30:51), last_action(51:72), gait(72:76)]
# History layout: each term's 5-frame block is contiguous, in the same order
# as the observation group, matching Isaac Lab ObservationManager.
JOINT_POS_SLICE = slice(9, 30)
JOINT_VEL_SLICE = slice(30, 51)
LAST_ACTION_SLICE = slice(51, 72)
HIST_JOINT_POS_OFFSET = 3 * HISTORY_LEN * 3
HIST_JOINT_VEL_OFFSET = HIST_JOINT_POS_OFFSET + HISTORY_LEN * 21
HIST_LAST_ACTION_OFFSET = HIST_JOINT_VEL_OFFSET + HISTORY_LEN * 21


class ActorMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CODE_DIM + OBS_DIM, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, N_JOINTS),
        )

    def forward(self, x):
        return self.net(x)


class ContextVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(OBS_HISTORY_DIM, 128), nn.ELU(),
            nn.Linear(128, 64),
        )
        self.mean_vel = nn.Linear(64, 3)
        self.logvar_vel = nn.Linear(64, 3)
        self.mean_latent = nn.Linear(64, 16)
        self.logvar_latent = nn.Linear(64, 16)

    def encode_mean(self, x):
        h = self.encoder(x)
        return torch.cat([self.mean_vel(h), self.mean_latent(h)], dim=-1)


class DwaqOnnxWrapper(nn.Module):
    """Accepts URDF-ordered obs and obs_history, returns URDF-ordered action."""

    def __init__(self, actor: ActorMLP, vae: ContextVAE):
        super().__init__()
        self.actor = actor
        self.vae = vae
        self.register_buffer("usd_to_urdf", usd_to_urdf)
        self.register_buffer("urdf_to_usd", urdf_to_usd)

    def forward(self, obs_urdf: torch.Tensor, obs_history_urdf: torch.Tensor) -> torch.Tensor:
        batch = obs_urdf.shape[0]

        # Permute obs joints URDF -> USD
        obs_usd = obs_urdf.clone()
        obs_usd[..., JOINT_POS_SLICE] = obs_urdf[..., JOINT_POS_SLICE][..., self.usd_to_urdf]
        obs_usd[..., JOINT_VEL_SLICE] = obs_urdf[..., JOINT_VEL_SLICE][..., self.usd_to_urdf]
        obs_usd[..., LAST_ACTION_SLICE] = obs_urdf[..., LAST_ACTION_SLICE][..., self.usd_to_urdf]

        # Permute term-major history joints URDF -> USD.  Isaac Lab stores
        # each observation term's complete history contiguously.
        hist_usd = obs_history_urdf.clone()
        for frame in range(HISTORY_LEN):
            pos_base = HIST_JOINT_POS_OFFSET + frame * 21
            vel_base = HIST_JOINT_VEL_OFFSET + frame * 21
            act_base = HIST_LAST_ACTION_OFFSET + frame * 21
            hist_usd[..., pos_base: pos_base + 21] = obs_history_urdf[..., pos_base: pos_base + 21][..., self.usd_to_urdf]
            hist_usd[..., vel_base: vel_base + 21] = obs_history_urdf[..., vel_base: vel_base + 21][..., self.usd_to_urdf]
            hist_usd[..., act_base: act_base + 21] = obs_history_urdf[..., act_base: act_base + 21][..., self.usd_to_urdf]

        # VAE encode
        code = self.vae.encode_mean(hist_usd)

        # Actor
        actor_input = torch.cat([code, obs_usd], dim=-1)
        action_usd = self.actor(actor_input)

        # Permute action USD -> URDF
        action_urdf = action_usd[..., self.urdf_to_usd]

        return action_urdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="policy_dwaq_urdf.onnx")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]

    actor = ActorMLP()
    actor.load_state_dict({k.replace("actor.", "net."): v for k, v in sd.items() if k.startswith("actor.")})

    vae = ContextVAE()
    vae.load_state_dict(
        {k.replace("context_vae.", ""): v for k, v in sd.items() if k.startswith("context_vae.")},
        strict=False,
    )

    wrapper = DwaqOnnxWrapper(actor, vae)
    wrapper.eval()

    obs_dummy = torch.zeros(1, OBS_DIM)
    hist_dummy = torch.zeros(1, OBS_HISTORY_DIM)

    torch.onnx.export(
        wrapper,
        (obs_dummy, hist_dummy),
        args.output,
        input_names=["obs", "obs_history"],
        output_names=["action"],
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"[OK] Exported to {args.output}")

    # Verify
    import onnxruntime as ort
    sess = ort.InferenceSession(args.output)
    out = sess.run(None, {"obs": obs_dummy.numpy(), "obs_history": hist_dummy.numpy()})
    print(f"[OK] ONNX output shape: {out[0].shape}")


if __name__ == "__main__":
    main()
