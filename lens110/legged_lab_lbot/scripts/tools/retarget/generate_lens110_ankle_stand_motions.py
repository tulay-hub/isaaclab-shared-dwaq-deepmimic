import argparse
import pathlib
import pickle

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


WORKSPACE_ROOT = next(
    (
        parent
        for parent in pathlib.Path(__file__).resolve().parents
        if (parent / "projects").is_dir() and (parent / "frameworks").is_dir()
    ),
    pathlib.Path(__file__).resolve().parents[3],
)
DEFAULT_SOURCE = WORKSPACE_ROOT / "projects/04_fall_to_stand/data/motions/walk0821/walk0821/lens110_13.pkl"
DEFAULT_XML = WORKSPACE_ROOT / "frameworks/shared/lens110_isaaclab/lens110/legged_lab_lbot/source/legged_lab/legged_lab/data/Robots/model_humanoid_lens110/mjcf/lens110.xml"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "projects/02_dance_half_body/data/processed/retargeted_actions/generated_stand_ankle"


def moving_average(values, window):
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window) / window
    padded = np.pad(values, [(pad, pad)] + [(0, 0)] * (values.ndim - 1), mode="edge")
    return np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="valid"), 0, padded)


def triangle_wave(phase):
    return 2.0 * np.abs(2.0 * (phase - np.floor(phase + 0.5))) - 1.0


def sinusoid_sweep(t, lo, hi, freq=0.25, phase=0.0):
    center = 0.5 * (lo + hi)
    amp = 0.5 * (hi - lo)
    return center + amp * np.sin(2.0 * np.pi * freq * t + phase)


def triangle_sweep(t, lo, hi, freq=0.18, phase=0.0):
    center = 0.5 * (lo + hi)
    amp = 0.5 * (hi - lo)
    return center + amp * triangle_wave(freq * t + phase)


def load_motion(path):
    with path.open("rb") as file:
        return pickle.load(file)


def get_joint_limits(model):
    limits = {}
    for joint_id in range(model.njnt):
        qpos_addr = model.jnt_qposadr[joint_id]
        if qpos_addr < 7:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        limits[name] = tuple(float(x) for x in model.jnt_range[joint_id])
    return limits


def get_qpos_dof_names(model):
    names = []
    for joint_id in range(model.njnt):
        qpos_addr = model.jnt_qposadr[joint_id]
        if qpos_addr >= 7:
            names.append((qpos_addr - 7, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)))
    return [name for _, name in sorted(names)]


def clip_to_limits(dof, dof_names, limits, margin=0.0):
    out = dof.copy()
    for idx, name in enumerate(dof_names):
        if name not in limits:
            continue
        lo, hi = limits[name]
        out[:, idx] = np.clip(out[:, idx], lo + margin, hi - margin)
    return out


def get_foot_ground_contact_ids(model):
    ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    if ground_id < 0:
        ground_id = 0

    foot_body_ids = set()
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if (
            "ankle" in name
            or name.startswith("lleg_Link4")
            or name.startswith("rleg_Link4")
        ):
            foot_body_ids.add(body_id)

    foot_geom_ids = {
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in foot_body_ids
    }
    return ground_id, foot_geom_ids


def lift_root_out_of_ground(model, motion, clearance=0.005):
    """Raise root z just enough that foot collision geoms do not penetrate ground."""
    ground_id, foot_geom_ids = get_foot_ground_contact_ids(model)
    if not foot_geom_ids:
        return 0.0

    data = mujoco.MjData(model)
    max_penetration = 0.0
    for frame_id in range(motion["dof_pos"].shape[0]):
        data.qpos[:3] = motion["root_pos"][frame_id]
        data.qpos[3:7] = motion["root_rot"][frame_id, [3, 0, 1, 2]]
        data.qpos[7:] = motion["dof_pos"][frame_id]
        mujoco.mj_forward(model, data)
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            is_foot_ground = (
                contact.geom1 == ground_id
                and contact.geom2 in foot_geom_ids
            ) or (
                contact.geom2 == ground_id
                and contact.geom1 in foot_geom_ids
            )
            if is_foot_ground:
                max_penetration = max(max_penetration, max(0.0, -float(contact.dist)))

    if max_penetration > 0.0:
        lift = max_penetration + clearance
        motion["root_pos"][:, 2] += lift
        return lift
    return 0.0


def make_base(source, frames, fps, target_dof_names):
    # Use a stable standing pose from the mean of 77_02.
    source_index = {name: idx for idx, name in enumerate(source["dof_names"])}
    source_mean = np.mean(source["dof_pos"], axis=0)
    target_mean = np.zeros(len(target_dof_names), dtype=np.float64)
    for idx, name in enumerate(target_dof_names):
        if name in source_index:
            target_mean[idx] = source_mean[source_index[name]]
    return {
        "fps": fps,
        "root_pos": np.repeat(np.mean(source["root_pos"], axis=0, keepdims=True), frames, axis=0),
        "root_rot": np.repeat(source["root_rot"][:1], frames, axis=0),
        "dof_pos": np.repeat(target_mean[None, :], frames, axis=0),
        "dof_names": list(target_dof_names),
        "local_body_pos": None,
        "link_body_list": None,
        "root_rot_format": "xyzw",
    }


def apply_small_root_bob(motion, t, amp=0.012):
    motion["root_pos"][:, 2] += amp * np.sin(2.0 * np.pi * 1.0 * t)


def apply_arm_and_body_motion(motion, t, index, mode):
    dof = motion["dof_pos"]
    if mode == "arms_stand":
        dof[:, index["torso_yaw_joint"]] += 0.18 * np.sin(2.0 * np.pi * 0.35 * t)
        for side, sign in (("left", 1.0), ("right", -1.0)):
            dof[:, index[f"{side}_shoulder_pitch_joint"]] += 0.85 * np.sin(2.0 * np.pi * 0.65 * t + (0 if side == "left" else np.pi))
            dof[:, index[f"{side}_shoulder_roll_joint"]] += sign * 0.35 * np.sin(2.0 * np.pi * 0.45 * t + 0.5)
            dof[:, index[f"{side}_shoulder_yaw_joint"]] += 0.45 * np.sin(2.0 * np.pi * 0.55 * t + (0.8 if side == "left" else -0.8))
            dof[:, index[f"{side}_elbow_joint"]] += 0.45 * np.sin(2.0 * np.pi * 0.7 * t + (1.5 if side == "left" else -1.5))
        # Gentle in-place stepping: keep feet near the ground but move hips/knees enough
        # to avoid a frozen lower body.
        dof[:, index["left_hip_pitch_joint"]] += 0.18 * np.sin(2.0 * np.pi * 0.8 * t)
        dof[:, index["right_hip_pitch_joint"]] += 0.18 * np.sin(2.0 * np.pi * 0.8 * t + np.pi)
        dof[:, index["left_knee_joint"]] += 0.14 * np.maximum(0.0, np.sin(2.0 * np.pi * 0.8 * t))
        dof[:, index["right_knee_joint"]] += 0.14 * np.maximum(0.0, np.sin(2.0 * np.pi * 0.8 * t + np.pi))
        apply_small_root_bob(motion, t, amp=0.008)


def apply_full_ankle_sweep(motion, t, index, limits, sides=("left", "right"), freq_scale=1.0):
    for side_id, side in enumerate(sides):
        pitch_name = f"{side}_ankle_pitch_joint"
        roll_name = f"{side}_ankle_roll_joint"
        pitch_lo, pitch_hi = limits[pitch_name]
        roll_lo, roll_hi = limits[roll_name]
        phase = side_id * np.pi / 2.0
        motion["dof_pos"][:, index[pitch_name]] = sinusoid_sweep(t, pitch_lo, pitch_hi, freq=0.22 * freq_scale, phase=phase)
        motion["dof_pos"][:, index[roll_name]] = triangle_sweep(t, roll_lo, roll_hi, freq=0.17 * freq_scale, phase=0.25 * side_id)


def apply_scaled_ankle_sweep(motion, t, index, limits, sides=("left", "right"), scale=0.6, freq_scale=1.0):
    for side_id, side in enumerate(sides):
        pitch_name = f"{side}_ankle_pitch_joint"
        roll_name = f"{side}_ankle_roll_joint"
        pitch_lo, pitch_hi = limits[pitch_name]
        roll_lo, roll_hi = limits[roll_name]
        pitch_mid = 0.5 * (pitch_lo + pitch_hi)
        roll_mid = 0.5 * (roll_lo + roll_hi)
        pitch_half = 0.5 * (pitch_hi - pitch_lo) * scale
        roll_half = 0.5 * (roll_hi - roll_lo) * scale
        phase = side_id * np.pi / 2.0
        motion["dof_pos"][:, index[pitch_name]] = sinusoid_sweep(
            t,
            pitch_mid - pitch_half,
            pitch_mid + pitch_half,
            freq=0.22 * freq_scale,
            phase=phase,
        )
        motion["dof_pos"][:, index[roll_name]] = triangle_sweep(
            t,
            roll_mid - roll_half,
            roll_mid + roll_half,
            freq=0.17 * freq_scale,
            phase=0.25 * side_id,
        )


def make_single_leg_high_knee(source, frames, fps, limits, target_dof_names, support_side="left"):
    motion = make_base(source, frames, fps, target_dof_names)
    index = {name: idx for idx, name in enumerate(motion["dof_names"])}
    t = np.arange(frames) / fps
    lift_side = "right" if support_side == "left" else "left"
    support_sign = 1.0 if support_side == "left" else -1.0

    # Bias the support leg under the pelvis and lift the other leg into a high-knee pose.
    motion["dof_pos"][:, index[f"{support_side}_hip_roll_joint"]] += support_sign * 0.08
    motion["dof_pos"][:, index[f"{support_side}_knee_joint"]] += 0.08
    motion["dof_pos"][:, index[f"{lift_side}_hip_pitch_joint"]] += 0.95
    motion["dof_pos"][:, index[f"{lift_side}_knee_joint"]] += 1.15
    motion["dof_pos"][:, index[f"{lift_side}_ankle_pitch_joint"]] += -0.20
    motion["dof_pos"][:, index[f"{lift_side}_hip_roll_joint"]] += -support_sign * 0.18
    motion["root_pos"][:, 2] += 0.04

    # Only the lifted leg sweeps ankle roll/pitch. The support ankle stays at
    # the standing-pose value to make the support-foot contact clean.
    apply_full_ankle_sweep(motion, t, index, limits, sides=(lift_side,), freq_scale=1.15)
    return motion


def make_ankle_only(source, frames, fps, limits, target_dof_names):
    motion = make_base(source, frames, fps, target_dof_names)
    index = {name: idx for idx, name in enumerate(motion["dof_names"])}
    t = np.arange(frames) / fps
    apply_scaled_ankle_sweep(motion, t, index, limits, sides=("left", "right"), scale=0.6, freq_scale=1.0)
    return motion


def make_arms_stand(source, frames, fps, limits, target_dof_names):
    motion = make_base(source, frames, fps, target_dof_names)
    index = {name: idx for idx, name in enumerate(motion["dof_names"])}
    t = np.arange(frames) / fps
    apply_arm_and_body_motion(motion, t, index, mode="arms_stand")
    # Moderate ankle activity, not full-range, so this file remains primarily
    # a standing/stepping whole-body motion.
    for side_id, side in enumerate(("left", "right")):
        motion["dof_pos"][:, index[f"{side}_ankle_pitch_joint"]] += 0.18 * np.sin(2.0 * np.pi * 0.8 * t + side_id * np.pi)
        motion["dof_pos"][:, index[f"{side}_ankle_roll_joint"]] += 0.10 * np.sin(2.0 * np.pi * 0.55 * t + side_id * np.pi)
    return motion


def write_legged_lab_txt(motion, output_txt):
    fps = float(motion["fps"])
    dt = 1.0 / fps
    root_pos = motion["root_pos"]
    root_rot_wxyz = motion["root_rot"][:, [3, 0, 1, 2]]
    dof = motion["dof_pos"]
    root_lin_vel = (root_pos[1:] - root_pos[:-1]) / dt
    delta = (
        R.from_quat(root_rot_wxyz[:-1, [1, 2, 3, 0]]).inv()
        * R.from_quat(root_rot_wxyz[1:, [1, 2, 3, 0]])
    )
    root_ang_vel = delta.as_rotvec() / dt
    dof_vel = (dof[1:] - dof[:-1]) / dt
    euler = R.from_quat(root_rot_wxyz[:-1, [1, 2, 3, 0]]).as_euler("XYZ", degrees=False)
    euler = np.unwrap(euler, axis=0)
    frames = np.concatenate((root_pos[:-1], euler, dof[:-1], root_lin_vel, root_ang_vel, dof_vel), axis=1)

    output_txt.parent.mkdir(parents=True, exist_ok=True)
    with output_txt.open("w") as file:
        file.write("{\n")
        file.write('"LoopMode": "Wrap",\n')
        file.write(f'"FrameDuration": {dt:.3f},\n')
        file.write('"EnableCycleOffsetPosition": true,\n')
        file.write('"EnableCycleOffsetRotation": true,\n')
        file.write('"MotionWeight": 0.5,\n\n')
        file.write('"Frames":\n[\n')
        for idx, frame in enumerate(frames):
            suffix = "\n" if idx == len(frames) - 1 else ",\n"
            file.write("  [" + ", ".join(f"{value:f}" for value in frame) + "]" + suffix)
        file.write("]\n}")


def save_motion(motion, pkl_path, txt_path):
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with pkl_path.open("wb") as file:
        pickle.dump(motion, file)
    write_legged_lab_txt(motion, txt_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, default=DEFAULT_SOURCE)
    parser.add_argument("--xml", type=pathlib.Path, default=DEFAULT_XML)
    parser.add_argument("--out_dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    source = load_motion(args.source)
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    limits = get_joint_limits(model)
    target_dof_names = get_qpos_dof_names(model)
    frames = int(round(args.duration * args.fps))
    outputs = {
        "77_02_synth_stand_arm_body_step_lens110xml": make_arms_stand(
            source, frames, args.fps, limits, target_dof_names
        ),
        "77_02_synth_left_support_right_high_knee_ankle_sweep_lens110xml": make_single_leg_high_knee(
            source, frames, args.fps, limits, target_dof_names, support_side="left"
        ),
        "77_02_synth_right_support_left_high_knee_ankle_sweep_lens110xml": make_single_leg_high_knee(
            source, frames, args.fps, limits, target_dof_names, support_side="right"
        ),
        "77_02_synth_stand_ankle_only_safe_sweep_lens110xml": make_ankle_only(
            source, frames, args.fps, limits, target_dof_names
        ),
    }
    for name, motion in outputs.items():
        motion["dof_pos"] = clip_to_limits(motion["dof_pos"], motion["dof_names"], limits, margin=1e-4)
        root_lift = lift_root_out_of_ground(model, motion)
        save_motion(motion, args.out_dir / "pkl" / f"{name}.pkl", args.out_dir / "txt" / f"{name}.txt")
        print(f"wrote {name}: frames={frames} fps={args.fps} root_lift={root_lift:.4f}")
        index = {joint: idx for idx, joint in enumerate(motion["dof_names"])}
        for joint in ("left_ankle_pitch_joint", "left_ankle_roll_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"):
            values = motion["dof_pos"][:, index[joint]]
            print(f"  {joint}: {values.min():.3f} .. {values.max():.3f}")


if __name__ == "__main__":
    main()
