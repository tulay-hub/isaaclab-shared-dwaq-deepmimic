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
DEFAULT_XML = WORKSPACE_ROOT / "tools/retargeting/gmr_lens110/assets/lens110_21dof/mjcf/lens110_21dof.xml"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "projects/02_dance_half_body/data/processed/retargeted_actions/legacy_49_balance_fixed"


LEG_JOINT_LIMITS = {
    "left_hip_roll_joint": (-0.35, 0.22),
    "right_hip_roll_joint": (-0.35, 0.28),
    "left_hip_yaw_joint": (-0.28, 0.28),
    "right_hip_yaw_joint": (-0.28, 0.28),
    "left_ankle_roll_joint": (-0.18, 0.22),
    "right_ankle_roll_joint": (-0.14, 0.22),
}


def moving_average(values, window):
    window = int(max(1, window))
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window) / window
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def unwrap_and_center(euler):
    out = np.unwrap(euler, axis=0)
    return out - out[0]


def soft_clip(values, limit):
    if limit <= 0.0:
        return np.zeros_like(values)
    return limit * np.tanh(values / limit)


def stabilize_root(root_rot_xyzw, roll_limit, pitch_limit, smooth_window):
    euler = R.from_quat(root_rot_xyzw).as_euler("XYZ", degrees=False)
    euler = np.unwrap(euler, axis=0)
    yaw = euler[:, 2].copy()
    roll = moving_average(euler[:, 0], smooth_window)
    pitch = moving_average(euler[:, 1], smooth_window)
    euler[:, 0] = soft_clip(roll, roll_limit)
    euler[:, 1] = soft_clip(pitch, pitch_limit)
    euler[:, 2] = yaw
    return R.from_euler("XYZ", euler).as_quat()


def clip_and_smooth_dofs(dof, dof_names, smooth_window):
    fixed = dof.copy()
    name_to_index = {name: idx for idx, name in enumerate(dof_names)}
    for name, (lo, hi) in LEG_JOINT_LIMITS.items():
        if name not in name_to_index:
            continue
        idx = name_to_index[name]
        fixed[:, idx] = np.clip(fixed[:, idx], lo, hi)

    for name in (
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
    ):
        if name in name_to_index:
            idx = name_to_index[name]
            fixed[:, idx] = moving_average(fixed[:, idx], smooth_window)
    return fixed


def body_y_relative_to_pelvis(model, data, qpos, body_name):
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    return float(data.body(body_name).xpos[1] - data.body("pelvis").xpos[1])


def correct_lateral_foot_targets(model, motion, source_motion, max_delta=0.45):
    """Preserve the source motion's lateral foot placement after root stabilization.

    The 49 clips are single-leg balance motions, so the first frames must keep the
    feet outside the pelvis instead of averaging the hip roll inward.
    """
    data = mujoco.MjData(model)
    dof = motion["dof_pos"].copy()
    dof_names = motion["dof_names"]
    index = {name: idx for idx, name in enumerate(dof_names)}
    joint_body_pairs = (
        ("left_hip_roll_joint", "left_ankle_roll_link"),
        ("right_hip_roll_joint", "right_ankle_roll_link"),
    )
    source_qpos = build_qpos(source_motion)

    target_y = {}
    for _, body_name in joint_body_pairs:
        target_y[body_name] = np.asarray(
            [body_y_relative_to_pelvis(model, data, qpos, body_name) for qpos in source_qpos],
            dtype=np.float64,
        )

    for frame in range(len(dof)):
        for joint_name, body_name in joint_body_pairs:
            joint_idx = index[joint_name]
            original_value = dof[frame, joint_idx]
            lo, hi = LEG_JOINT_LIMITS[joint_name]
            lo = max(lo, original_value - max_delta)
            hi = min(hi, original_value + max_delta)
            best_value = original_value
            best_error = np.inf
            for value in np.linspace(lo, hi, 41):
                test_dof = dof[frame].copy()
                test_dof[joint_idx] = value
                qpos = np.concatenate(
                    [
                        motion["root_pos"][frame],
                        motion["root_rot"][frame, [3, 0, 1, 2]],
                        test_dof,
                    ]
                )
                error = abs(body_y_relative_to_pelvis(model, data, qpos, body_name) - target_y[body_name][frame])
                if error < best_error:
                    best_error = error
                    best_value = value
            dof[frame, joint_idx] = best_value

    motion["dof_pos"] = dof


def foot_mesh_min_z(model, qpos):
    data = mujoco.MjData(model)
    foot_body_ids = {
        model.body("left_ankle_roll_link").id,
        model.body("right_ankle_roll_link").id,
    }
    geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if model.geom_bodyid[geom_id] in foot_body_ids
        and model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
    ]
    mins = []
    for frame in qpos:
        data.qpos[:] = frame
        mujoco.mj_forward(model, data)
        frame_min = np.inf
        for geom_id in geom_ids:
            mesh_id = model.geom_dataid[geom_id]
            start = model.mesh_vertadr[mesh_id]
            count = model.mesh_vertnum[mesh_id]
            vertices = model.mesh_vert[start : start + count]
            rot = data.geom_xmat[geom_id].reshape(3, 3)
            world_vertices = data.geom_xpos[geom_id] + vertices @ rot.T
            frame_min = min(frame_min, float(world_vertices[:, 2].min()))
        mins.append(frame_min)
    return np.asarray(mins)


def build_qpos(motion):
    root_rot_wxyz = motion["root_rot"][:, [3, 0, 1, 2]]
    return np.concatenate([motion["root_pos"], root_rot_wxyz, motion["dof_pos"]], axis=1)


def write_legged_lab_txt(motion, output_txt):
    fps = float(motion.get("fps", 30.0))
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


def process_file(input_pkl, output_pkl, output_txt, xml_path, args):
    with input_pkl.open("rb") as file:
        source_motion = pickle.load(file)

    motion = dict(source_motion)
    source_motion = dict(source_motion)
    source_motion["root_pos"] = np.asarray(source_motion["root_pos"]).copy()
    source_motion["root_rot"] = np.asarray(source_motion["root_rot"]).copy()
    source_motion["dof_pos"] = np.asarray(source_motion["dof_pos"]).copy()
    motion["root_pos"] = np.asarray(motion["root_pos"]).copy()
    motion["root_rot"] = stabilize_root(
        np.asarray(motion["root_rot"]),
        np.deg2rad(args.root_roll_limit_deg),
        np.deg2rad(args.root_pitch_limit_deg),
        args.root_rot_window,
    )
    motion["root_rot_format"] = "xyzw"
    if args.preserve_leg_dofs:
        motion["dof_pos"] = np.asarray(motion["dof_pos"]).copy()
    else:
        motion["dof_pos"] = clip_and_smooth_dofs(np.asarray(motion["dof_pos"]), motion["dof_names"], args.dof_window)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    if not args.preserve_leg_dofs:
        correct_lateral_foot_targets(model, motion, source_motion, args.max_lateral_correction)
    qpos = build_qpos(motion)
    foot_z = foot_mesh_min_z(model, qpos)
    if args.raise_only_ground:
        lift = moving_average(args.ground_clearance - foot_z, args.ground_window)
        lift = np.clip(lift, 0.0, args.max_ground_lift)
        motion["root_pos"][:, 2] += lift
    elif not args.preserve_root_z:
        offsets = moving_average(foot_z - args.ground_clearance, args.ground_window)
        offsets = np.minimum(offsets, args.max_ground_lowering)
        motion["root_pos"][:, 2] -= offsets
    motion["root_pos"][:, 2] += args.root_z_offset

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl.open("wb") as file:
        pickle.dump(motion, file)
    write_legged_lab_txt(motion, output_txt)
    print(
        f"{input_pkl.name}: root_z={motion['root_pos'][:,2].min():.3f}/{motion['root_pos'][:,2].mean():.3f}/"
        f"{motion['root_pos'][:,2].max():.3f}, foot_z_before={foot_z.min():.3f}/{np.percentile(foot_z, 5):.3f}/"
        f"{foot_z.mean():.3f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=pathlib.Path, required=True, help="Local directory containing 49_*_stageii_balance_one_leg.pkl files.")
    parser.add_argument("--output_dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--xml", type=pathlib.Path, default=DEFAULT_XML)
    parser.add_argument("--root_roll_limit_deg", type=float, default=12.0)
    parser.add_argument("--root_pitch_limit_deg", type=float, default=10.0)
    parser.add_argument("--root_rot_window", type=int, default=9)
    parser.add_argument("--root_z_offset", type=float, default=0.0)
    parser.add_argument("--dof_window", type=int, default=7)
    parser.add_argument("--preserve_root_z", action="store_true", default=True)
    parser.add_argument("--raise_only_ground", action="store_true", default=True)
    parser.add_argument("--max_ground_lift", type=float, default=0.08)
    parser.add_argument("--preserve_leg_dofs", action="store_true", default=False)
    parser.add_argument("--max_lateral_correction", type=float, default=0.55)
    parser.add_argument("--ground_window", type=int, default=41)
    parser.add_argument("--ground_clearance", type=float, default=0.015)
    parser.add_argument("--max_ground_lowering", type=float, default=0.12)
    args = parser.parse_args()

    for input_pkl in sorted(args.input_dir.glob("49_*_stageii_balance_one_leg.pkl")):
        process_file(
            input_pkl,
            args.output_dir / "pkl" / input_pkl.name,
            args.output_dir / "txt" / input_pkl.with_suffix(".txt").name,
            args.xml,
            args,
        )


if __name__ == "__main__":
    main()
