"""Procedural MuJoCo terrains for LENS110 playback.

The terrain patterns follow the open-source MuJoCo Playground approach of
building reusable procedural terrain primitives.  Only MuJoCo box geoms are
added here, so the robot XML and its mesh assets remain untouched.
"""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


TERRAIN_NAMES = ("flat", "stairs", "rough", "mixed")
STAIR_RISES_M = (0.05, 0.10, 0.15, 0.20)
STAIR_TREADS_PER_LEVEL = 3
STAIR_TREAD_DEPTH = 0.28


def _add_box(
    worldbody: ET.Element,
    name: str,
    x: float,
    y: float,
    z: float,
    sx: float,
    sy: float,
    sz: float,
    rgba: str,
) -> None:
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": name,
            "type": "box",
            "pos": f"{x:.4f} {y:.4f} {z:.4f}",
            "size": f"{sx:.4f} {sy:.4f} {sz:.4f}",
            "contype": "1",
            "conaffinity": "0",
            "friction": "0.9 0.02 0.001",
            "rgba": rgba,
            # Keep terrain in the viewer's default visible geom group.  The
            # previous group=3 still collided but was hidden by the viewer.
            "group": "0",
        },
    )


def _add_stairs(worldbody: ET.Element, start_x: float, scale: float = 1.0) -> float:
    depth = STAIR_TREAD_DEPTH
    half_width = 1.8
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("stair_scale must be finite and positive")
    rises = np.repeat(STAIR_RISES_M, STAIR_TREADS_PER_LEVEL) * scale
    ascending = np.cumsum(rises)
    # Mirror the elevations; the last downward riser lands on the floor.
    heights = np.concatenate([ascending, ascending[-2::-1]])
    for index, height in enumerate(heights):
        _add_box(
            worldbody,
            f"terrain_stair_{index:02d}",
            start_x + depth * (index + 0.5),
            0.0,
            height * 0.5,
            depth * 0.5,
            half_width,
            height * 0.5,
            "0.36 0.47 0.62 1",
        )
    return start_x + depth * len(heights)


def _add_rough_patch(worldbody: ET.Element, start_x: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    x_step = 0.48
    y_step = 0.48
    patch_cols = 14
    patch_rows = 7
    for ix in range(patch_cols):
        for iy in range(patch_rows):
            if rng.random() < 0.22:
                continue
            height = float(rng.uniform(0.025, 0.11))
            sx = float(rng.uniform(0.16, 0.25))
            sy = float(rng.uniform(0.16, 0.25))
            x = start_x + ix * x_step + float(rng.uniform(-0.07, 0.07))
            y = (iy - (patch_rows - 1) * 0.5) * y_step + float(rng.uniform(-0.07, 0.07))
            _add_box(
                worldbody,
                f"terrain_rough_{ix:02d}_{iy:02d}",
                x,
                y,
                height * 0.5,
                sx,
                sy,
                height * 0.5,
                "0.48 0.42 0.30 1",
            )


def load_model_with_terrain(
    xml_path: str,
    terrain: str = "flat",
    seed: int = 7,
    stair_scale: float = 1.0,
) -> mujoco.MjModel:
    """Load a robot XML and add a procedural terrain before compilation."""
    if terrain not in TERRAIN_NAMES:
        raise ValueError(f"Unknown terrain '{terrain}', choose from {TERRAIN_NAMES}")
    if terrain == "flat":
        return mujoco.MjModel.from_xml_path(xml_path)

    xml_file = Path(xml_path).resolve()
    root = ET.parse(xml_file).getroot()
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", str(xml_file.parent / "meshes") + "/")
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"XML has no worldbody: {xml_path}")

    terrain_end_x = 1.6
    if terrain in ("stairs", "mixed"):
        terrain_end_x = _add_stairs(worldbody, start_x=1.6, scale=stair_scale)
    if terrain in ("rough", "mixed"):
        rough_start_x = 1.6 if terrain == "rough" else terrain_end_x + 0.6
        _add_rough_patch(worldbody, start_x=rough_start_x, seed=seed)

    xml_string = ET.tostring(root, encoding="unicode")
    return mujoco.MjModel.from_xml_string(xml_string)
