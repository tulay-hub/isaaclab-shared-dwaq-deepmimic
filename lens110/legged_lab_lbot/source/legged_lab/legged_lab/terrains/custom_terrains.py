"""Procedural LENS110 terrains for Isaac Lab training."""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.terrains.trimesh.mesh_terrains_cfg import (
    MeshInvertedPyramidStairsTerrainCfg,
    MeshPyramidStairsTerrainCfg,
)
from isaaclab.terrains.trimesh.utils import make_border
from isaaclab.utils import configclass


GRID_HORIZONTAL_SCALE = 0.1
GRID_CELL_SIZE = 0.4
LENS110_PATH_START_X = 2.0
LENS110_PATH_WIDTH_CELLS = 2


def _append_segment(path: list[tuple[int, int]], start: tuple[int, int], end: tuple[int, int]) -> None:
    """Append a grid-aligned segment without duplicating its first cell."""
    x0, y0 = start
    x1, y1 = end
    if x0 != x1 and y0 != y1:
        raise ValueError(f"Terrain path segment must be axis-aligned: {start} -> {end}")
    if not path:
        path.append(start)
    step_x = int(np.sign(x1 - x0))
    step_y = int(np.sign(y1 - y0))
    x, y = x0, y0
    while (x, y) != end:
        x += step_x
        y += step_y
        path.append((x, y))


def _build_rectangular_spiral(num_cells_x: int, num_cells_y: int, start_cell: int) -> list[tuple[int, int]]:
    """Build one continuous nested-rectangle path shaped like ``回``.

    The entry starts at the flat spawn area and winds around two nested
    rectangles.  The one-cell gap between turns keeps the path connected
    without crossing itself, which makes the height assignment unambiguous.
    """
    left = max(start_cell, 5)
    right = num_cells_x - 4
    bottom = 2
    top = num_cells_y - 3
    gap = 3
    if right - left < 2 * gap + 6 or top - bottom < 2 * gap + 2:
        raise ValueError("Terrain patch is too small for the rectangular spiral")

    middle = (bottom + top) // 2
    waypoints = [
        (left, middle),
        (left, bottom),
        (right, bottom),
        (right, top),
        (left + gap, top),
        (left + gap, bottom + gap),
        (right - gap, bottom + gap),
        (right - gap, top - gap),
        (left + 2 * gap, top - gap),
    ]

    path: list[tuple[int, int]] = []
    for start, end in zip(waypoints, waypoints[1:]):
        _append_segment(path, start, end)
    return path


def _random_path_levels(
    count: int, difficulty: float, rng: np.random.Generator, vertical_scale: float
) -> np.ndarray:
    """Create random, bounded platform heights along a connected path."""
    if count <= 0:
        return np.zeros(0, dtype=np.int16)

    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    # Curriculum range: easy rows are 5 cm terrain, late rows reach 25 cm.
    max_surface = int(round((0.05 + 0.25 * difficulty) / vertical_scale))
    anchor_spacing = max(6, min(12, count // 8))
    anchor_positions = list(range(0, count, anchor_spacing))
    if anchor_positions[-1] != count - 1:
        anchor_positions.append(count - 1)

    anchors = rng.integers(0, max_surface + 1, size=len(anchor_positions), dtype=np.int32)
    anchors[0] = 0
    anchors[-1] = 0
    levels = np.zeros(count, dtype=np.float64)
    for index, start in enumerate(anchor_positions[:-1]):
        end = anchor_positions[index + 1]
        levels[start:end] = np.linspace(anchors[index], anchors[index + 1], end - start, endpoint=False)
    levels[anchor_positions[-1]] = anchors[-1]

    # Quantize only after interpolation so every square receives one stable
    # height-field level and no floating point seam is introduced.
    return np.clip(np.rint(levels), 0, max_surface).astype(np.int16)


def lens110_random_loop_terrain(
    difficulty: float, cfg: SubTerrainBaseCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a connected random-height rectangular ``回`` terrain.

    The flat spawn area leads into a nested rectangular path made from square
    cells.  Heights vary randomly along the path, while every neighboring path
    cell shares an edge.  Cells outside the path remain at the base floor so
    the arrangement is visible in Isaac Sim.
    """
    size_x, size_y = cfg.size
    horizontal_scale = GRID_HORIZONTAL_SCALE
    vertical_scale = 0.005
    cell_pixels = int(round(GRID_CELL_SIZE / horizontal_scale))
    pixels_x = int(size_x / horizontal_scale) + 1
    pixels_y = int(size_y / horizontal_scale) + 1
    num_cells_x = pixels_x // cell_pixels
    num_cells_y = pixels_y // cell_pixels
    start_cell = int(round(LENS110_PATH_START_X / GRID_CELL_SIZE))

    rng = np.random.default_rng(int(getattr(cfg, "seed", 7) or 7) + int(float(difficulty) * 10000))
    path = _build_rectangular_spiral(num_cells_x, num_cells_y, start_cell)
    levels = _random_path_levels(len(path), difficulty, rng, vertical_scale)

    heights = np.zeros((pixels_x, pixels_y), dtype=np.int16)
    radius = LENS110_PATH_WIDTH_CELLS // 2
    for (cell_x, cell_y), level in zip(path, levels):
        for offset_x in range(-radius, radius):
            for offset_y in range(-radius, radius):
                x0 = max(0, cell_x + offset_x) * cell_pixels
                y0 = max(0, cell_y + offset_y) * cell_pixels
                x1 = min((cell_x + offset_x + 1) * cell_pixels + 1, pixels_x)
                y1 = min((cell_y + offset_y + 1) * cell_pixels + 1, pixels_y)
                if x0 < x1 and y0 < y1:
                    heights[x0:x1, y0:y1] = level

    vertices, triangles = convert_height_field_to_mesh(
        heights,
        horizontal_scale,
        vertical_scale,
        slope_threshold=None,
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
    origin = np.array([LENS110_PATH_START_X, size_y / 2.0, 0.0], dtype=np.float64)
    return [mesh], origin


@configclass
class Lens110RandomLoopTerrainCfg(SubTerrainBaseCfg):
    function = lens110_random_loop_terrain


# Keep the old symbol import-compatible for local scripts that may have cached
# the connected-grid configuration name.
Lens110ConnectedRandomGridTerrainCfg = Lens110RandomLoopTerrainCfg


STAIR_HEIGHT_QUANTIZATION = 0.005


def _sample_random_stair_heights(
    difficulty: float,
    cfg: MeshPyramidStairsTerrainCfg,
    count: int,
    inverted: bool,
) -> np.ndarray:
    """Sample one independent riser height for every pyramid ring.

    The lower bound is always 5 cm.  The curriculum controls the upper bound,
    reaching 25 cm only at the hardest row.  A fixed seed makes the generated
    scene reproducible while still giving different heights around one stair.
    """
    lower, upper = (float(value) for value in cfg.step_height_range)
    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    upper_at_level = lower + difficulty * (upper - lower)
    seed = 42 if getattr(cfg, "seed", None) is None else int(cfg.seed)
    seed += int(round(difficulty * 1_000_003))
    seed += 0x5EED if inverted else 0
    rng = np.random.default_rng(seed)
    heights = rng.uniform(lower, max(lower, upper_at_level), size=count)
    heights = np.round(heights / STAIR_HEIGHT_QUANTIZATION) * STAIR_HEIGHT_QUANTIZATION
    return np.maximum(heights, STAIR_HEIGHT_QUANTIZATION)


def _make_stair_box(
    dimensions: tuple[float, float, float], position: tuple[float, float, float]
) -> trimesh.Trimesh:
    return trimesh.creation.box(dimensions, trimesh.transformations.translation_matrix(position))


def _random_pyramid_stairs_terrain(
    difficulty: float,
    cfg: MeshPyramidStairsTerrainCfg,
    inverted: bool,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a pyramid stair terrain with independent random ring heights."""
    num_steps_x = (cfg.size[0] - 2 * cfg.border_width - cfg.platform_width) // (2 * cfg.step_width) + 1
    num_steps_y = (cfg.size[1] - 2 * cfg.border_width - cfg.platform_width) // (2 * cfg.step_width) + 1
    num_steps = int(min(num_steps_x, num_steps_y))
    step_heights = _sample_random_stair_heights(difficulty, cfg, num_steps + 1, inverted)
    cumulative_heights = np.cumsum(step_heights)

    meshes: list[trimesh.Trimesh] = []
    if cfg.border_width > 0.0 and not cfg.holes:
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -step_heights[0] / 2.0]
        border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        meshes.extend(make_border(cfg.size, border_inner_size, step_heights[0], border_center))

    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    terrain_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)

    if inverted:
        base_z = -float(cumulative_heights[-1])
    else:
        base_z = -float(step_heights[0])

    for ring_index in range(num_steps):
        if cfg.holes:
            box_size = (cfg.platform_width, cfg.platform_width)
        else:
            box_size = (
                terrain_size[0] - 2 * ring_index * cfg.step_width,
                terrain_size[1] - 2 * ring_index * cfg.step_width,
            )

        if inverted:
            top_z = -float(cumulative_heights[ring_index])
        else:
            top_z = float(cumulative_heights[ring_index])
        box_height = top_z - base_z
        box_z = (top_z + base_z) / 2.0
        box_offset = (ring_index + 0.5) * cfg.step_width

        box_dims = (box_size[0], cfg.step_width, box_height)
        top_position = (
            terrain_center[0],
            terrain_center[1] + terrain_size[1] / 2.0 - box_offset,
            box_z,
        )
        bottom_position = (
            terrain_center[0],
            terrain_center[1] - terrain_size[1] / 2.0 + box_offset,
            box_z,
        )
        meshes.extend([_make_stair_box(box_dims, top_position), _make_stair_box(box_dims, bottom_position)])

        if cfg.holes:
            side_width = box_size[1]
        else:
            side_width = box_size[1] - 2 * cfg.step_width
        side_dims = (cfg.step_width, side_width, box_height)
        right_position = (
            terrain_center[0] + terrain_size[0] / 2.0 - box_offset,
            terrain_center[1],
            box_z,
        )
        left_position = (
            terrain_center[0] - terrain_size[0] / 2.0 + box_offset,
            terrain_center[1],
            box_z,
        )
        meshes.extend([_make_stair_box(side_dims, right_position), _make_stair_box(side_dims, left_position)])

    if inverted:
        center_top = -float(cumulative_heights[-1])
        center_height = float(step_heights[-1])
        center_z = center_top - center_height / 2.0
        origin_z = center_top
    else:
        center_top = float(cumulative_heights[-1])
        center_height = center_top - base_z
        center_z = (center_top + base_z) / 2.0
        origin_z = center_top

    center_dims = (
        terrain_size[0] - 2 * num_steps * cfg.step_width,
        terrain_size[1] - 2 * num_steps * cfg.step_width,
        center_height,
    )
    meshes.append(_make_stair_box(center_dims, (terrain_center[0], terrain_center[1], center_z)))
    origin = np.array([terrain_center[0], terrain_center[1], origin_z])
    return meshes, origin


def lens110_random_pyramid_stairs_terrain(
    difficulty: float, cfg: MeshPyramidStairsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    return _random_pyramid_stairs_terrain(difficulty, cfg, inverted=False)


def lens110_random_inverted_pyramid_stairs_terrain(
    difficulty: float, cfg: MeshInvertedPyramidStairsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    return _random_pyramid_stairs_terrain(difficulty, cfg, inverted=True)


@configclass
class Lens110RandomPyramidStairsTerrainCfg(MeshPyramidStairsTerrainCfg):
    function = lens110_random_pyramid_stairs_terrain


@configclass
class Lens110RandomInvertedPyramidStairsTerrainCfg(MeshInvertedPyramidStairsTerrainCfg):
    function = lens110_random_inverted_pyramid_stairs_terrain
