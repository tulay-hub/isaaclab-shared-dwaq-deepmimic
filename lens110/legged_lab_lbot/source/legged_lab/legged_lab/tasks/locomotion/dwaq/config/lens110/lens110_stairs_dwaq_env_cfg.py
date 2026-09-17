"""LENS110 DWAQ stair-training configuration.

The stair project is intentionally a separate task entry point even though it
shares the DWAQ actor, VAE and reward implementation with flat walking.  The
terrain generator is the grouped random path in ``legged_lab.terrains`` and
the curriculum advances only after two successful full-episode traversals.
"""

from __future__ import annotations

import copy

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from legged_lab.terrains import LENS110_RANDOM_STAIRS_ROUGH_TERRAINS_CFG
from legged_lab.tasks.locomotion.dwaq.config.lens110.lens110_dwaq_env_cfg import (
    Lens110DwaqEnvCfg,
)
from legged_lab.tasks.locomotion.dwaq.mdp.curriculums import terrain_levels_completed


@configclass
class Lens110StairsDwaqEnvCfg(Lens110DwaqEnvCfg):
    """DWAQ training on randomized 5--30 cm grouped stair paths."""

    def __post_init__(self):
        super().__post_init__()

        # Do not mutate the module-level terrain singleton used by flat DWAQ.
        self.scene.terrain.terrain_generator = copy.deepcopy(LENS110_RANDOM_STAIRS_ROUGH_TERRAINS_CFG)
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.terrain_generator.curriculum = True
        self.curriculum.terrain_levels = CurrTerm(func=terrain_levels_completed)


@configclass
class Lens110StairsDwaqEnvCfg_PLAY(Lens110StairsDwaqEnvCfg):
    """PLAY configuration for a fixed randomized stair patch."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0
        self.curriculum.terrain_levels = None
        self.scene.terrain.terrain_generator.curriculum = False
        self.observations.policy.enable_corruption = False
        self.observations.obs_history.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
