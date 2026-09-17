"""MDP functions for DreamWAQ (DWAQ) locomotion.

Re-exports all velocity locomotion MDP terms plus DWAQ-specific rewards
and observation functions.
"""

from legged_lab.tasks.locomotion.velocity.mdp import *  # noqa: F401, F403

from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
