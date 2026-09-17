# BeyondMimic T800/PM01 Example

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

[[Website]](https://beyondmimic.github.io/)
[[Arxiv]](https://arxiv.org/abs/2508.08241)
[[Video]](https://youtu.be/RS_MtKVIAzY)

## Important Notice

This repository is based on the BeyondMimic framework. We would like to express our sincere appreciation to the
BeyondMimic authors and contributors for their excellent work and for making their research and engineering efforts
available to the community.

This repository is provided as an example package for the boxing competition, with the primary purpose of helping
participants use the `T800` assets more conveniently and reducing the practical difficulty of sim-to-real deployment in
their development workflow.

If any content in this repository is believed to infringe third-party rights or otherwise requires adjustment, please
contact us promptly. We will review the matter carefully and make any necessary updates or corrections as quickly as
possible.

## Overview

This repository is based on `whole_body_tracking` / BeyondMimic and has been adapted for the `T800` and `PM01`
humanoid platforms used in this example workflow.

The current maintained robot targets in this repository are `T800` and `PM01`. The intended training workflow in this
repository uses local motion files stored under `data/` in `.npz` format.

## Commercial Use

This repository is being prepared for commercial use, but the repository license alone does not clear all third-party
rights.

Important: robot assets, NVIDIA Isaac ecosystem components, and optional cloud services such as `wandb` remain subject
to their own licenses, terms, or internal ownership approvals.

## Deployment

For `sim-to-sim` and `sim-to-real` deployment in this workflow, please refer to the EngineAI deployment framework:

- [engineai_robotics_native_sdk](https://github.com/engineai-robotics/engineai_robotics_native_sdk/tree/main)

The original BeyondMimic deployment reference is also kept here for attribution and compatibility context:

- [motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller)
- [mjlab alternative implementation](https://github.com/mujocolab/mjlab/blob/main/src/mjlab/tasks/tracking/tracking_env_cfg.py)

## Demo Videos

The following local media files are included in this package:

- Sim-to-real deployment demo: [deployment_victory.mp4](docs/media/deployment_victory.mp4)
- Sim-to-sim demo: [sim2sim_demo.mp4](docs/media/sim2sim_demo.mp4)

## Installation

- Install Isaac Lab v2.1.0 by following
  the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend
  using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone this repository separately from the Isaac Lab installation (i.e., outside the `IsaacLab` directory):

```bash
# Option 1: SSH
git clone git@github.com:HybridRobotics/whole_body_tracking.git

# Option 2: HTTPS
git clone https://github.com/HybridRobotics/whole_body_tracking.git
```

- Install the extension package

```bash
# Enter the repository
cd whole_body_tracking
```

- Using a Python interpreter that has Isaac Lab installed, install the library

```bash
python -m pip install -e source/whole_body_tracking
```

## Motion Tracking

### Local Motion Files

This repository trains against local `.npz` motion files placed under `data/`.
The recommended workflow is:

1. Convert or place motion files into `data/`
2. Replay the local motion file
3. Train with `--motion_file`

- Convert a local T800 `.npy` motion into a training-ready `.npz` file at 50 Hz:

```bash
python scripts/npy_to_npz.py \
  -i /path/to/motion.npy \
  -o data/{motion_name}_50hz.npz \
  --fps 50 \
  --input_fps 30 \
  --engineai_lab /path/to/engineaimuaythailab
```

- Replay a local motion file in Isaac Sim before training:

```bash
python scripts/replay_npz.py --robot=t800 --input_file data/{motion_name}_50hz.npz
```

- Debugging
    - `scripts/npy_to_npz.py` uses the repository-standard DFS body order for `T800`.
    - If replay looks wrong, regenerate the source motion rather than changing body order locally.

### Policy Training

- Train policy by the following command with a local motion file:

```bash
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-T800-v0 \
  --motion_file data/victory_50hz.npz \
  --num_envs 4096 \
  --headless
```

### Policy Evaluation

- Play or validate with the following command:

```bash
python scripts/rsl_rl/play.py \
  --task=Tracking-Flat-T800-v0 \
  --motion_file data/victory_50hz.npz \
  --num_envs 1
```

## Code Structure

Below is an overview of the code structure for this repository:

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**
  This directory contains the atomic functions to define the MDP for BeyondMimic. Below is a breakdown of the functions:

    - **`commands.py`**
      Command library to compute relevant variables from the reference motion, current robot state, and error
      computations. This includes pose and velocity error calculation, initial state randomization, and adaptive
      sampling.

    - **`rewards.py`**
      Implements the DeepMimic reward functions and smoothing terms.

    - **`events.py`**
      Implements domain randomization terms.

    - **`observations.py`**
      Implements observation terms for motion tracking and data collection.

    - **`terminations.py`**
      Implements early terminations and timeouts.

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**
  Contains the environment (MDP) hyperparameters configuration for the tracking task.

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/agents/rsl_rl_ppo_cfg.py`**
  Contains the PPO hyperparameters for the T800 tracking task.

- **`source/whole_body_tracking/whole_body_tracking/robots`**
  Contains robot-specific settings, including armature parameters, joint stiffness/damping calculation, and action scale
  calculation.

- **`scripts`**
  Includes utility scripts for preprocessing motion data, training policies, and evaluating trained policies.

This structure is designed to ensure modularity and ease of navigation for developers expanding the project.
