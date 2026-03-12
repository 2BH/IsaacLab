# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the dexsuite task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def out_of_bound(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    in_bound_range: dict[str, tuple[float, float]] = {},
) -> torch.Tensor:
    """Termination condition for the object falls out of bound.

    Args:
        env: The environment.
        asset_cfg: The object configuration. Defaults to SceneEntityCfg("object").
        in_bound_range: The range in x, y, z such that the object is considered in range
    """
    object: RigidObject = env.scene[asset_cfg.name]
    range_list = [in_bound_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=env.device)

    object_pos_local = object.data.root_pos_w - env.scene.env_origins
    outside_bounds = ((object_pos_local < ranges[:, 0]) | (object_pos_local > ranges[:, 1])).any(dim=1)
    return outside_bounds


def abnormal_robot_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    arm_joint_names: str = "xarm_joint_(1|2|3|4|5|6|7)",
) -> torch.Tensor:
    """Terminate when arm joint velocities exceed 2× their limits.

    Only arm joints are checked.  Hand (finger) joints intentionally excluded:
    contact impulses during grasping can momentarily push finger velocities past
    the 2× threshold without indicating any physics instability, which would
    cause false-positive resets of otherwise valid grasp episodes.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    arm_cfg = SceneEntityCfg(asset_cfg.name, joint_names=[arm_joint_names])
    arm_cfg.resolve(env.scene)
    arm_vel = robot.data.joint_vel[:, arm_cfg.joint_ids]
    arm_limits = robot.data.joint_vel_limits[:, arm_cfg.joint_ids]
    return (arm_vel.abs() > arm_limits * 2).any(dim=1)
