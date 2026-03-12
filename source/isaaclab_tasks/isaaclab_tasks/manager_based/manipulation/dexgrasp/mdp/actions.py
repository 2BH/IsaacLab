# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.joint_actions import RelativeJointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg as BaseRelativeJointPositionActionCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class RelativeJointPositionToLimitsAction(RelativeJointPositionAction):
    """Relative joint position action with target clamped to soft joint limits."""

    cfg: "RelativeJointPositionToLimitsActionCfg"

    def __init__(self, cfg: "RelativeJointPositionToLimitsActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def apply_actions(self):
        # add current joint positions to the processed actions
        current_actions = self.processed_actions + self._asset.data.joint_pos[:, self._joint_ids]
        # clamp to joint limits (soft limits already include any configured soft limit factor)
        joint_pos_limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        if self.cfg.limit_margin != 0.0:
            lower = joint_pos_limits[..., 0] + self.cfg.limit_margin
            upper = joint_pos_limits[..., 1] - self.cfg.limit_margin
        else:
            lower = joint_pos_limits[..., 0]
            upper = joint_pos_limits[..., 1]
        current_actions = torch.clamp(current_actions, min=lower, max=upper)
        # set position targets
        self._asset.set_joint_position_target(current_actions, joint_ids=self._joint_ids)


@configclass
class RelativeJointPositionToLimitsActionCfg(BaseRelativeJointPositionActionCfg):
    """Configuration for relative joint position action clamped to limits."""

    class_type: type[ActionTerm] = RelativeJointPositionToLimitsAction
    limit_margin: float = 0.0
    """Margin to shrink joint limits before clamping (radians). Defaults to 0.0."""
