# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from pxr import Gf, Sdf, UsdGeom, Vt

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.utils.stage import get_current_stage

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_rigid_body_scale_safe(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    scale_range: tuple[float, float] | dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
    relative_child_path: str | None = None,
):
    """Task-local variant of rigid-body scale randomization with robust USD attribute writes."""
    if env.sim.is_playing():
        raise RuntimeError(
            "Randomizing scale while simulation is running leads to unpredictable behaviors."
            " Please ensure this event is called before simulation starts."
        )

    asset: RigidObject = env.scene[asset_cfg.name]
    if isinstance(asset, Articulation):
        raise ValueError("Scaling articulations is not supported.")

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    stage = get_current_stage()
    prim_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)

    if isinstance(scale_range, dict):
        ranges = torch.tensor([scale_range.get(key, (1.0, 1.0)) for key in ["x", "y", "z"]], device="cpu")
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu")
    else:
        rand_samples = math_utils.sample_uniform(*scale_range, (len(env_ids), 1), device="cpu").repeat(1, 3)
    rand_samples = rand_samples.tolist()

    if relative_child_path is None:
        relative_child_path = ""
    elif not relative_child_path.startswith("/"):
        relative_child_path = "/" + relative_child_path

    with Sdf.ChangeBlock():
        for i, env_id in enumerate(env_ids):
            prim_path = prim_paths[env_id] + relative_child_path
            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)

            # Note:
            # - GetAttributeAtPath expects a full Sdf path (e.g. /Prim.attr).
            # - Sdf.AttributeSpec expects an attribute name (e.g. xformOp:scale).
            scale_attr_name = "xformOp:scale"
            scale_attr_path = prim_path + ".xformOp:scale"
            scale_spec = prim_spec.GetAttributeAtPath(scale_attr_path)
            has_scale_attr = scale_spec is not None
            if not has_scale_attr:
                scale_spec = Sdf.AttributeSpec(prim_spec, scale_attr_name, Sdf.ValueTypeNames.Double3)

            scale_spec.default = Gf.Vec3f(*rand_samples[i])

            if not has_scale_attr:
                op_order_name = UsdGeom.Tokens.xformOpOrder
                op_order_attr_path = prim_path + ".xformOpOrder"
                op_order_spec = prim_spec.GetAttributeAtPath(op_order_attr_path)
                if op_order_spec is None:
                    op_order_spec = Sdf.AttributeSpec(prim_spec, op_order_name, Sdf.ValueTypeNames.TokenArray)
                op_order_spec.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])
