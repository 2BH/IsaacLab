# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import grasp_env_cfg as grasp
from . import mdp


_TRAIN_SET_SUCCESS_WEIGHT: float = 50.0
_TRAIN_SET_POSITION_TRACKING_WEIGHT: float = 50.0
_TRAIN_SET_ACTION_L2_WEIGHT: float = -0.0005
_TRAIN_SET_ACTION_RATE_L2_WEIGHT: float = -0.0005


def _rigidify_train_set_urdf(src_urdf_path: Path, dst_urdf_path: Path) -> None:
    """Convert train-set URDFs into a rigid-body equivalent by fixing the internal hinge."""
    tree = ET.parse(src_urdf_path)
    root = tree.getroot()

    # Preserve mesh resolution after relocating the rewritten URDF into /tmp cache.
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if not filename or filename.startswith(("package://", "/")):
            continue
        mesh.attrib["filename"] = str((src_urdf_path.parent / filename).resolve())

    for joint in root.findall("joint"):
        if joint.attrib.get("name") == "rotation":
            joint.attrib["type"] = "fixed"
            # Keep the kinematic tree but remove dynamic/limit fields that are not meaningful for fixed joints.
            for tag in ("limit", "dynamics"):
                for child in list(joint.findall(tag)):
                    joint.remove(child)

    dst_urdf_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dst_urdf_path, encoding="utf-8", xml_declaration=True)


def _resolve_train_set_urdf_paths() -> list[Path]:
    train_set_dir = Path(__file__).resolve().parent / "assets" / "objects" / "train_set"
    source_urdf_paths = sorted(train_set_dir.glob("*/*.urdf"))
    source_urdf_paths = [p for p in source_urdf_paths if not p.name.endswith("_fixed_base.urdf")]
    if not source_urdf_paths:
        raise RuntimeError(f"No URDF files found under: {train_set_dir}")

    rigid_urdf_cache_dir = Path("/tmp/IsaacLab/dexgrasp_train_set_rigid_urdf_cache_v3")
    rigid_urdf_paths: list[Path] = []
    for src_urdf_path in source_urdf_paths:
        dst_urdf_path = rigid_urdf_cache_dir / src_urdf_path.relative_to(train_set_dir)
        if (not dst_urdf_path.exists()) or (dst_urdf_path.stat().st_mtime < src_urdf_path.stat().st_mtime):
            _rigidify_train_set_urdf(src_urdf_path, dst_urdf_path)
        rigid_urdf_paths.append(dst_urdf_path)
    return rigid_urdf_paths


_TRAIN_SET_URDF_PATHS = _resolve_train_set_urdf_paths()

def _make_usd_safe_name(name: str) -> str:
    # USD prim names cannot start with a digit. The URDF importer will auto-fix such names and emit warnings.
    # We still use a stable, "safe" USD file name for caching.
    return f"a_{name}" if name and name[0].isdigit() else name


def _make_train_set_assets_cfg() -> list[sim_utils.UrdfFileCfg]:
    assets_cfg: list[sim_utils.UrdfFileCfg] = []
    for urdf_path in _TRAIN_SET_URDF_PATHS:
        # Use a stable USD output directory so URDF->USD conversion is cached (avoids repeated conversion + warnings).
        safe_stem = _make_usd_safe_name(urdf_path.stem)
        assets_cfg.append(
            sim_utils.UrdfFileCfg(
                asset_path=str(urdf_path),
                # Separate cache namespace so rigidified URDFs always regenerate USDs
                # instead of reusing older articulation-with-revolute conversions.
                usd_dir=f"/tmp/IsaacLab/dexgrasp_train_set_usd_cache_rigid_v3/{safe_stem}",
                usd_file_name=f"{safe_stem}_rigid_v3.usd",
                fix_base=False,
                merge_fixed_joints=True,
                joint_drive=None,
            )
        )
    return assets_cfg

@configclass
class TrainSetSceneCfg(grasp.SceneCfg):
    """Dexgrasp scene with random train-set objects (URDF)."""

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=_make_train_set_assets_cfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                max_depenetration_velocity=1000.0,
            ),
            # NOTE:
            # Many URDF-converted assets end up with instanced collision prims (instanceable meshes).
            # In that case, `modify_collision_properties` can't apply overrides at runtime and emits warnings.
            # We therefore avoid overriding collision properties here (behavior stays the same as before, but without warnings).
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.45, 0.1, 0.25),
        ),
    )

    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.CuboidCfg(
            size=(1.4, 1.5, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # trick: we let visualizer's color to show the table with success coloring
            visible=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.45, 0.0, 0.235), rot=(1.0, 0.0, 0.0, 0.0)),
    )


@configclass
class TrainSetEventCfg(grasp.EventCfg):
    # Use task-local scale randomizer to support URDF-based train-set rigid objects at prestartup.
    randomize_object_scale = EventTerm(
        func=mdp.randomize_rigid_body_scale_safe,
        mode="prestartup",
        params={"scale_range": (0.75, 1.5), "asset_cfg": SceneEntityCfg("object")},
    )

    # Override the reset distribution: the base task uses offsets suited for the default table/object placement.
    # Here we keep the object closer to the robot and centered on the resized table.
    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.30, 0.0],
                "y": [-0.15, 0.15],
                "z": [0.05, 0.10],
                "roll": [-3.14, 3.14],
                "pitch": [-3.14, 3.14],
                "yaw": [-3.14, 3.14],
            },
            "velocity_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

@configclass
class TrainSetRewardsCfg(grasp.RewardsCfg):
    """Rewards for train-set tasks (same terms, higher key weights)."""

    def __post_init__(self):
        super().__post_init__()
        self.position_tracking.weight = _TRAIN_SET_POSITION_TRACKING_WEIGHT
        self.success.weight = _TRAIN_SET_SUCCESS_WEIGHT
        # Keep action terms as penalties. Treat the configured values as magnitudes.
        self.action_l2.weight = _TRAIN_SET_ACTION_L2_WEIGHT
        self.action_rate_l2.weight = _TRAIN_SET_ACTION_RATE_L2_WEIGHT


@configclass
class DexgraspReorientTrainSetEnvCfg(grasp.DexgraspReorientEnvCfg):
    """Dexgrasp Reorient task with random train-set objects."""

    scene: TrainSetSceneCfg = TrainSetSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=False)
    rewards: TrainSetRewardsCfg = TrainSetRewardsCfg()
    events: TrainSetEventCfg = TrainSetEventCfg()

    def __post_init__(self):
        super().__post_init__()
        if self.curriculum is not None:
            self.curriculum.adr.params["promotion_only"] = False


@configclass
class DexgraspLiftTrainSetEnvCfg(grasp.DexgraspLiftEnvCfg):
    """Dexgrasp Lift task with random train-set objects."""

    scene: TrainSetSceneCfg = TrainSetSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=False)
    rewards: TrainSetRewardsCfg = TrainSetRewardsCfg()
    events: TrainSetEventCfg = TrainSetEventCfg()

    def __post_init__(self):
        super().__post_init__()
        if self.curriculum is not None:
            self.curriculum.adr.params["promotion_only"] = True


@configclass
class DexgraspReorientTrainSetEnvCfg_PLAY(grasp.DexgraspReorientEnvCfg_PLAY):
    """Dexgrasp Reorient evaluation task with random train-set objects."""

    scene: TrainSetSceneCfg = TrainSetSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=False)
    rewards: TrainSetRewardsCfg = TrainSetRewardsCfg()
    events: TrainSetEventCfg = TrainSetEventCfg()

    def __post_init__(self):
        super().__post_init__()
        if self.curriculum is not None:
            self.curriculum.adr.params["promotion_only"] = True


@configclass
class DexgraspLiftTrainSetEnvCfg_PLAY(grasp.DexgraspLiftEnvCfg_PLAY):
    """Dexgrasp Lift evaluation task with random train-set objects."""

    scene: TrainSetSceneCfg = TrainSetSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=False)
    rewards: TrainSetRewardsCfg = TrainSetRewardsCfg()
    events: TrainSetEventCfg = TrainSetEventCfg()

    def __post_init__(self):
        super().__post_init__()
        if self.curriculum is not None:
            self.curriculum.adr.params["promotion_only"] = True


@configclass
class DexgraspLiftTrainSetEnvCfg_PLAY_Mild(grasp.DexgraspLiftEnvCfg_PLAY_Mild):
    """Dexgrasp Lift mild-eval task with random train-set objects."""

    scene: TrainSetSceneCfg = TrainSetSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=False)
    rewards: TrainSetRewardsCfg = TrainSetRewardsCfg()
    events: TrainSetEventCfg = TrainSetEventCfg()

    def __post_init__(self):
        super().__post_init__()
        if self.curriculum is not None:
            self.curriculum.adr.params["promotion_only"] = True
