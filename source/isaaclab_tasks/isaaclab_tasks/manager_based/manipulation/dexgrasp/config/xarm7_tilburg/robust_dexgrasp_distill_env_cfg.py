# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Distillation environment config for teacher→student transfer.

The teacher receives the full privileged observation (496D) including the
ground-truth object point cloud, while the student sees only real-robot
proprioception (92D) plus a depth image (1×120×160) from a TiledCamera.

Student proprioception breakdown (92D):
  - joint_pos:            23D  (7 arm + 16 hand joint positions)
  - joint_vel:            23D  (7 arm + 16 hand joint velocities)
  - joint_target_error:   23D  (target – current for all 23 joints)
  - last_action:          23D  (previous action sent to actuators)

  Excluded (not available on real hardware or redundant with joint_pos):
  - affordance_contact / affordance_impulse: require sim contact sensors
  - hand_center_w / wrist_euler: fully determined by joint_pos via FK
  - joint_height / arm_height: removed — FK-computable but adds little
    beyond joint_pos for the CNN-based student; DEXTRAH omits them too.

Teacher observation breakdown (496D):
  - joint_pos:            23D
  - joint_target_error:   23D
  - affordance_contact:   17D
  - affordance_impulse:   17D
  - joint_height:         17D
  - arm_height:            6D
  - hand_center_w:         3D
  - wrist_euler:           3D
  - wrist_euler_delta:     3D  (wrist orientation change since reset)
  - object_point_cloud:  384D  (128 surface points × 3, body-frame, flattened)
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from ... import mdp
from .robust_dexgrasp_xarm7_tilburg_env_cfg import (
    DexgraspXArm7TilburgRobustTrainSetEnvCfg,
    RobustDexgraspTrainSetSceneCfg,
    RobustDexgraspEventCfg,
    _ARM_BODIES,
    _BODY_PART_BODIES,
    _HAND_CONTACT_BODIES,
    _HAND_OBJECT_SENSOR_NAMES,
    _TABLE_HEIGHT,
)

# ---------------------------------------------------------------------------
# Depth camera placement.
#
# Tripod-mounted, ~80 cm from workspace center, looking down at 35°.
# Key positions:
#   robot_base   ≈ (-0.10, 0, 0.28)       depth ≈ 1.11 m
#   table_center ≈ (-0.45, 0, 0.255)      depth ≈ 0.84 m
#   object_spawn ≈ x ∈ [-0.75, -0.45]     depth ≈ 0.62–0.82 m
#   hand_grasp   ≈ (-0.60, 0, 0.40)       depth ≈ 0.63 m
#   floor        ≈ z = 0                  depth ≈ 1.03 m
#
# Camera placed past the objects (further along -X), looking BACK toward
# the robot arm (+X direction) at 35° down.  Both objects (foreground) and
# the robot arm (background) are visible.
#   look_at = (-0.425, 0, 0.35)   (midpoint between objects and robot base)
#   cam is 80 cm from that:  (-1.080, 0, 0.809)
#   forward = (+0.819, 0, -0.574)
#   83° FOV covers ~1.42 m width at 80 cm → full table + arm visible.
_CAM_POS = (-1.080, 0.0, 0.809)
# Quaternion (convention="world"): default forward=+X, up=+Z.
# +35° rotation about Y tilts forward from +X down to (+0.819, 0, -0.574).
_CAM_QUAT = (0.9537, 0.0, 0.3007, 0.0)  # (w, x, y, z)

_DEPTH_WIDTH = 160
_DEPTH_HEIGHT = 120
_DEPTH_NEAR = 0.10  # camera hardware near plane
_DEPTH_FAR = 1.20   # clip floor (1.13m) and background; keep table + objects + hand
# Normalization range: depth values are clamped to [MIN, MAX] then scaled to [0, 1].
# This improves object-vs-table contrast (3cm object ≈ 4.3% of range).
_DEPTH_NORM_MIN = 0.40  # hand/objects start at ~0.62 m; margin for close approach
_DEPTH_NORM_MAX = 1.20  # robot arm at ~1.11 m; excludes far background


# ---------------------------------------------------------------------------
# Scene: inherit base robust scene + add depth camera
# ---------------------------------------------------------------------------
@configclass
class DistillSceneCfg(RobustDexgraspTrainSetSceneCfg):
    """Robust scene augmented with a TiledCamera for depth observation."""

    depth_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/DepthCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=_CAM_POS,
            rot=_CAM_QUAT,
            convention="world",
        ),
        data_types=["distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,  # wider FOV (~83°) to cover the hand + object workspace
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(_DEPTH_NEAR, _DEPTH_FAR),  # 0.10–1.20 m
        ),
        width=_DEPTH_WIDTH,
        height=_DEPTH_HEIGHT,
    )


# ---------------------------------------------------------------------------
# Observations: split into student (proprio + depth) and teacher (privileged)
# ---------------------------------------------------------------------------
@configclass
class DistillObservationsCfg:
    """Three observation groups for distillation.

    - ``student``: real-robot proprioception (92D) — joint pos/vel, target error, last action.
    - ``student_depth``: depth image (1×120×160) from the TiledCamera.
    - ``teacher``: full privileged observations (496D, same as teacher PPO policy).
    """

    @configclass
    class StudentCfg(ObsGroup):
        """Proprioceptive observations available on the real robot (92D).

        Follows DEXTRAH's student obs design: joint positions, velocities,
        target errors, and previous actions.  All are directly available on
        physical hardware from joint encoders and the action buffer.
        """

        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["xarm_joint_(1|2|3|4|5|6|7)", "(thumb|index|middle|ring)_joint_(0|1|2|3)"],
                )
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["xarm_joint_(1|2|3|4|5|6|7)", "(thumb|index|middle|ring)_joint_(0|1|2|3)"],
                )
            },
        )
        joint_target_error = ObsTerm(
            func=mdp.joint_target_error,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["xarm_joint_(1|2|3|4|5|6|7)", "(thumb|index|middle|ring)_joint_(0|1|2|3)"],
                )
            },
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True  # noise injected for sim-to-real
            self.concatenate_terms = True
            self.history_length = 0

    @configclass
    class StudentDepthCfg(ObsGroup):
        """Depth image from the TiledCamera sensor."""

        depth_image = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "data_type": "distance_to_camera",
                "normalize": True,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 0

    @configclass
    class TeacherCfg(ObsGroup):
        """Full privileged observations (496D, same as the trained teacher PPO policy)."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["xarm_joint_(1|2|3|4|5|6|7)", "(thumb|index|middle|ring)_joint_(0|1|2|3)"],
                )
            },
        )
        joint_target_error = ObsTerm(
            func=mdp.joint_target_error,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["xarm_joint_(1|2|3|4|5|6|7)", "(thumb|index|middle|ring)_joint_(0|1|2|3)"],
                )
            },
        )
        affordance_contact = ObsTerm(
            func=mdp.contact_binary_from_sensors,
            params={"sensor_names": _HAND_OBJECT_SENSOR_NAMES, "threshold": 0.01},
        )
        affordance_impulse = ObsTerm(
            func=mdp.force_norm_from_sensors,
            params={"sensor_names": _HAND_OBJECT_SENSOR_NAMES},
        )
        joint_height = ObsTerm(
            func=mdp.body_height_to_table,
            params={
                "body_asset_cfg": SceneEntityCfg("robot", body_names=_BODY_PART_BODIES),
                "table_height": _TABLE_HEIGHT,
            },
        )
        arm_height = ObsTerm(
            func=mdp.body_height_to_table,
            params={
                "body_asset_cfg": SceneEntityCfg("robot", body_names=_ARM_BODIES),
                "table_height": _TABLE_HEIGHT,
            },
        )
        hand_center_w = ObsTerm(
            func=mdp.hand_center_world,
            params={"hand_body_cfg": SceneEntityCfg("robot", body_names=["palm_link_site"])},
        )
        wrist_euler_delta = ObsTerm(
            func=mdp.WristEulerDeltaObservation,
            params={"wrist_body_cfg": SceneEntityCfg("robot", body_names=["palm_link_site"])},
        )
        wrist_euler = ObsTerm(
            func=mdp.wrist_euler_xyz,
            params={"wrist_body_cfg": SceneEntityCfg("robot", body_names=["palm_link_site"])},
        )
        object_point_cloud = ObsTerm(
            func=mdp.object_point_cloud_b,
            clip=(-2.0, 2.0),
            params={
                "object_cfg": SceneEntityCfg("object"),
                "ref_asset_cfg": SceneEntityCfg("robot"),
                "num_points": 128,
                "flatten": True,
                "visualize": False,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            # Must match teacher training config (history_length=2)
            self.history_length = 2

    student: StudentCfg = StudentCfg()
    student_depth: StudentDepthCfg = StudentDepthCfg()
    teacher: TeacherCfg = TeacherCfg()


# ---------------------------------------------------------------------------
# Events: inherit base robust events + add sim-to-real domain randomization
# ---------------------------------------------------------------------------
@configclass
class DistillEventCfg(RobustDexgraspEventCfg):
    """Domain randomization for sim-to-real transfer.

    All physics DR (friction, mass, actuator gains, joint friction, robot base
    height) is inherited from ``RobustDexgraspEventCfg``.  Distillation-specific
    additions (e.g. camera pose randomization) go here.
    """

    pass


# ---------------------------------------------------------------------------
# Top-level distillation environment config
# ---------------------------------------------------------------------------
@configclass
class DexgraspXArm7TilburgDistillEnvCfg(DexgraspXArm7TilburgRobustTrainSetEnvCfg):
    """Distillation env: teacher (496D privileged) → student (92D proprio + 1×120×160 depth)."""

    scene: DistillSceneCfg = DistillSceneCfg(
        num_envs=1024, env_spacing=3, replicate_physics=False
    )
    observations: DistillObservationsCfg = DistillObservationsCfg()
    events: DistillEventCfg = DistillEventCfg()


@configclass
class DexgraspXArm7TilburgDistillEnvCfg_PLAY(DexgraspXArm7TilburgDistillEnvCfg):
    scene: DistillSceneCfg = DistillSceneCfg(
        num_envs=128, env_spacing=3, replicate_physics=False
    )
