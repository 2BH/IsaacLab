# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ...assets.robots.xarm7_tilburg import XARM7_TILBURG_CFG
from ... import mdp
from ... import train_set_env_cfg as train_set

_TABLE_HEIGHT = 0.255
# For train-set rigid objects, top affordance geometry is nested under `bottom/collisions`.
# Sampling from this path matches the original Raisim setup that used the `top` body.
_OBJECT_PRIM_PATH = "/World/envs/env_.*/Object/bottom/collisions"
# Keep robust control slower than base dexgrasp (50 Hz) without the severe rollout-time hit of 0.2s control_dt.
_ROBUST_PD_STIFFNESS_SCALE = 1.0
_ROBUST_PD_DAMPING_SCALE = 1.0
_ROBUST_ARM_ACTION_SCALE = 0.1
_ROBUST_HAND_ACTION_SCALE = 0.0075
_HOLD_TARGET_X = -0.75
_HOLD_TARGET_Y = 0.0
_HOLD_TARGET_HEIGHT = 0.25

_HAND_TIPS_BODIES = [
    "palm_link_site",
    "thumb_link_3",
    "index_link_3",
    "middle_link_3",
    "ring_link_3",
]

_BODY_PART_BODIES = [
    "palm_link_site",
    "thumb_link_0",
    "thumb_link_1",
    "thumb_link_2",
    "thumb_link_3",
    "index_link_0",
    "index_link_1",
    "index_link_2",
    "index_link_3",
    "middle_link_0",
    "middle_link_1",
    "middle_link_2",
    "middle_link_3",
    "ring_link_0",
    "ring_link_1",
    "ring_link_2",
    "ring_link_3",
]

_HAND_CONTACT_BODIES = [
    "palm_link",
    "thumb_link_0",
    "thumb_link_1",
    "thumb_link_2",
    "thumb_link_3",
    "index_link_0",
    "index_link_1",
    "index_link_2",
    "index_link_3",
    "middle_link_0",
    "middle_link_1",
    "middle_link_2",
    "middle_link_3",
    "ring_link_0",
    "ring_link_1",
    "ring_link_2",
    "ring_link_3",
]

_GOOD_CONTACT_TIP_BODIES = [
    "thumb_link_3",
    "index_link_3",
    "middle_link_3",
    "ring_link_3",
]
_GOOD_CONTACT_TIP_SENSOR_NAMES = [f"{name}_object_s" for name in _GOOD_CONTACT_TIP_BODIES]

_ARM_BODIES = [
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
]
_ARM_COLLISION_BODIES = _ARM_BODIES[1:5]

_HAND_OBJECT_SENSOR_NAMES = [f"{name}_object_s" for name in _HAND_CONTACT_BODIES]
_PUSH_STRICT_SENSOR_NAMES = ["thumb_link_3_object_s"]
_HAND_TABLE_SENSOR_NAMES = [f"{name}_table_s" for name in _HAND_CONTACT_BODIES]
_ARM_INTERACTION_SENSOR_NAMES = [f"{name}_arm_s" for name in _ARM_BODIES]
_ARM_COLLISION_SENSOR_NAMES = [f"{name}_all_s" for name in _ARM_COLLISION_BODIES]


@configclass
class RobustDexgraspActionsCfg:
    action = mdp.RelativeJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=["xarm_joint_(1|2|3|4|5|6|7)", "(thumb|index|middle|ring)_joint_(0|1|2|3)"],
        scale={
            "xarm_joint_(1|2|3|4|5|6|7)": _ROBUST_ARM_ACTION_SCALE,
            "(thumb|index|middle|ring)_joint_(0|1|2|3)": _ROBUST_HAND_ACTION_SCALE,
        },
    )


@configclass
class RobustDexgraspObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["xarm_joint_(1|2|3|4|5|6|7)", "(thumb|index|middle|ring)_joint_(0|1|2|3)"])},
        )
        joint_target_error = ObsTerm(
            func=mdp.joint_target_error,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["xarm_joint_(1|2|3|4|5|6|7)", "(thumb|index|middle|ring)_joint_(0|1|2|3)"])},
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
            params={"body_asset_cfg": SceneEntityCfg("robot", body_names=_BODY_PART_BODIES), "table_height": _TABLE_HEIGHT},
        )
        arm_height = ObsTerm(
            func=mdp.body_height_to_table,
            params={"body_asset_cfg": SceneEntityCfg("robot", body_names=_ARM_BODIES), "table_height": _TABLE_HEIGHT},
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
        # affordance_vec = ObsTerm(
        #     func=mdp.AffordanceVectorObservation,
        #     params={
        #         "contact_body_cfg": SceneEntityCfg("robot", body_names=_BODY_PART_BODIES),
        #         "object_cfg": SceneEntityCfg("object"),
        #         "num_points": 200,
        #         "object_top_prim_path": _OBJECT_PRIM_PATH,
        #         "rotate_with_object": True,
        #     },
        # )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 0

    policy: PolicyCfg = PolicyCfg()


@configclass
class RobustDexgraspEventCfg(train_set.TrainSetEventCfg):
    # Keep original dexgrasp train-set events, but disable zero-gravity reset for robust training.
    # This keeps the physics scene at realistic gravity (-9.81 m/s^2).
    variable_gravity = None

    # Cap object scale at 1.1× (parent uses 1.5×).  Objects larger than ~1.1× the canonical size
    # can exceed the hand's grasping span, making them ungrasped by top-surface fingertip contact.
    # Keeping ungrasplable objects in training wastes gradient capacity without useful signal.
    randomize_object_scale = EventTerm(
        func=mdp.randomize_rigid_body_scale_safe,
        mode="prestartup",
        params={"scale_range": (0.9, 1.1), "asset_cfg": SceneEntityCfg("object")},
    )

    # Tighten object reset range: keep objects closer to the hold target (x=-0.75).
    # Base object pos is (-0.45), offset [-0.40, -0.20] yields x in [-0.85, -0.65],
    # centered around the hold target and within one arm reach.
    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.40, -0.20],
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
class RobustDexgraspRewardsCfg:
    # Dense bounded proximity reward in [0, 1]: pull fingertips toward nearest affordance point.
    # Intentionally fingertip-only: using all hand links would allow the agent to satisfy the reward
    # by pressing the palm flat onto the top surface (a local optimum that is not a grasp).
    # Uses tanh kernel (bounded) instead of raw negative distance to keep signal predictable
    # and comparable in scale with penalty terms.
    r_dis = RewTerm(
        func=mdp.AffordanceDistanceTanhReward,
        weight=4.0,
        params={
            "contact_body_cfg": SceneEntityCfg("robot", body_names=_HAND_TIPS_BODIES),
            "object_cfg": SceneEntityCfg("object"),
            "num_points": 200,
            "object_top_prim_path": _OBJECT_PRIM_PATH,
            "std": 0.15,
        },
    )
    # Desired hand-object contact magnitude reward (XY impulse/force proxy in object frame).
    # Soft height scaling: 20% reward while object is on the table (enough to learn grasping),
    # ramps to 100% once lifted 2cm. Prevents the local optimum of pressing/squeezing without
    # lifting while still providing signal for learning to close fingers.
    r_contact_desired_impulse = RewTerm(
        func=mdp.affordance_impulse_reward,
        weight=2.0,
        params={
            "hand_object_sensor_names": _HAND_OBJECT_SENSOR_NAMES,
            "object_cfg": SceneEntityCfg("object"),
            "table_height": _TABLE_HEIGHT,
            "on_table_scale": 0.2,
            "lift_bonus_height": 0.02,
        },
    )
    # Dexgrasp-style good contact shaping: thumb tip must contact together with any other fingertip.
    # good_finger_contact = RewTerm(
    #     func=mdp.contacts,
    #     weight=0.5,
    #     params={"threshold": 1.0},
    # )
    # Continuous lift reward toward target height with tanh smoothing, gated by soft finger contact.
    # Includes a hold-duration bonus that rewards sustained grasps over brief tosses.
    # XY proximity is now a separate additive reward (xy_proximity_reward) to avoid
    # multiplicative gating that kills lift signal when object is far from hold target.
    lift_reward = RewTerm(
        func=mdp.SustainedLiftReward,
        weight=10.0,
        params={
            "table_height": _TABLE_HEIGHT,
            "target_height": _HOLD_TARGET_HEIGHT,
            "std": 0.12,
            "min_hold_height": 0.03,
            "hold_steps_for_full_bonus": 60.0,
            "object_cfg": SceneEntityCfg("object"),
            "good_contact_tip_sensor_names": _GOOD_CONTACT_TIP_SENSOR_NAMES,
            "good_contact_threshold": 0.5,
            "contact_temperature": 0.5,
        },
    )
    # Encourage stable hold at a fixed world pose to avoid keeping the object too close to the arm.
    hold_position_reward = RewTerm(
        func=mdp.fixed_hold_position_reward,
        weight=10.0,
        params={
            "table_height": _TABLE_HEIGHT,
            "target_height": _HOLD_TARGET_HEIGHT,
            "target_x": _HOLD_TARGET_X,
            "target_y": _HOLD_TARGET_Y,
            "std": 0.12,
            "min_hold_height": 0.03,
            "object_cfg": SceneEntityCfg("object"),
            "good_contact_tip_sensor_names": _GOOD_CONTACT_TIP_SENSOR_NAMES,
            "good_contact_threshold": 0.5,
            "contact_temperature": 0.5,
        },
    )
    # Standalone XY proximity reward — decoupled from lift to provide independent
    # horizontal positioning gradient without suppressing the vertical lift signal.
    xy_proximity_reward = RewTerm(
        func=mdp.xy_proximity_reward,
        weight=5.0,
        params={
            "target_x": _HOLD_TARGET_X,
            "target_y": _HOLD_TARGET_Y,
            "std": 0.15,
            "min_hold_height": 0.03,
            "table_height": _TABLE_HEIGHT,
            "object_cfg": SceneEntityCfg("object"),
            "good_contact_tip_sensor_names": _GOOD_CONTACT_TIP_SENSOR_NAMES,
            "good_contact_threshold": 0.5,
            "contact_temperature": 0.5,
        },
    )

    # Additive bonus for engaging more fingertips — discourages degenerate 2-finger grips.
    r_finger_diversity = RewTerm(
        func=mdp.finger_diversity_reward,
        weight=3.0,
        params={
            "good_contact_tip_sensor_names": _GOOD_CONTACT_TIP_SENSOR_NAMES,
            "good_contact_threshold": 0.5,
            "temperature": 0.5,
        },
    )

    # Penalize hand links approaching the table (log barrier near table height).
    hand_table_reward = RewTerm(
        func=mdp.table_log_penalty,
        weight=-0.15,
        params={"contact_body_cfg": SceneEntityCfg("robot", body_names=_BODY_PART_BODIES), "table_height": _TABLE_HEIGHT},
    )
    # Penalize hand links touching the table.
    hand_table_contact_reward = RewTerm(
        func=mdp.table_contact_reward,
        weight=-0.15,
        params={"hand_table_sensor_names": _HAND_TABLE_SENSOR_NAMES, "threshold": 0.01},
    )
    # Penalize hand-table contact impulse magnitude.
    hand_table_impulse_reward = RewTerm(
        func=mdp.table_impulse_reward,
        weight=-0.15,
        params={"hand_table_sensor_names": _HAND_TABLE_SENSOR_NAMES},
    )

    # Penalize arm links getting too close to the table.
    arm_height_reward = RewTerm(
        func=mdp.arm_height_log_penalty,
        weight=-0.05,
        params={"arm_body_cfg": SceneEntityCfg("robot", body_names=_ARM_COLLISION_BODIES), "table_height": _TABLE_HEIGHT},
    )
    # Penalize arm contacts with object/table (undesired arm interaction).
    arm_contact_reward = RewTerm(
        func=mdp.arm_contact_reward,
        weight=-0.1,
        params={"arm_sensor_names": _ARM_INTERACTION_SENSOR_NAMES, "threshold": 0.01},
    )
    # Penalize arm contact impulse magnitude (often the strongest instability term).
    arm_impulse_reward = RewTerm(
        func=mdp.arm_impulse_reward,
        weight=-0.1,
        params={"arm_sensor_names": _ARM_INTERACTION_SENSOR_NAMES},
    )
    # Penalize arm collision count from dedicated collision sensors.
    arm_collision_reward = RewTerm(
        func=mdp.arm_collision_count,
        weight=-0.1,
        params={"arm_contact_sensor_names": _ARM_COLLISION_SENSOR_NAMES, "threshold": 0.01},
    )

    # Penalize downward pressing of the object into the table (Z-axis only).
    push_reward = RewTerm(
        func=mdp.push_reward,
        weight=-0.5,
        params={
            "object_contact_sensor_names": _HAND_OBJECT_SENSOR_NAMES,
            "strict_threshold_sensor_names": _PUSH_STRICT_SENSOR_NAMES,
            "strict_threshold": 0.5,
            "default_threshold": 1.0,
        },
    )

    # Penalize wrist linear velocity — prevents high-speed smashing approaches.
    wrist_vel_reward = RewTerm(
        func=mdp.wrist_vel_reward,
        weight=-0.01,
        params={"wrist_body_cfg": SceneEntityCfg("robot", body_names=["palm_link"])},
    )
    # Penalize wrist angular velocity.
    wrist_qvel_reward = RewTerm(
        func=mdp.wrist_qvel_reward,
        weight=-0.005,
        params={"wrist_body_cfg": SceneEntityCfg("robot", body_names=["palm_link"])},
    )
    # Penalize high arm joint speed (includes extra scaling for large |qdot| in term implementation).
    arm_joint_vel_reward = RewTerm(
        func=mdp.arm_joint_vel_reward,
        weight=-0.02,
        params={"arm_joint_cfg": SceneEntityCfg("robot", joint_names="xarm_joint_(1|2|3|4|5|6|7)")},
    )

    # One-shot penalty on configured terminal events.
    terminal_penalty = RewTerm(
        func=mdp.is_terminated_term,
        weight=-15.0,
        # params={"term_keys": ["hand_below_table", "abnormal_robot"]},
        params={"term_keys": ["abnormal_robot"]},
    )


@configclass
class RobustDexgraspTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_out_of_bound = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "in_bound_range": {"x": (-1.5, 0.5), "y": (-2.0, 2.0), "z": (0.0, 2.0)},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    # Disabled: for larger objects the hand may need to reach below the table surface
    # to push the object to the edge and grasp it from the side.
    # The table-contact penalty rewards still discourage prolonged table contact.
    # hand_below_table = DoneTerm(
    #     func=mdp.any_body_below_table,
    #     params={
    #         "body_asset_cfg": SceneEntityCfg("robot", body_names=_BODY_PART_BODIES),
    #         "table_height": _TABLE_HEIGHT,
    #         "margin": 0.0,
    #     },
    # )

    abnormal_robot = DoneTerm(func=mdp.abnormal_robot_state)


@configclass
class RobustDexgraspTrainSetSceneCfg(train_set.TrainSetSceneCfg):
    """Train-set scene with an explicit visible table, matching old train-set setup style."""

    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.CuboidCfg(
            size=(1.4, 1.5, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.15, 0.15), roughness=0.25),
            visible=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.45, 0.0, 0.235), rot=(1.0, 0.0, 0.0, 0.0)),
    )


@configclass
class DexgraspXArm7TilburgRobustTrainSetEnvCfg(ManagerBasedRLEnvCfg):
    viewer: ViewerCfg = ViewerCfg(eye=(-1.75, 0.0, 0.9), lookat=(-0.45, 0.0, 0.4), origin_type="env")
    scene: RobustDexgraspTrainSetSceneCfg = RobustDexgraspTrainSetSceneCfg(
        num_envs=1024, env_spacing=3, replicate_physics=False
    )

    observations: RobustDexgraspObservationsCfg = RobustDexgraspObservationsCfg()
    actions: RobustDexgraspActionsCfg = RobustDexgraspActionsCfg()
    rewards: RobustDexgraspRewardsCfg = RobustDexgraspRewardsCfg()
    terminations: RobustDexgraspTerminationsCfg = RobustDexgraspTerminationsCfg()
    events: RobustDexgraspEventCfg = RobustDexgraspEventCfg()
    commands: object | None = None
    curriculum: object | None = None

    episode_length_s: float = 14.0
    is_finite_horizon: bool = True

    def __post_init__(self):
        self.scene.robot = XARM7_TILBURG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # RobustDexgrasp runs a slower control loop; soften PD gains for stability under longer command hold.
        actuator_cfg = self.scene.robot.actuators["xarm7_tilburg_actuators"]
        if isinstance(actuator_cfg.stiffness, dict):
            stiffness = {k: float(v) * _ROBUST_PD_STIFFNESS_SCALE for k, v in actuator_cfg.stiffness.items()}
        else:
            stiffness = actuator_cfg.stiffness
        if isinstance(actuator_cfg.damping, dict):
            damping = {k: float(v) * _ROBUST_PD_DAMPING_SCALE for k, v in actuator_cfg.damping.items()}
        else:
            damping = actuator_cfg.damping
        self.scene.robot.actuators["xarm7_tilburg_actuators"] = actuator_cfg.replace(
            stiffness=stiffness,
            damping=damping,
        )

        for body_name in _HAND_CONTACT_BODIES:
            body_path = "{ENV_REGEX_NS}/Robot/tilburg_hand/" + body_name
            setattr(
                self.scene,
                f"{body_name}_object_s",
                ContactSensorCfg(
                    prim_path=body_path,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object/bottom"],
                ),
            )
            setattr(
                self.scene,
                f"{body_name}_table_s",
                ContactSensorCfg(
                    prim_path=body_path,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/table"],
                ),
            )

        for body_name in _ARM_BODIES:
            setattr(
                self.scene,
                f"{body_name}_arm_s",
                ContactSensorCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/xarm7/{body_name}",
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object/bottom", "{ENV_REGEX_NS}/table"],
                ),
            )

        for body_name in _ARM_COLLISION_BODIES:
            setattr(
                self.scene,
                f"{body_name}_all_s",
                ContactSensorCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/xarm7/{body_name}",
                ),
            )

        # Keep physics step-size but run policy/control lower than the default task.
        self.decimation = 2 # 60 Hz control frequency
        self.sim.dt = 1/120
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01


@configclass
class DexgraspXArm7TilburgRobustTrainSetEnvCfg_PLAY(DexgraspXArm7TilburgRobustTrainSetEnvCfg):
    scene: RobustDexgraspTrainSetSceneCfg = RobustDexgraspTrainSetSceneCfg(
        num_envs=128, env_spacing=3, replicate_physics=False
    )
