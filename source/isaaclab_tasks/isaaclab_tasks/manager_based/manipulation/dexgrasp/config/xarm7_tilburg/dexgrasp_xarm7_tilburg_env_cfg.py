# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from ...assets.robots.xarm7_tilburg import XARM7_TILBURG_CFG
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

# Ensure these match your actual folder structure
from ... import grasp_env_cfg as grasp
from ... import train_set_env_cfg as train_set
from ... import mdp


@configclass
class XArm7TilburgRelJointPosActionCfg:
    # Controls all joints (arm + hand) with relative position
    action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.1,
    )


@configclass
class XArm7TilburgReorientRewardCfg(grasp.RewardsCfg):
    # Note: Ensure your custom 'mdp.contacts' function implements the logic 
    # for "one finger must be thumb", as standard contact functions usually just sum forces.
    good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=0.5,
        params={"threshold": 1.0},
    )


@configclass
class XArm7TilburgTrainSetRewardCfg(XArm7TilburgReorientRewardCfg):
    """Reward cfg for XArm7-Tilburg TrainSet tasks.

    Keeps the same reward terms as the default XArm7-Tilburg tasks (including `good_finger_contact`),
    but applies the TrainSet-specific weights.
    """

    def __post_init__(self):
        super().__post_init__()
        self.position_tracking.weight = train_set._TRAIN_SET_POSITION_TRACKING_WEIGHT
        self.success.weight = train_set._TRAIN_SET_SUCCESS_WEIGHT
        # Keep action terms as penalties. Treat the configured values as magnitudes.
        self.action_l2.weight = -abs(train_set._TRAIN_SET_ACTION_L2_WEIGHT)
        self.action_rate_l2.weight = -abs(train_set._TRAIN_SET_ACTION_RATE_L2_WEIGHT)


@configclass
class XArm7TilburgMixinCfg:
    rewards: XArm7TilburgReorientRewardCfg = XArm7TilburgReorientRewardCfg()
    actions: XArm7TilburgRelJointPosActionCfg = XArm7TilburgRelJointPosActionCfg()

    def __post_init__(self: grasp.DexgraspReorientEnvCfg):
        super().__post_init__()
        
        # 1. Set the correct Robot Config
        self.scene.robot = XARM7_TILBURG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 2. Update Object Pose tracking to the Hand Base
        self.commands.object_pose.body_name = "palm_link_site"

        # 3. Define the Tip Bodies (Rigid Bodies from your XML)
        finger_tip_body_list = [
            "index_link_3", 
            "middle_link_3", 
            "ring_link_3", 
            "thumb_link_3",
        ]
        
        # 4. Create Contact Sensors for each tip
        for link_name in finger_tip_body_list:
            setattr(
                self.scene,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/tilburg_hand/" + link_name,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                ),
            )

        # 5. Update Observations
        self.observations.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": [f"{link}_object_s" for link in finger_tip_body_list]},
            clip=(-20.0, 20.0),  # contact force in finger tips is under 20N normally
        )

        # 6. Update Hand State Observation (Proprioception)
        # Tracks the base and the tips. 
        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = ["palm_link_site", ".*_link_3"]

        # 7. Update Reward Assets
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["palm_link_site", ".*_link_3"])


@configclass
class XArm7TilburgTrainSetMixinCfg(XArm7TilburgMixinCfg):
    """Mixin for train-set objects converted to rigid bodies."""

    rewards: XArm7TilburgTrainSetRewardCfg = XArm7TilburgTrainSetRewardCfg()

    def __post_init__(self: train_set.DexgraspReorientTrainSetEnvCfg):
        super().__post_init__()

        # Train-set objects are rigid bodies; filter against the object root prim.
        finger_tip_body_list = [
            "index_link_3",
            "middle_link_3",
            "ring_link_3",
            "thumb_link_3",
        ]
        for link_name in finger_tip_body_list:
            sensor_cfg = getattr(self.scene, f"{link_name}_object_s", None)
            if sensor_cfg is not None:
                sensor_cfg.filter_prim_paths_expr = ["{ENV_REGEX_NS}/Object/bottom"]


@configclass
class DexgraspXArm7TilburgReorientEnvCfg(XArm7TilburgMixinCfg, grasp.DexgraspReorientEnvCfg):
    pass


@configclass
class DexgraspXArm7TilburgReorientEnvCfg_PLAY(XArm7TilburgMixinCfg, grasp.DexgraspReorientEnvCfg_PLAY):
    pass


@configclass
class DexgraspXArm7TilburgLiftEnvCfg(XArm7TilburgMixinCfg, grasp.DexgraspLiftEnvCfg):
    pass


@configclass
class DexgraspXArm7TilburgLiftEnvCfg_PLAY(XArm7TilburgMixinCfg, grasp.DexgraspLiftEnvCfg_PLAY):
    pass

@configclass
class DexgraspXArm7TilburgLiftEnvCfg_PLAY_Mild(XArm7TilburgMixinCfg, grasp.DexgraspLiftEnvCfg_PLAY_Mild):
    pass


@configclass
class DexgraspXArm7TilburgReorientTrainSetEnvCfg(XArm7TilburgTrainSetMixinCfg, train_set.DexgraspReorientTrainSetEnvCfg):
    pass


@configclass
class DexgraspXArm7TilburgReorientTrainSetEnvCfg_PLAY(
    XArm7TilburgTrainSetMixinCfg, train_set.DexgraspReorientTrainSetEnvCfg_PLAY
):
    pass


@configclass
class DexgraspXArm7TilburgLiftTrainSetEnvCfg(XArm7TilburgTrainSetMixinCfg, train_set.DexgraspLiftTrainSetEnvCfg):
    pass


@configclass
class DexgraspXArm7TilburgLiftTrainSetEnvCfg_PLAY(
    XArm7TilburgTrainSetMixinCfg, train_set.DexgraspLiftTrainSetEnvCfg_PLAY
):
    pass


@configclass
class DexgraspXArm7TilburgLiftTrainSetEnvCfg_PLAY_Mild(
    XArm7TilburgTrainSetMixinCfg, train_set.DexgraspLiftTrainSetEnvCfg_PLAY_Mild
):
    pass
