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
from ... import mdp


@configclass
class XArm7TilburgRelJointPosActionCfg:
    # Controls all joints (arm + hand) with relative position
    action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.1)


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
class XArm7TilburgMixinCfg:
    rewards: XArm7TilburgReorientRewardCfg = XArm7TilburgReorientRewardCfg()
    actions: XArm7TilburgRelJointPosActionCfg = XArm7TilburgRelJointPosActionCfg()

    def __post_init__(self: grasp.DexgraspReorientEnvCfg):
        super().__post_init__()
        
        # 1. Set the correct Robot Config
        self.scene.robot = XARM7_TILBURG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 2. Update Object Pose tracking to the Hand Base
        # "hand_base" is the root body of the hand in your XML
        self.commands.object_pose.body_name = "hand_base"

        # 3. Define the Tip Bodies (Rigid Bodies from your XML)
        finger_tip_body_list = [
            "index_digit360_tip", 
            "middle_digit360_tip", 
            "ring_digit360_tip", 
            "thumb_digit360_tip",
        ]
        finger_tip_body_base_list = [
            "index_digit360_base", 
            "middle_digit360_base", 
            "ring_digit360_base", 
            "thumb_digit360_base"
        ]
        
        # 4. Create Contact Sensors for each tip
        for link_name in finger_tip_body_list+finger_tip_body_base_list:
            setattr(
                self.scene,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/base/.*{link_name}",
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
        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = ["hand_base", ".*_digit360_tip"]

        # 7. Update Reward Assets
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["hand_base", ".*_digit360_tip"])

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