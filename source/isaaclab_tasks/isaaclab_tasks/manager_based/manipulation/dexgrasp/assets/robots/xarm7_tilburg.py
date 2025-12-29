# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the XArm7 robots with Tilburg Hand and Digit 360 mounted.

The following configurations are available:

* :obj:`XARM7_TILBURG_CFG`: XArm7 with Tilburg Hand and Digit 360 with implicit actuator model.

"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##
import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

XARM7_TILBURG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=os.path.join(CURRENT_DIR, "old_assets", "xarm7_tilburg_digitv2.usd"),
        usd_path=os.path.join(CURRENT_DIR, "xarm7_tilburg_hand.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0)
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0.10, 0.0, 0.280), 
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            # "xarm_joint_(1|3|5|6|7)": 0.0,
            # "xarm_joint_2": -0.8,
            # "xarm_joint_4": 0.7,
            "xarm_joint_(1|2)": 0.0,
            "xarm_joint_3": 0.785398,
            "xarm_joint_4": 1.570796,
            "xarm_joint_5": 1.570796,
            "xarm_joint_6": 1.082104,
            "xarm_joint_7": -2.684857,
            "(index|middle|ring)_joint_0": 0.0,
            "(index|middle|ring)_joint_1": 0.6,
            "(index|middle|ring)_joint_2": 0.6,
            "(index|middle|ring)_joint_3": 0.6,
            "thumb_joint_0": 1.5,
            "thumb_joint_1": -0.60147215,
            "thumb_joint_2": 0.33795027,
            "thumb_joint_3": 0.60845138,
            # "xarm_joint_(1|2)": 0.0,
            # "xarm_joint_3": 0.0,
            # "xarm_joint_4": 1.3,
            # "xarm_joint_5": 0.0,
            # "xarm_joint_6": 0.0,
            # "xarm_joint_7": 0.0,
            # "(index|middle|ring)_joint_0": 0.0,
            # "(index|middle|ring)_joint_1": 0.0,
            # "(index|middle|ring)_joint_2": 0.0,
            # "(index|middle|ring)_joint_3": 0.0,
            # "thumb_joint_0": 0.0,
            # "thumb_joint_1": 0.0,
            # "thumb_joint_2": 0.0,
            # "thumb_joint_3": 0.0,
        },
    ),
    actuators={
        "xarm7_tilburg_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                "xarm_joint_(1|2|3|4|5|6|7)",
                "index_joint_(0|1|2|3)",
                "middle_joint_(0|1|2|3)",
                "ring_joint_(0|1|2|3)",
                "thumb_joint_(0|1|2|3)",
            ],
            effort_limit_sim={
                "xarm_joint_(1|2)": 50.0,
                "xarm_joint_(3|4|5)": 30.0,
                "xarm_joint_(6|7)": 20.0,
                "(thumb|index|middle|ring)_joint_(0|1|2|3)": 0.5,
            },
            stiffness={
                # "xarm_joint_(1|2)": 1500.0,     # From gainprm="1500"
                # "xarm_joint_(3|4|5)": 1000.0,   # From gainprm="1000"
                # "xarm_joint_(6|7)": 800.0,      # From gainprm="800"
                "xarm_joint_(1|2|3|4)": 300.0,     # From gainprm="1500"
                "xarm_joint_5": 100.0, # Derived from the Kuka Allegro setting
                "xarm_joint_6": 50.0, 
                "xarm_joint_7": 25.0,
                "(thumb|index|middle|ring)_joint_(0|1|2|3)": 3, # Following HORA paper's setting
            },
            damping={
                # "xarm_joint_(1|2)": 150.0,      # Derived from biasprm="... -150"
                # "xarm_joint_(3|4|5)": 100.0,    # Derived from biasprm="... -100"
                # "xarm_joint_(6|7)": 80.0,       # Derived from biasprm="... -80"
                "xarm_joint_(1|2|3|4)": 45.0,
                "xarm_joint_5": 20.0,
                "xarm_joint_6": 15.0,
                "xarm_joint_7": 15.0,
                "(thumb|index|middle|ring)_joint_(0|1|2|3)": 0.1,
            },
            friction={
                "xarm_joint_(1|2|3|4|5|6|7)": 1.0,  # From frictionloss="1"
                "(thumb|index|middle|ring)_joint_(0|1|2|3)": 0.01, # From frictionloss="0.01"
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
