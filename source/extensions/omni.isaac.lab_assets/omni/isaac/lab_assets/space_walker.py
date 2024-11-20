# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Space Walker robot.

The following configurations are available:

* :obj:`SPACE_WALKER_CFG`: Franka Emika Panda robot with Panda hand
* :obj:`SPACE_WALKER_HIGH_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control

"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

SPACE_WALKER_USD_PATH = "/mnt/Downloads/2F-85/space_walker_only.usd"

##
# Configuration
##
SPACE_WALKER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{SPACE_WALKER_USD_PATH}",
        activate_contact_sensors=False,
        # scale=(100.0, 100.0, 100.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # pos=(320.0, 650.0, 100.0),
        # rot=(0.707, 0., 0., -0.707),
        # joint_pos={
        #     "joint_1_.*": 0.0,
        #     "joint_2_.*": 0.175,
        #     "joint_3_.*": 0.0,
        #     "joint_4_.*": 1.89,
        #     "joint_5_.*": 0.0,
        #     "joint_6_.*": 0.6,
        #     "joint_7_.*": -1.57,
        #     "finger_joint_.*": 0.0,
        #     ".*_finger_joint_.*": 0.0,
        #     ".*_knuckle_joint_.*": 0.0
        # },
        #     joint_pos={
        #     "joint_1_2": 0.0,
        #     "joint_2_2": 0.175,
        #     "joint_3_2": 0.0,
        #     "joint_4_2": 1.89,
        #     "joint_5_2": 0.0,
        #     "joint_6_2": 0.6,
        #     "joint_7_2": -1.57,
        #     "finger_joint_2": 0.0,
        #     ".*_finger_joint_2": 0.0,
        #     ".*_knuckle_joint_2": 0.0
        # },
        joint_pos={
            "joint_1_2": 0.0,
            # "joint_2_2": 0.175,
            "joint_2_2": 1.13,
            "joint_3_2": 0.0,
            "joint_4_2": 1.89,
            "joint_5_2": 0.0,
            "joint_6_2": 0.6,
            "joint_7_2": -1.57,
            # "finger_joint_2": 0.0,
            # ".*_finger_joint_2": 0.0,
            # ".*_knuckle_joint_2": 0.0
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_[1-7]_2"],
            velocity_limit=100.0,
            effort_limit={
                "joint_[1-4]_2": 39.0,
                "joint_[5-7]_2": 9.0,
            },
            stiffness={
                "joint_[1-4]_2": 40.0,
                "joint_[5-7]_2": 15.0,
            },
            damping={
                "joint_[1-4]_2": 1.0,
                "joint_[5-7]_2": 0.5,
            },
        ),
        "robotiq_hand": ImplicitActuatorCfg(
            # joint_names_expr=["finger_joint_2"],
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit=20.0,
            velocity_limit=3.0,
            # stiffness=2e3,
            stiffness=1e2,
            damping=1,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of space walker robot."""


"""Configuration of Kinova arm robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""
