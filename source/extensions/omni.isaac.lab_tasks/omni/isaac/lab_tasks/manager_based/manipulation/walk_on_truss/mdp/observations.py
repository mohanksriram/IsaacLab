# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


# def object_position_in_robot_root_frame(
#     env: ManagerBasedRLEnv,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
# ) -> torch.Tensor:
#     """The position of the object in the robot's root frame."""
#     robot: RigidObject = env.scene[robot_cfg.name]
#     object: RigidObject = env.scene[object_cfg.name]
#     object_pos_w = object.data.root_pos_w[:, :3]
#     object_pos_b, _ = subtract_frame_transforms(
#         robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
#     )
#     return object_pos_b


def rel_ee_cube_distance(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    target_pos = env.scene['cube_frame'].data.target_pos_w[..., 0, :]
    ee_frame = env.scene['ee_frame'].data.target_pos_w[..., 0, :]
    rel_cube_pose = target_pos - ee_frame

    return rel_cube_pose

def gripper_state(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[robot_cfg.name]
    angles = asset.data.joint_pos[:, robot_cfg.joint_ids]
    left_gripper_angle = angles[:, 20]
    right_gripper_angle = angles[:, 21]

    gripper_angle = torch.maximum(left_gripper_angle, right_gripper_angle)

    gripper_state = 1-(gripper_angle/0.04)

    return gripper_state

    # return torch.maximum(left_gripper_angle, right_gripper_angle)


def eef_pose_with_base_frame(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    # import pdb; pdb.set_trace()
    ee_pos = env.scene['ee_frame'].data.target_pos_source[..., 0, :]
    ee_quat = env.scene['ee_frame'].data.target_quat_source[..., 0, :]
    ee_pose = torch.cat((ee_pos, ee_quat), dim=-1)
    return ee_pose
