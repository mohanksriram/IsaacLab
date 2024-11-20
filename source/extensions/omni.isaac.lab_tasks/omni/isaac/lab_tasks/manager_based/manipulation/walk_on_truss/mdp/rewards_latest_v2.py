# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import FrameTransformer
from omni.isaac.lab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul


gripper_angle_thres = 0.001
cube_holding_dist_thres = 0.005
reaching_dist_thres = 0.01  # Threshold for when to start encouraging closing

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def full_reward(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    
    cube_pos_w = env.scene['cube_frame'].data.target_pos_w[..., 0, :]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]

    cube_rot_w = env.scene['cube_frame'].data.target_quat_w[..., 0, :]
    ee_rot_w = ee_frame.data.target_quat_w[..., 0, :]
    
    # Get gripper angles for both finger joints
    asset = env.scene['robot']
    asset_cfg = SceneEntityCfg("robot")
    angles = asset.data.joint_pos[:, asset_cfg.joint_ids]
    gripper_angles = angles[:, 20:22]  # Both finger joints

    left_gripper_contact = env.scene['contact_forces'].data.net_forces_w[0, 0]
    right_gripper_contact = env.scene['contact_forces'].data.net_forces_w[0, 1]

    print(f"contact forces: {left_gripper_contact}, {right_gripper_contact}")
    
    # Normalize gripper state for reward calculations (1 is closed, 0 is open)
    normalized_gripper_state = 1.0 - (torch.mean(gripper_angles, dim=1) / 0.04)
    
    # Calculate distance for use in multiple rewards
    distance = torch.norm(ee_pos_w - cube_pos_w, dim=-1)
     
    # Calculate rewards
    reaching_reward = calculate_reaching_reward(distance)
    orientation_reward = calculate_orientation_reward(ee_rot_w, cube_rot_w)
    
    # print(f"dists: {distance[0]}")
    # Only calculate grasping rewards when close enough
    grasping_reward = torch.where(
        distance < reaching_dist_thres,
        calculate_grasping_reward(normalized_gripper_state, gripper_angles, distance),
        torch.zeros_like(distance)
    )
    
    # Penalize closing the gripper too early
    premature_closing_penalty = calculate_premature_closing_penalty(normalized_gripper_state, distance)
    
    # Combine rewards with appropriate weights
    total_reward = (
        2.0 * reaching_reward + 
        0.5 * orientation_reward + 
        1.5 * grasping_reward - 
        1.0 * premature_closing_penalty  # Subtract the penalty
    )

    # breakpoint()
    
    return total_reward

def calculate_reaching_reward(distance):
    reaching_reward = torch.exp(-5.0 * distance)
    return reaching_reward

def calculate_orientation_reward(ee_rot, cube_rot):
    quat_diff = quaternion_difference(ee_rot, cube_rot)
    angle_diff = 2 * torch.acos(torch.clamp(quat_diff[..., 0], -1.0, 1.0))
    orientation_reward = torch.exp(-2.0 * angle_diff)
    return orientation_reward

def calculate_grasping_reward(normalized_gripper_state, gripper_angles, distance):
    # Only give grasping reward when very close
    close_enough = distance < cube_holding_dist_thres
    
    # Check if both fingers are closing symmetrically
    finger_diff = torch.abs(gripper_angles[:, 0] - gripper_angles[:, 1])
    symmetric_closure = torch.exp(-10.0 * finger_diff)
    
    # Reward for closing gripper when close to cube
    grasping_reward = torch.where(
        close_enough,
        2.0 * normalized_gripper_state * symmetric_closure,
        torch.zeros_like(normalized_gripper_state)
    )
    # if close_enough[0]:
    #     import pdb; pdb.set_trace()
    # print(f"grasping_reward: {grasping_reward[0]}")
    return grasping_reward

def calculate_premature_closing_penalty(normalized_gripper_state, distance):
    # Penalize closing the gripper when far from the cube
    far_from_cube = distance > reaching_dist_thres
    
    penalty = torch.where(
        far_from_cube,
        normalized_gripper_state,  # Penalty proportional to how closed the gripper is
        torch.zeros_like(normalized_gripper_state)
    )
    return penalty

def quaternion_difference(q1, q2):
    q1_inv = torch.cat([-q1[..., 1:], q1[..., :1]], dim=-1)
    return quaternion_multiply(q1_inv, q2)

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([w, x, y, z], dim=-1)