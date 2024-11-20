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
cube_holding_dist_thres = 0.01

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)

def exponential_dists(norm_dist):
    cs = torch.tensor([10., 5., 1.], dtype=norm_dist.dtype, device=norm_dist.device)
    res = torch.zeros_like(norm_dist)
    for c in cs:
        res += torch.exp(-norm_dist/c)
    
    res -= 3*norm_dist
    # res -= 4*norm_dist
    return res

def high_reward_falloff(norm_dist, a=15, b=30, c=1):
    return a * torch.exp(-b * norm_dist) - c * norm_dist

def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    # object: RigidObject = env.scene[object_cfg.name]
    # new_object = env.scene['new_object']
    cube_pos_w = env.scene['cube_frame'].data.target_pos_w[..., 0, :]
    # cube_pos_w += torch.tensor([-15.82, -9.067, -3.8756], dtype=cube_pos_w.dtype, device=cube_pos_w.device)

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    # cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]

    # print(f"ee_w: {ee_w}, cube_pos_w: {cube_pos_w}")
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return object_ee_distance

def calculate_stable_grasp_reward(gripper_angle, object_ee_distance, stable_grasp_wt):
    # Create boolean tensors for each condition
    # angle_condition = gripper_angle > 0.6
    # angle_condition = gripper_angle > gripper_angle_thres
    angle_condition = gripper_angle < gripper_angle_thres
    distance_condition = object_ee_distance < cube_holding_dist_thres
    
    # Combine conditions
    stable_grasp_condition = torch.logical_and(angle_condition, distance_condition)
    
    # Convert boolean tensor to float
    reward = stable_grasp_condition.float()
    
    # Apply the weight
    weighted_reward = reward * stable_grasp_wt
    
    return weighted_reward

def object_ee_distance_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    # object: RigidObject = env.scene[object_cfg.name]
    # new_object = env.scene['new_object']
    cube_pos_w = env.scene['cube_frame'].data.target_pos_w[..., 0, :]
    # cube_pos_w += torch.tensor([-15.82, -9.067, -3.8756], dtype=cube_pos_w.dtype, device=cube_pos_w.device)

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    # cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]

    # print(f"ee_w: {ee_w}, cube_pos_w: {cube_pos_w}")
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    if torch.isnan(object_ee_distance).any() or torch.isinf(object_ee_distance).any():
        print("Warning: NaN or Inf detected in object_ee_distance")
        breakpoint()
        object_ee_distance = torch.nan_to_num(object_ee_distance, nan=0.0, posinf=1e6, neginf=0.0)

    # object_ee_distance = torch.clamp(object_ee_distance, min=0, max=1e6)

    # final_reward = exponential_dists(object_ee_distance)
    # print(f"final_reward: {final_reward[0]}, object_ee_distance: {object_ee_distance[0]}")

    # setup a success reward for when the gripper has grasped the cube
    # import pdb; pdb.set_trace()
    asset = env.scene['robot']
    asset_cfg = SceneEntityCfg("robot")
    angles = asset.data.joint_pos[:, asset_cfg.joint_ids]

    # import pdb; pdb.set_trace()
    # gripper_angle = angles[:, 7]
    gripper_angle = angles[:, 21]
    # print(f"gripper_angle: {gripper_angle[0]}")

    # print(f"object distance: {object_ee_distance[0]}, gripper angle: {gripper_angle[0]}")
    # import pdb; pdb.set_trace()
    stable_grasp_reward = calculate_stable_grasp_reward_simple(gripper_angle, object_ee_distance, 5.0)
    # stable_grasp_reward = calculate_stable_grasp_reward_simple(gripper_angle, object_ee_distance, 2.5)

    neg_reward = gripper_close_when_far_reward(gripper_angle, object_ee_distance, -10.0)



    # add additional success reward when the gripper is closed and has a firm grasp on the cube
    # success_reward  = torch.where(gripper_angle > 0.7, 1.0, 0.0) and torch.where(object_ee_distance < 0.02, 1.0, 0.0)

    # print(f"stable grasp reward: {stable_grasp_reward[0]}")
    # print(f"gripper_angle: {gripper_angle}")
    # print(f'total reward: {final_reward[0] + stable_grasp_reward[0]}')
    # final_reward = exponential_dists(object_ee_distance)
    final_reward = high_reward_falloff(object_ee_distance)

    # print(f"final_reward: {final_reward[0]}, object_ee_distance: {object_ee_distance[0]}, stable_grasp_reward: {stable_grasp_reward[0]}, neg_reward: {neg_reward[0]}")

    # print(f"total reward: {final_reward[0] + stable_grasp_reward[0] + neg_reward[0]}")
    return final_reward
    return final_reward + stable_grasp_reward + neg_reward
    return final_reward #+ stable_grasp_reward + neg_reward

    # return 1 - torch.tanh(object_ee_distance / std)
    # return 1 - torch.tanh(object_ee_distance / std)

def improved_object_ee_distance_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Improved reward function for reaching and grasping the object."""
    cube_pos_w = env.scene['cube_frame'].data.target_pos_w[..., 0, :]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]

    # Distance of the end-effector to the object
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    # Reaching reward
    reaching_reward = exponential_dists(object_ee_distance)

    # Gripper angle
    asset = env.scene['robot']
    asset_cfg = SceneEntityCfg("robot")
    angles = asset.data.joint_pos[:, asset_cfg.joint_ids]
    gripper_angle = angles[:, 16]

    # Grasping reward
    grasping_reward = calculate_stable_grasp_reward(gripper_angle, object_ee_distance)


    # Combine rewards
    total_reward = reaching_reward + 2.0 * grasping_reward

    return total_reward

# def exponential_dists(distances: torch.Tensor, scale: float = 0.1) -> torch.Tensor:
#     """Calculate exponential reward based on distances."""
#     return torch.exp(-scale * distances)

def calculate_stable_grasp_reward_simple(gripper_angle: torch.Tensor, object_ee_distance: torch.Tensor, reward_scaler: float=1.0) -> torch.Tensor:
    """Calculate reward for stable grasping."""
    # Create boolean tensors for each condition
    angle_condition = gripper_angle < gripper_angle_thres
    # distance_condition = object_ee_distance < 0.1
    distance_condition = object_ee_distance < cube_holding_dist_thres
    # distance_condition = object_ee_distance < 0.4

    # Combine conditions
    stable_grasp_condition = torch.logical_and(angle_condition, distance_condition)

    # Convert boolean tensor to float
    reward = stable_grasp_condition.float()

    # if the gripper is closed but far from the object, negate the reward

    

    return reward * reward_scaler

def gripper_close_when_far_reward(gripper_angle: torch.Tensor, object_ee_distance: torch.Tensor, reward_scaler: float=-1.0) -> torch.Tensor:
    """Calculate reward for stable grasping."""
    # Create boolean tensors for each condition
    angle_condition = gripper_angle > 0.03
    # distance_condition = object_ee_distance < 0.1
    distance_condition = object_ee_distance > cube_holding_dist_thres*1.5

    # Combine conditions
    stable_grasp_condition = torch.logical_and(angle_condition, distance_condition)

    # Convert boolean tensor to float
    reward = stable_grasp_condition.float()

    # if the gripper is closed but far from the object, negate the reward
    

    return reward * reward_scaler


def calculate_stable_grasp_reward(gripper_angle: torch.Tensor, object_ee_distance: torch.Tensor) -> torch.Tensor:
    """Calculate reward for stable grasping."""
    close_to_object = torch.exp(-5.0 * object_ee_distance)
    gripper_closed = torch.sigmoid(10.0 * (gripper_angle - 0.5))
    return close_to_object * gripper_closed

def calculate_orientation_reward(ee_orientation: torch.Tensor) -> torch.Tensor:
    """Calculate reward for correct end-effector orientation."""
    target_orientation = torch.tensor([0, 1, 0, 0])  # Example target orientation (adjust as needed)
    orientation_diff = torch.sum((ee_orientation - target_orientation)**2, dim=-1)
    return torch.exp(-5.0 * orientation_diff)

def calculate_smoothness_penalty(action: torch.Tensor) -> torch.Tensor:
    """Calculate penalty for non-smooth actions."""
    return torch.sum(action**2, dim=-1)

def object_ee_orientation(
    env: ManagerBasedRLEnv,
    std: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    # import pdb; pdb.set_trace()
    # new_object = env.scene['new_object']
    # import pdb; pdb.set_trace()
    cube_quat_w = env.scene['cube_frame'].data.target_quat_w[..., 0, :]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    # cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_quat_w[..., 0, :]

    # Distance of the end-effector to the object: (num_envs,)
    # object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    # object
    return quat_error_magnitude(ee_w, cube_quat_w)
    # return 1 - torch.tanh(quat_error_magnitude(ee_w, cube_quat_w) / std)
    # return 1 - torch.tanh(quat_error_magnitude(ee_w, rotated_cube_quat_w) / std)

def object_ee_orientation_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    # import pdb; pdb.set_trace()
    # new_object = env.scene['new_object']
    # import pdb; pdb.set_trace()
    cube_quat_w = env.scene['cube_frame'].data.target_quat_w[..., 0, :]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    # cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_quat_w[..., 0, :]

    # Distance of the end-effector to the object: (num_envs,)
    # object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    # object
    quat_dist = quat_error_magnitude(ee_w, cube_quat_w)
    return exponential_dists(quat_dist)

