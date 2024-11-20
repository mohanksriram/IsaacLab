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

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


# def object_ee_distance(
#     env: ManagerBasedRLEnv,
#     # std: float,
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
#     ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
# ) -> torch.Tensor:
#     """Reward the agent for reaching the object using tanh-kernel."""
#     # extract the used quantities (to enable type-hinting)
#     object: RigidObject = env.scene[object_cfg.name]
#     # import pdb; pdb.set_trace()
#     # new_object = env.scene['new_object']
#     cube_pos_w = env.scene['cube_frame'].data.target_pos_w[..., 0, :]
#     ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
#     # Target object position: (num_envs, 3)
#     # cube_pos_w = object.data.root_pos_w
#     # End-effector position: (num_envs, 3)
#     ee_w = ee_frame.data.target_pos_w[..., 0, :]
#     # Distance of the end-effector to the object: (num_envs,)
#     return torch.norm(cube_pos_w - ee_w, dim=1)

    # return 1 - torch.tanh(object_ee_distance / std)

def exponential_dists(norm_dist):
    cs = torch.tensor([10., 5., 1.], dtype=norm_dist.dtype, device=norm_dist.device)
    res = torch.zeros_like(norm_dist)
    for c in cs:
        res += torch.exp(-norm_dist/c)
    
    res -= 4*norm_dist
    return res

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

    return exponential_dists(object_ee_distance)

    # return 1 - torch.tanh(object_ee_distance / std)
    # return 1 - torch.tanh(object_ee_distance / std)

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
    # return quat_error_magnitude(ee_w, cube_quat_w)
    return 1 - torch.tanh(quat_error_magnitude(ee_w, cube_quat_w) / std)
    # return 1 - torch.tanh(quat_error_magnitude(ee_w, rotated_cube_quat_w) / std)

def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)


# def object_goal_distance(
#     env: ManagerBasedRLEnv,
#     std: float,
#     minimal_height: float,
#     command_name: str,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
# ) -> torch.Tensor:
#     """Reward the agent for tracking the goal pose using tanh-kernel."""
#     # extract the used quantities (to enable type-hinting)
#     robot: RigidObject = env.scene[robot_cfg.name]
#     object: RigidObject = env.scene[object_cfg.name]
#     command = env.command_manager.get_command(command_name)
#     # compute the desired position in the world frame
#     des_pos_b = command[:, :3]
#     des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
#     # distance of the end-effector to the object: (num_envs,)
#     distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
#     # rewarded if the object is lifted above the threshold
#     return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))
