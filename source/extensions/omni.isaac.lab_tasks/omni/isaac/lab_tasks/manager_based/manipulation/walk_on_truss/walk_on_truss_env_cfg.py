# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg


from . import mdp

##
# Scene definition
##


@configclass
class SpaceSceneCfg(InteractiveSceneCfg):
    """Configuration for the space with truss scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # # target object: will be populated by agent env cfg
    # object: RigidObjectCfg = MISSING

    cube_frame: FrameTransformerCfg = MISSING

    # Table
    # table = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Table",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[1.0, 0.5, 0], rot=[0.707, 0, 0, 0.707]),
    #     spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    # )

    # new_object = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/NewObject",
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.6, 0.5, 0], rot=[0.707, 0, 0, 0.707]),
    #     spawn=UsdFileCfg(usd_path=f"/home/mohan/Downloads/2F-85/cube_plus_cylinder.usd",
    #                      rigid_props=RigidBodyPropertiesCfg(
    #                 solver_position_iteration_count=16,
    #                 solver_velocity_iteration_count=1,
    #                 max_angular_velocity=10.0,
    #                 max_linear_velocity=10.0,
    #                 max_depenetration_velocity=5.0,
    #                 disable_gravity=False,
    #             ),
    #                      ),

    truss = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Truss",
        # init_state=AssetBaseCfg.InitialStateCfg(pos=(0.2, -.9, -.3),),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.5, -.9, -.3),),
                # init_state=AssetBaseCfg.InitialStateCfg(pos=[0.65, 0.6, -0.2], rot=[0.707, 0, 0, 0.707]),
                # init_state=AssetBaseCfg.InitialStateCfg(pos=[-0.2, 0.4, 0.9], rot=[0.707, 0, 0.707, 0]),
                # init_state=AssetBaseCfg.InitialStateCfg(pos=[0.65, 0.6, 0.]),
        # init_state=AssetBaseCfg.InitialStateCfg(pos=[0.65, 0.6, 0.], rot=[-0.5, 0, 0.5, 0.5]),
        spawn=UsdFileCfg(usd_path=f"/mnt/Downloads/2F-85/truss_single_line.usd",
                         
                         rigid_props=RigidBodyPropertiesCfg(
                            kinematic_enabled=True,
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=0.0,
                    max_linear_velocity=0.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                         ),  
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            # pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
            pos_x=(0.6, 0.8), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
            # pos_x=(0.0, 0.0), pos_y=(0.0, 0.0), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    body_joint_pos: mdp.JointPositionActionCfg = MISSING
    finger_joint_pos: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.cube_position_in_robot_root_frame)
        # target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # reset_object_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         # "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
    #         "pose_range": {"x": (0.0, 0.0), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("object", body_names="Object"),
    #     },
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # reaching_object_tracking = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=-0.2)
    # reaching_object_tracking_fine_grained = RewTerm(func=mdp.object_ee_distance_tanh, params={"std": 0.1}, weight=0.1)
    # reaching_object_tracking_fine_grained = RewTerm(func=mdp.object_ee_distance_tanh, params={"std": 0.1}, weight=1)
    # orienting_object_tracking = RewTerm(func=mdp.object_ee_orientation, params={"std": 0.1}, weight=-0.3)
    
    
    reaching_object_tracking_fine_grained = RewTerm(func=mdp.object_ee_distance_tanh, params={"std": 0.1}, weight=0.5)
    orienting_object_tracking = RewTerm(func=mdp.object_ee_orientation, params={"std": 0.1}, weight=-0.3)
    
    # improved_object_ee_distance_reward = RewTerm(func=mdp.improved_object_ee_distance_reward, weight=1.0)

    # lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=15.0)

    # object_goal_tracking = RewTerm(
    #     func=mdp.object_goal_distance,
    #     params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
    #     weight=16.0,
    # )

    # object_goal_tracking_fine_grained = RewTerm(
    #     func=mdp.object_goal_distance,
    #     params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
    #     weight=5.0,
    # )
    # end_effector_position_tracking = RewTerm(
    #     func=mdp.object_ee_distance,
    #     weight=-0.2,
    #     # params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose"},
    # )
    # end_effector_position_tracking_fine_grained = RewTerm(
    #     func=mdp.object_ee_distance_tanh,
    #     weight=0.1,
    #     params={"std": 0.1},
        # params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "std": 0.1, "command_name": "ee_pose"},
    # )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        # weight=-1e-3,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # object_dropping = DoneTerm(
    #     func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    # )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    # joint_vel = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    # )


##
# Environment configuration
##

@configclass
class SpaceWalkerEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: SpaceSceneCfg = SpaceSceneCfg(num_envs=1024, env_spacing=10)
    # scene: SpaceSceneCfg = SpaceSceneCfg(num_envs=1536, env_spacing=10)
    # scene: SpaceSceneCfg = SpaceSceneCfg(num_envs=1600, env_spacing=10)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 15.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_max_rigid_patch_count = 250000
        # import pdb; pdb.set_trace()
