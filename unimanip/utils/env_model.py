# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import omni.usd
import isaaclab.sim as sim_utils

from pxr import UsdGeom, Gf
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sensors.camera import Camera, TiledCameraCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from unimanip.utils.general_utils import *
from unimanip.utils.ycb_model import YCBObjModel
from unimanip.utils.partnet_model import PartNetObjModel
from unimanip.utils.unidoor_model import UniDoorObjModel
from unimanip.utils.room_model import RoomModel



@configclass
class UniManipEnvCfg(DirectRLEnvCfg):
    """
    Configuration for the UniManip Environment defined with the direct workflow.
    
    UniManipEnvCfg defines: scene, object, room, ground, stage, terrain
    """
    
    # basic
    log_dir = ''
    vla_mode = False
    test_mode = False
    exit_flag = False
    action_type = 'open'
    success_threshold = 0.05
    heading_direction = (0.0, 1.0, 0.0)
    
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=5.0, replicate_physics=True)
    
    
    # object_infos
    object_infos = {
        'type': None,
        'group': None,
        'index': 0.0,
        'name': None,
        'scale': 1.0,
        'open_ratio': 0.6,
    }
    # default object: articulation
    object_usd_path = osp.join(ASSET_DIR)
    object_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Object",
        # UsdFileCfg
        spawn=sim_utils.UsdFileCfg(
            usd_path=object_usd_path,
            activate_contact_sensors=False,
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, max_depenetration_velocity=1.0, #linear_damping=0.0, angular_damping=0.0
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            semantic_tags=[("class", "object")],
        ),
        # InitialStateCfg
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "joints": 0.0,
            },
        ),
        # ImplicitActuatorCfg
        actuators={
            "joints": ImplicitActuatorCfg(
                joint_names_expr=["joints"],
                effort_limit=None,
                velocity_limit=None,
                stiffness=0.0,
                damping=0.1,
            ),
        },
    )
    
    # optional object: rigid
    rigid_object_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        # UsdFileCfg
        spawn=sim_utils.UsdFileCfg(
            usd_path=object_usd_path,
            activate_contact_sensors=False,
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, max_depenetration_velocity=1.0, #linear_damping=0.0, angular_damping=0.0
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            semantic_tags=[("class", "object")],
        ),
        # InitialStateCfg
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    
    
    # room_infos
    room_id = 0
    room_space = 4
    room_infos = {
        'name': None,
        'scale': 1.0,
        'place': None,
        'translation': (0.0, 0.0, 0.0),
        'orientation': (1.0, 0.0, 0.0, 0.0),
    }
    # room
    room_cfg = sim_utils.UsdFileCfg(
        usd_path=osp.join(ASSET_DIR, 'room/SimpleRoom.usd'),
        # usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Room/simple_room.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True, kinematic_enabled=True, disable_gravity=True
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        semantic_tags=[("class", "room")],
        scale=(1.0, 1.0, 1.0),
    )
    
    
    # ground
    ground_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Ground",
        # UsdFileCfg
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, kinematic_enabled=True, disable_gravity=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
            semantic_tags=[("class", "ground")],
        ),
        # InitialStateCfg
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, -0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    
    # stage
    stage_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Stage",
        # UsdFileCfg
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, kinematic_enabled=True, disable_gravity=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            semantic_tags=[("class", "stage")],
        ),
        # InitialStateCfg
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, -0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    
    
    # handle marker
    handle_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Marker",
        markers={
            "handle": sim_utils.SphereCfg(
                radius=0.04,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 1.0, 0.2)),
            ),
        },
    )
    
    # goal marker
    goal_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Marker",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.04,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2)),
            ),
        },
    )
    
    # control marker
    control_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Marker",
        markers={
            "control": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),
            ),
        },
    )
    
    
    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    
    """
    Please define the following in your RobotEnvCfg
    """
    
    # space
    state_space = 0
    action_space = 6 + 1  # hand pose + hand joint
    action_frame = 'world'
    observation_space = 0
    
    # run
    dt_s = 1 / 60  # sim runs in 60 fps
    decimation = 2  # control runs in 60 / 2 = 30 fps
    episode_length_s = 10  # episode runs in 10 x 30 = 300 timesteps
    # sim
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=dt_s,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    
    # reward_params
    reward_params = {}
    # control_params
    control_params = {}
    
    
    # robot_names
    robot_name = ''
    robot_base_body_name = ''
    robot_hand_body_name = ''
    robot_arm_body_names = []
    robot_hand_body_names = []
    robot_finger_body_names = []
    robot_arm_joint_names = []
    robot_hand_joint_names = []
    # robot
    robot_cfg = ArticulationCfg()
    
    
    # record camera
    record_size = 520
    record_step, save_record_step = 1, 1
    record_period = dt_s * decimation * record_step
    record_split, record_exist, record_episode, record_trajectory = 'train', 0, 10, False
    # camera head
    camera_head_cfg = TiledCameraCfg()
    # camera arm
    camera_arm_cfg = TiledCameraCfg()



class UniManipEnv(DirectRLEnv):
    # initialization
    #   |-- _setup_scene()
    #   |-- __init__()
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()
    
    cfg: UniManipEnvCfg
    
    def __init__(self, cfg: UniManipEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # simulation dt
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        
        """================ Init Markers ================"""
        # init goal, handle, control markers
        self.goal_markers = VisualizationMarkers(self.cfg.goal_marker_cfg)
        self.handle_markers = VisualizationMarkers(self.cfg.handle_marker_cfg)
        self.control_markers = VisualizationMarkers(self.cfg.control_marker_cfg)
        
        """================ Init Logger and Tracker ================"""
        # init global time and episode
        self.global_time, self.global_episode = 0, 0
        # init scene env origins (nenv, 3) and times(nenv, )
        self.env_origins = self.scene.env_origins
        self.env_times = torch.zeros(self.num_envs).to(self.device)
        self.env_episodes = torch.zeros(self.num_envs).to(self.device)
        
        # init log setting and log_tracker
        self.log_time, self.log_dir, self.log_tracker = 60 * self.cfg.episode_length_s, self.cfg.log_dir, []
        # init env_success, env_success_flag, env_success_count and success_tracker
        self.env_success = torch.zeros(self.num_envs, device=self.device)
        self.env_success_flag = torch.zeros(self.num_envs, device=self.device)
        self.env_success_ratio = torch.zeros(self.num_envs, device=self.device)
        self.env_success_count = torch.zeros(self.num_envs, device=self.device)
        self.success_tracker = {'episode': torch.tensor([1e-6], device=self.device),
                                'success': torch.tensor([0.0], device=self.device), 
                                'success_flag': torch.tensor([0.0], device=self.device),
                                'success_ratio': torch.tensor([0.0], device=self.device)}
        # init reward_tracker
        self.reward_tracker = None
        
        """================ Init Robot Default Params ================"""
        # init robot default root in global state
        self.robot_default_root_state = self._robot.data.default_root_state.clone()
        self.robot_default_root_state_w = self._robot.data.default_root_state.clone()
        self.robot_default_root_state_w[:, :3] += self.env_origins
        
        # init robot default joint
        self.robot_default_joint_pos = self._robot.data.default_joint_pos.clone()
        self.robot_default_joint_vel = torch.zeros_like(self.robot_default_joint_pos)
        # init robot default joint limits
        self.robot_joint_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0]
        self.robot_joint_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1]
        
        # init robot dimensions
        self.num_bodys = self._robot.data.body_link_state_w.shape[1]
        self.num_joints = self.robot_default_joint_pos.shape[1]
        
        # init robot body indices
        self.robot_base_body_index = self._robot.find_bodies(self.cfg.robot_base_body_name)[0][0]
        self.robot_hand_body_index = self._robot.find_bodies(self.cfg.robot_hand_body_name)[0][0]
        self.robot_arm_body_indices = [self._robot.find_bodies(name)[0][0] for name in self.cfg.robot_arm_body_names]
        self.robot_hand_body_indices = [self._robot.find_bodies(name)[0][0] for name in self.cfg.robot_hand_body_names]
        self.robot_finger_body_indices = [self._robot.find_bodies(name)[0][0] for name  in self.cfg.robot_finger_body_names]
        # init robot joint indices
        self.robot_arm_joint_indices = [self._robot.find_joints(name)[0][0] for name in self.cfg.robot_arm_joint_names]
        self.robot_hand_joint_indices = [self._robot.find_joints(name)[0][0] for name in self.cfg.robot_hand_joint_names]
        # init robot active body and joint indices
        self.robot_active_body_indices = self.robot_arm_body_indices + self.robot_hand_body_indices
        self.robot_active_joint_indices = self.robot_arm_joint_indices + self.robot_hand_joint_indices
        
        # rotate_base
        if 'rotate_base' in self.cfg.control_params and self.cfg.control_params['rotate_base']:
            joint_index = self._robot.find_joints('slider_basex')[0][0]
            self._robot.data.soft_joint_pos_limits[:, joint_index, 0] = - torch.pi * (self.cfg.control_params['rotate_base_angle'] / 180)
            self._robot.data.soft_joint_pos_limits[:, joint_index, 1] = torch.pi * (self.cfg.control_params['rotate_base_angle'] / 180)
            self._robot.write_joint_position_limit_to_sim(self._robot.data.soft_joint_pos_limits[:, joint_index].unsqueeze(1), [joint_index])
        
        """================ Init Object Default Params ================"""
        # ArticulationObject: init object root, joint, grasp states
        if isinstance(self._object, Articulation):
            # set object joint_limit for chair
            if self.cfg.object_infos['group'] in ['chair'] and 'PRISMATIC' in self._object_model.obj_joint_types:
                joint_index = self._object.find_joints(self._object_model.obj_joint_names[self._object_model.obj_joint_types.index('PRISMATIC')])[0][0]
                self._object.data.soft_joint_pos_limits[:, joint_index, 1] = self._object.data.soft_joint_pos_limits[:, joint_index, 0] + 0.001
                self._object.write_joint_position_limit_to_sim(self._object.data.soft_joint_pos_limits[:, joint_index].unsqueeze(1), [joint_index])
            
            # set object joint_limit for faucet
            if self.cfg.object_infos['group'] in ['faucet']:
                joint_index = self._object.find_joints(self.object_grasp_joint_name)[0][0]
                if self._object.data.soft_joint_pos_limits[0, joint_index, 1] - self._object.data.soft_joint_pos_limits[0, joint_index, 0] > torch.pi * 2 / 3:
                    self._object.data.soft_joint_pos_limits[:, joint_index, 0] = (self._object.data.soft_joint_pos_limits[:, joint_index, 1] + self._object.data.soft_joint_pos_limits[:, joint_index, 0]) / 2
                    self._object.write_joint_position_limit_to_sim(self._object.data.soft_joint_pos_limits[:, joint_index].unsqueeze(1), [joint_index])
            
            # set object joint_limit for bucket and kettle
            if self.cfg.object_infos['group'] in ['bucket', 'kettle']:
                joint_index = self._object.find_joints(self.object_grasp_joint_name)[0][0]
                shift_angle = (self._object.data.soft_joint_pos_limits[:, joint_index, 1] - self._object.data.soft_joint_pos_limits[:, joint_index, 0]).clone() * 1 / 3
                self._object.data.soft_joint_pos_limits[:, joint_index, 0] = self._object.data.soft_joint_pos_limits[:, joint_index, 0] + shift_angle
                self._object.data.soft_joint_pos_limits[:, joint_index, 1] = self._object.data.soft_joint_pos_limits[:, joint_index, 1] - shift_angle
                self._object.write_joint_position_limit_to_sim(self._object.data.soft_joint_pos_limits[:, joint_index].unsqueeze(1), [joint_index])
            
            # init object default root state
            self.object_default_root_state = self._object.data.default_root_state.clone()
            self.object_default_root_state_w = self._object.data.default_root_state.clone()
            self.object_default_root_state_w[:, :3] += self.env_origins
            
            # init object default joint
            self.object_default_joint_pos = self._object.data.default_joint_pos.clone()
            self.object_default_joint_vel = torch.zeros_like(self.object_default_joint_pos)
            # init object default joint limits
            self.object_joint_lower_limits = self._object.data.soft_joint_pos_limits[0, :, 0]
            self.object_joint_upper_limits = self._object.data.soft_joint_pos_limits[0, :, 1]
            
            # init object_grasp_body_idx, object_grasp_joint_idx
            self.object_grasp_body_idx = self._object.find_bodies(self.object_grasp_joint_link)[0][0]
            self.object_grasp_joint_idx = self._object.find_joints(self.object_grasp_joint_name)[0][0] if self.object_grasp_joint_name in self._object.joint_names else None
            # open articulated object
            if self.cfg.action_type == 'open':
                # define local grasp pos and rot relative to object grasp body pose
                self.object_local_grasp_pos = self.object_closed_handle_pose[0:3].repeat((self.num_envs, 1))
                self.object_local_grasp_rot = self.object_closed_handle_pose[3:7].repeat((self.num_envs, 1))
                # define local grasp pos_goal and rot_goal relative to object grasp body pose
                self.object_local_grasp_pos_goal = self.object_opened_handle_pose[0:3].repeat((self.num_envs, 1))
                self.object_local_grasp_rot_goal = self.object_opened_handle_pose[3:7].repeat((self.num_envs, 1))
            # close articulated object
            elif self.cfg.action_type == 'close':
                # define local grasp pos and rot relative to object grasp body pose
                self.object_local_grasp_pos = self.object_closed_handle_pose[0:3].repeat((self.num_envs, 1))
                self.object_local_grasp_rot = self.object_closed_handle_pose[3:7].repeat((self.num_envs, 1))
                # set object close config for cart, chair
                if self.cfg.object_infos['group'] in ['cart', 'chair']:
                    # define local grasp pos_goal and rot_goal relative to object grasp body pose
                    self.object_local_grasp_pos_goal = self.object_local_grasp_pos - (self.object_opened_handle_pose[0:3].repeat((self.num_envs, 1)) - self.object_local_grasp_pos)
                    self.object_local_grasp_rot_goal = self.object_opened_handle_pose[3:7].repeat((self.num_envs, 1))
                else:
                    # define local grasp pos_goal and rot_goal relative to object grasp body pose
                    self.object_local_grasp_pos_goal = self.object_closed_handle_pose[0:3].repeat((self.num_envs, 1))
                    self.object_local_grasp_rot_goal = self.object_closed_handle_pose[3:7].repeat((self.num_envs, 1))
            
            # combine local object grasp pose with global object grasp body pose
            self.object_grasp_rot, self.object_grasp_pos = combine_transformation(
                self._object.data.body_link_quat_w[:, self.object_grasp_body_idx],
                self._object.data.body_link_pos_w[:, self.object_grasp_body_idx] - self.env_origins,
                self.object_local_grasp_rot, self.object_local_grasp_pos
            )
            # combine local object grasp pose goal with global object grasp body pose
            self.object_grasp_rot_goal, self.object_grasp_pos_goal = combine_transformation(
                self._object.data.body_link_quat_w[:, self.object_grasp_body_idx].clone(), 
                self._object.data.body_link_pos_w[:, self.object_grasp_body_idx].clone() - self.env_origins,
                self.object_local_grasp_rot_goal, self.object_local_grasp_pos_goal
            )
            
            # define local grasp pos_close and rot_close relative to object grasp body pose
            self.object_local_grasp_pos_close = self.object_closed_handle_pose[0:3].repeat((self.num_envs, 1))
            self.object_local_grasp_rot_close = self.object_closed_handle_pose[3:7].repeat((self.num_envs, 1))
            # define local grasp pos_open and rot_open relative to object grasp body pose
            self.object_local_grasp_pos_open = self.object_opened_handle_pose[0:3].repeat((self.num_envs, 1))
            self.object_local_grasp_rot_open = self.object_opened_handle_pose[3:7].repeat((self.num_envs, 1))
            # combine local object grasp pose close with global object grasp body pose
            self.object_grasp_rot_close, self.object_grasp_pos_close = combine_transformation(
                self._object.data.body_link_quat_w[:, self.object_grasp_body_idx],
                self._object.data.body_link_pos_w[:, self.object_grasp_body_idx] - self.env_origins,
                self.object_local_grasp_rot_close, self.object_local_grasp_pos_close
            )
            # combine local object grasp pose open with global object grasp body pose
            self.object_grasp_rot_open, self.object_grasp_pos_open = combine_transformation(
                self._object.data.body_link_quat_w[:, self.object_grasp_body_idx].clone(), 
                self._object.data.body_link_pos_w[:, self.object_grasp_body_idx].clone() - self.env_origins,
                self.object_local_grasp_rot_open, self.object_local_grasp_pos_open
            )
            # define default object grasp pos open_close_shift
            self.object_grasp_pos_open_close_shift = self.object_grasp_pos_open - self.object_grasp_pos_close
            
            # set object grasp_pos_goal for bucket and kettle
            if self.cfg.object_infos['group'] in ['bucket', 'kettle']:
                self.object_grasp_pos_goal[:, -1] += 0.2
        
        # RigidObject: init object root, grasp states
        elif isinstance(self._object, RigidObject):
            # init object default root state
            self.object_default_root_state = self._object.data.default_root_state.clone()
            self.object_default_root_state_w = self._object.data.default_root_state.clone()
            self.object_default_root_state_w[:, :3] += self.env_origins
            
            # goal_distance
            self.goal_distance = 0.2
            # update object_grasp_pos
            self.object_grasp_pos = self._object.data.body_com_state_w[:, 0, :3].clone() - self.env_origins
            self.object_grasp_rot = self._object.data.body_com_state_w[:, 0, 3:7].clone()
            # update object_grasp_pos_goal
            self.object_grasp_pos_goal = self._object.data.body_com_state_w[:, 0, :3].clone() - self.env_origins
            self.object_grasp_pos_goal[:, -1] += self.goal_distance
            self.object_grasp_rot_goal = self.object_grasp_rot.clone()
        
        """================ Reset Params ================"""
        # reset and compute_intermediate_values
        self._reset_idx(env_ids=None, random=False)
        
        # ArticulationObject: center object grasp point
        if isinstance(self._object, Articulation):
            # center object_grasp_pos to origin
            object_shift_vector = -self.object_grasp_pos
            object_shift_vector[:, -1] *= 0.0
            self.object_default_root_state[:, :3] += object_shift_vector
            self.object_default_root_state_w = self.object_default_root_state.clone()
            self.object_default_root_state_w[:, :3] += self.env_origins
            # update object_grasp_pos and object_grasp_pos_goal
            self.object_grasp_pos += object_shift_vector
            self.object_grasp_pos_goal += object_shift_vector
            # init object_default_grasp_direction
            self.object_default_grasp_direction = torch.tensor([0.0, 1.0, 0.0]).repeat((self.num_envs, 1)).to(self.device)
            self.object_grasp_direction = self.object_default_grasp_direction.clone()
            # init object_default_goal_direction
            self.object_default_goal_direction = torch.nn.functional.normalize(self.object_grasp_pos_goal - self.object_grasp_pos, dim=1).clone()
            self.object_goal_direction = self.object_default_goal_direction.clone()
            # init object_default_goal_distance
            self.object_default_goal_distance = torch.norm(self.object_opened_handle_pose[0:3].repeat((self.num_envs, 1)) - self.object_closed_handle_pose[0:3].repeat((self.num_envs, 1)), dim=-1).clone()
            # init object_default_palm_shift
            self.object_default_palm_shift = self.robot_palm_pos - self.object_grasp_pos
        # RigidObject: center object
        elif isinstance(self._object, RigidObject):
            # TODO: init object_default_grasp_direction
            self.object_default_grasp_direction = torch.tensor([0.0, 0.0, -1.0]).repeat((self.num_envs, 1)).to(self.device)
            self.object_grasp_direction = self.object_default_grasp_direction.clone()
            
            if 'ycb_grasp_direction' in self.cfg.control_params:
                self.object_default_grasp_direction = torch.tensor(self.cfg.control_params['ycb_grasp_direction']).repeat((self.num_envs, 1)).to(self.device)
                self.object_grasp_direction = self.object_default_grasp_direction.clone()
            
            # init object_default_goal_direction
            self.object_default_goal_direction = torch.nn.functional.normalize(self.object_grasp_pos_goal - self.object_grasp_pos, dim=1).clone()
            self.object_goal_direction = self.object_default_goal_direction.clone()
            # init object_default_goal_distance
            self.object_default_goal_distance = torch.tensor([self.goal_distance]).repeat(self.num_envs).to(self.device)
            # init object_default_palm_shift
            self.object_default_palm_shift = self.robot_palm_pos - self.object_grasp_pos
        
        """================ Init Robot Targets ================"""
        # init robot_pos_goal
        self.robot_pos_goal = self.robot_palm_pos.clone()
        # init robot_rot_goal
        rot_goal = torch.zeros((self.num_envs, 3)).to(self.device)
        self.robot_rot_goal = quaternion_multiply(euler_angle_to_quaternion(rot_goal), self.robot_palm_rot.clone())
        # init robot_joint_goal
        self.robot_joint_pos_norm_goal = normalize_lower_upper(self.robot_default_joint_pos, self.robot_joint_lower_limits, self.robot_joint_upper_limits)
        
        # init robot_rotate_palm_rot: rotate around y for 90 degrees
        rotate_palm_rot = torch.zeros((self.num_envs, 3)).to(self.device)
        rotate_palm_rot[:, 1] = torch.pi * 90 / 180
        self.robot_rotate_palm_rot = quaternion_multiply(euler_angle_to_quaternion(rotate_palm_rot), self.robot_palm_rot.clone())
        
        """================ Init Robot Controller ================"""
        # init ik controller
        diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self.diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=self.num_envs, device=self.device)
    
    
    def _update_robot_palm_pos(self):
        # update robot_palm_pos for xhand_robot
        if self.cfg.robot_name == 'xhand_robot':
            shift_vector = torch.zeros_like(self.robot_palm_pos)
            shift_vector[..., 0] = -0.04
            shift_vector[..., 2] = 0.06
            self.robot_palm_pos += quaternion_rotate_vector(self.robot_palm_rot, shift_vector)
    
    def _update_robot_finger_pos(self):
        # update robot_finger_pos for xhand_robot
        if self.cfg.robot_name == 'xhand_robot':
            shift_vector = torch.zeros_like(self.robot_finger_pos)
            shift_vector[..., :4, 1] = 0.01
            shift_vector[..., :4, 2] = -0.03
            shift_vector[..., 4, 0] = 0.03
            shift_vector[..., 4, 2] = -0.01
            for n in range(len(self.robot_finger_body_indices)): self.robot_finger_pos[:, n] += quaternion_rotate_vector(self.robot_finger_rot[:, n], shift_vector[:, n])
    
    def _update_robot_palm_direction(self):
        # update robot_palm_direction for g1_robot
        if self.cfg.robot_name == 'g1_robot':
            shift_vector = torch.zeros_like(self.robot_palm_pos)
            shift_vector[..., 2] = 1.0
            self.robot_palm_direction = torch.nn.functional.normalize(quaternion_rotate_vector(self.robot_palm_rot, shift_vector), dim=1)
        # update robot_palm_direction for fetch_robot
        if self.cfg.robot_name == 'fetch_robot':
            shift_vector = torch.zeros_like(self.robot_palm_pos)
            shift_vector[..., 0] = 1.0
            self.robot_palm_direction = torch.nn.functional.normalize(quaternion_rotate_vector(self.robot_palm_rot, shift_vector), dim=1)
        # update robot_palm_direction for xhand_robot
        if self.cfg.robot_name == 'xhand_robot':
            shift_vector = torch.zeros_like(self.robot_palm_pos)
            shift_vector[..., 2] = 1.0
            self.robot_palm_direction = torch.nn.functional.normalize(quaternion_rotate_vector(self.robot_palm_rot, shift_vector), dim=1)
    
    
    def _compute_intermediate_values(self):
        """================ Update Robot States ================"""
        # update robot_root_state in local frame
        self.robot_root_pos = self._robot.data.root_link_state_w[:, :3] - self.env_origins
        self.robot_root_rot = self._robot.data.root_link_state_w[:, 3:7]
        self.robot_root_rot_euler = quaternion_to_euler_angle(self._robot.data.root_link_state_w[:, 3:7])
        self.robot_root_vel = self._robot.data.root_link_state_w[:, 7:10]
        self.robot_root_ang = self._robot.data.root_link_state_w[:, 10:13]
        
        # update robot_body_state in local frame
        self.robot_body_pos = self._robot.data.body_link_state_w[:, :, :3] - self.env_origins.unsqueeze(1)
        self.robot_body_rot = self._robot.data.body_link_state_w[:, :, 3:7]
        self.robot_body_rot_euler = quaternion_to_euler_angle(self._robot.data.body_link_state_w[:, :, 3:7].reshape(-1, 4)).reshape((-1, self.num_bodys, 3))
        self.robot_body_vel = self._robot.data.body_link_state_w[:, :, 7:10]
        self.robot_body_ang = self._robot.data.body_link_state_w[:, :, 10:13]
        
        # update robot_base_state in local_frame
        self.robot_base_pos = self.robot_body_pos[:, self.robot_base_body_index].clone()
        self.robot_base_rot = self.robot_body_rot[:, self.robot_base_body_index].clone()
        self.robot_base_rot_euler = self.robot_body_rot_euler[:, self.robot_base_body_index].clone()
        self.robot_base_vel = self.robot_body_vel[:, self.robot_base_body_index].clone()
        self.robot_base_ang = self.robot_body_ang[:, self.robot_base_body_index].clone()
        
        # update robot_hand_state in local_frame
        self.robot_hand_pos = self.robot_body_pos[:, self.robot_hand_body_index].clone()
        self.robot_hand_rot = self.robot_body_rot[:, self.robot_hand_body_index].clone()
        self.robot_hand_rot_euler = self.robot_body_rot_euler[:, self.robot_hand_body_index].clone()
        self.robot_hand_vel = self.robot_body_vel[:, self.robot_hand_body_index].clone()
        self.robot_hand_ang = self.robot_body_ang[:, self.robot_hand_body_index].clone()
        
        # update robot_palm_state in local frame
        self.robot_palm_body_index = self.robot_hand_body_index
        self.robot_palm_pos = self.robot_body_pos[:, self.robot_palm_body_index].clone()
        self.robot_palm_rot = self.robot_body_rot[:, self.robot_palm_body_index].clone()
        self.robot_palm_rot_euler = self.robot_body_rot_euler[:, self.robot_palm_body_index].clone()
        self.robot_palm_vel = self.robot_body_vel[:, self.robot_palm_body_index].clone()
        self.robot_palm_ang = self.robot_body_ang[:, self.robot_palm_body_index].clone()
        
        # update robot_finger_state in local frame
        self.robot_finger_pos = self.robot_body_pos[:, self.robot_finger_body_indices].clone()
        self.robot_finger_rot = self.robot_body_rot[:, self.robot_finger_body_indices].clone()
        self.robot_finger_rot_euler = self.robot_body_rot_euler[:, self.robot_finger_body_indices].clone()
        self.robot_finger_vel = self.robot_body_vel[:, self.robot_finger_body_indices].clone()
        self.robot_finger_ang = self.robot_body_ang[:, self.robot_finger_body_indices].clone()
        
        # update robot_joint_state
        self.robot_joint_pos = self._robot.data.joint_pos
        self.robot_joint_vel = self._robot.data.joint_vel
        self.robot_joint_acc = self._robot.data.joint_acc
        self.robot_joint_pos_norm = normalize_lower_upper(self.robot_joint_pos, self.robot_joint_lower_limits, self.robot_joint_upper_limits)
        
        # update robot_palm_pos
        self._update_robot_palm_pos()
        # update robot_finger_pos
        self._update_robot_finger_pos()
        # update robot_palm_direction
        self._update_robot_palm_direction()
        
        """================ Update Object States ================"""
        # ArticulationObject: update object joint, grasp states
        if isinstance(self._object, Articulation):
            # combine local object grasp pose with global object link pose
            self.object_grasp_rot, self.object_grasp_pos = combine_transformation(
                self._object.data.body_link_quat_w[:, self.object_grasp_body_idx],
                self._object.data.body_link_pos_w[:, self.object_grasp_body_idx] - self.env_origins,
                self.object_local_grasp_rot, self.object_local_grasp_pos
            )
            self.object_grasp_rot_euler = quaternion_to_euler_angle(self.object_grasp_rot)
            # update object velocity
            self.object_grasp_vel = self._object.data.body_link_state_w[:, self.object_grasp_body_idx, 7:10]
            self.object_grasp_ang = self._object.data.body_link_state_w[:, self.object_grasp_body_idx, 10:13]
            # update object_goal_direction and object_goal_distance
            self.object_goal_direction = torch.nn.functional.normalize(self.object_grasp_pos_goal - self.object_grasp_pos, dim=1)
            self.object_goal_distance = torch.norm(self.object_grasp_pos_goal - self.object_grasp_pos, dim=-1)
            
            # update object_joint_state
            self.object_joint_pos = self._object.data.joint_pos
            self.object_joint_vel = self._object.data.joint_vel
            self.object_joint_acc = self._object.data.joint_acc
            self.object_joint_pos_norm = normalize_lower_upper(self.object_joint_pos, self.object_joint_lower_limits, self.object_joint_upper_limits)
        
        # RigidObject: update object grasp states
        elif isinstance(self._object, RigidObject):
            # update object grasp pose in local frame
            self.object_grasp_pos = self._object.data.body_com_state_w[:, 0, :3] - self.env_origins
            self.object_grasp_rot = self._object.data.body_com_state_w[:, 0, 3:7]
            self.object_grasp_rot_euler = quaternion_to_euler_angle(self.object_grasp_rot)
            self.object_grasp_vel = self._object.data.body_com_state_w[:, 0, 7:10]
            self.object_grasp_ang = self._object.data.body_com_state_w[:, 0, 10:13]
            # update object_goal_direction and object_goal_distance
            self.object_goal_direction = torch.nn.functional.normalize(self.object_grasp_pos_goal - self.object_grasp_pos, dim=1)
            self.object_goal_distance = torch.norm(self.object_grasp_pos_goal - self.object_grasp_pos, dim=-1)
        
        """================ Update Robot-Object States ================"""
        # distance from hand palm to object mesh
        self.dist_hand_object = batch_sided_distance(self.robot_palm_pos.unsqueeze(1), self.object_grasp_pos.unsqueeze(1))
        # distance from hand body to object mesh
        self.dist_hand_body_object = batch_sided_distance(self.robot_body_pos[:, self.robot_active_body_indices], self.object_grasp_pos.unsqueeze(1))
        # self.dist_hand_body_object = batch_sided_distance(self.robot_body_pos, self.object_grasp_pos.unsqueeze(1))
        # distance from hand finger thumb to object mesh
        self.dist_hand_thumb_object = batch_sided_distance(self.robot_finger_pos[:, -1].unsqueeze(1), self.object_grasp_pos.unsqueeze(1))
        # distance from hand finger to object mesh
        self.dist_hand_finger_object = batch_sided_distance(self.robot_finger_pos, self.object_grasp_pos.unsqueeze(1))
        # distance from hand thumb_finger to object mesh
        self.dist_hand_thumb_finger_object = batch_sided_distance((self.robot_finger_pos[:, :-1] + self.robot_finger_pos[:, -1].unsqueeze(1)) / 2, self.object_grasp_pos.unsqueeze(1))
        
        """================ Update Markers ================"""
        # visualize control, handle, and goal markers
        self.handle_markers.visualize(self.object_grasp_pos + self.env_origins, self.object_grasp_rot)
        self.goal_markers.visualize(self.object_grasp_pos_goal + self.env_origins, self.object_grasp_rot_goal)
        self.control_markers.visualize(self.env_origins)
        
        """================ Record Trajectory ================"""
        # record camera images
        if self.cfg.record_trajectory and not self.cfg.exit_flag:
            # update camera states
            self.camera_head_pos = self._camera_head.data.pos_w.clone() - self.env_origins
            self.camera_head_rot = self._camera_head.data.quat_w_world.clone()
            self.camera_head_rot_euler = quaternion_to_euler_angle(self.camera_head_rot)
            self.camera_arm_pos = self._camera_arm.data.pos_w.clone() - self.env_origins
            self.camera_arm_rot = self._camera_arm.data.quat_w_world.clone()
            self.camera_arm_rot_euler = quaternion_to_euler_angle(self.camera_arm_rot)
            # disable markers
            self.control_markers.set_visibility(visible=False)
            self.handle_markers.set_visibility(visible=False)
            self.goal_markers.set_visibility(visible=False)
    
    
    def _setup_scene(self):
        """================ Load Robot Configs ================"""
        # rotate_base
        if 'rotate_base' in self.cfg.control_params and self.cfg.control_params['rotate_base']:
            # locate rotate_base usd
            self.cfg.robot_cfg.spawn.usd_path = self.cfg.robot_cfg.spawn.usd_path.replace('/{}/'.format(self.cfg.robot_name), '/{}_rotate/'.format(self.cfg.robot_name)) # type: ignore
            # update slider actuators
            self.cfg.robot_cfg.actuators.pop('slider')
            self.cfg.robot_cfg.actuators['slider_x'] = ImplicitActuatorCfg(joint_names_expr=["slider_basex"], stiffness=self.cfg.control_params['slider_basex_joint_stiffness'], damping=self.cfg.control_params['slider_basex_joint_damping'])
            self.cfg.robot_cfg.actuators['slider_y'] = ImplicitActuatorCfg(joint_names_expr=["slider_basey"], stiffness=self.cfg.control_params['slider_basey_joint_stiffness'], damping=self.cfg.control_params['slider_basey_joint_damping'])
        
        """================ Load Object Configs ================"""
        # load object_class
        self.object_class = 'Articulation'
        if self.cfg.object_infos['type'] in ['ycb']:
            # update object_class as Rigid
            self.object_class = 'Rigid'
            self.object_joint_handle_name = None
            # update object_cfg as rigid_object_cfg
            self.cfg.object_cfg = self.cfg.rigid_object_cfg # type: ignore
            self.cfg.object_cfg.spawn.usd_path = self.cfg.object_usd_path # type: ignore
            # update object mass and scale
            if 'ycb_mass' in self.cfg.control_params: SCENE_ANALYSIS['object'][self.cfg.object_infos['type']][self.cfg.object_infos['group']]['mass'] = self.cfg.control_params['ycb_mass']
            if 'ycb_scale' in self.cfg.control_params: SCENE_ANALYSIS['object'][self.cfg.object_infos['type']][self.cfg.object_infos['group']]['scale'] = self.cfg.control_params['ycb_scale']
            SCENE_ANALYSIS['object'][self.cfg.object_infos['type']][self.cfg.object_infos['group']]['scale'] = float(YCB_ANALYSIS[self.cfg.object_infos['group']][self.cfg.object_infos['name']].split('/')[0].split('_')[1])
            # update robot_cfg with rotate_init_palm
            if 'rotate_init_palm' in self.cfg.control_params and self.cfg.control_params['rotate_init_palm']:
                def _rotate_init_palm(robot_cfg):
                    if self.cfg.robot_name == 'g1_robot':
                        robot_cfg.init_state.joint_pos['idx67_arm_r_joint7'] = 1.5
                _rotate_init_palm(self.cfg.robot_cfg)
        
        # load object_model_class
        if self.cfg.object_infos['type'] == 'partnet': self._object_model_class = PartNetObjModel
        elif self.cfg.object_infos['type'] == 'unidoor': self._object_model_class = UniDoorObjModel
        elif self.cfg.object_infos['type'] == 'ycb': self._object_model_class = YCBObjModel
        else: raise ValueError("Unknown Object Type: {}".format(self.cfg.object_infos['type']))
        
        # load object model
        self._object_model = self._object_model_class(
            obj_id=self.cfg.object_infos['name'].split('/')[0], 
            obj_cat=self.cfg.object_infos['group'],
            target_open_ratio=self.cfg.object_infos['open_ratio'],
            target_handle_direction=-np.array(self.cfg.heading_direction))
        # load object type, group, name
        self.object_type = self.cfg.object_infos['type']
        self.object_group = self.cfg.object_infos['group']
        self.object_name = self.cfg.object_infos['name'].split('/')[0]
        # ArticulationObject: load object grasp joint name, type, link, handle
        if self.object_class == 'Articulation':
            self.object_grasp_joint_name = self.cfg.object_infos['name'].split('/')[1]
            self.object_grasp_joint_type = self.cfg.object_infos['name'].split('/')[2]
            self.object_grasp_joint_link = self.cfg.object_infos['name'].split('/')[3]
            self.object_grasp_joint_handle_name = self.cfg.object_infos['name'].split('/')[4]
            # load object handle pose relative to object_grasp_joint_link
            self.object_joint_handle_name = '{}_{}'.format(self.object_grasp_joint_name, self.object_grasp_joint_handle_name)
            self.object_closed_handle_pose = torch.tensor(self._object_model.obj_joint_handle_infos[self.object_joint_handle_name]['closed_handle_pos'].tolist() + [1., 0., 0., 0.]).to(self.device)
            self.object_opened_handle_pose = torch.tensor(self._object_model.obj_joint_handle_infos[self.object_joint_handle_name]['opened_handle_pos'].tolist() + [1., 0., 0., 0.]).to(self.device)
        
        # load object SCENE_ANALYSIS
        self._object_scene_infos = SCENE_ANALYSIS['object'][self.cfg.object_infos['type']][self.cfg.object_infos['group']]
        # set object scale
        self.cfg.object_infos['scale'] = self._object_scene_infos['scale']
        if 'object_scale' in self.cfg.control_params: self.cfg.object_infos['scale'] += self.cfg.control_params['object_scale']
        # set object mass_props
        self.cfg.object_cfg.spawn.mass_props.mass = self._object_scene_infos['mass'] # type: ignore
        # set object sim gravity
        self.cfg.object_cfg.spawn.rigid_props.disable_gravity = not self._object_scene_infos['gravity'] # type: ignore
        # set object higher height
        self.cfg.control_params['higher_random_object'] = self._object_scene_infos['init_height']
        # ArticulationObject: set object init state
        if self.object_class == 'Articulation':
            # set object joint lower pos for init
            self._object_model.obj_joint_link_infos[self.object_grasp_joint_name]['lower'] = min(
                self._object_model.obj_joint_link_infos[self.object_grasp_joint_name]['lower'] + np.pi * self._object_scene_infos['init_joint'] / 180, self._object_model.obj_joint_link_infos[self.object_grasp_joint_name]['upper'])
            # enable random height for windows and tabletop objects
            if self.cfg.object_infos['group'] not in ['window'] and self._object_scene_infos['place'] != 'tabletop':
                self.cfg.control_params['height_random_object'] = 0.0
            # zero friction and joint damping for sliding objects
            if self.cfg.object_infos['group'] in ['cart', 'chair']:
                self.cfg.sim.physics_material.static_friction, self.cfg.sim.physics_material.dynamic_friction = 0.0, 0.0
                self.cfg.object_cfg.actuators["joints"].stiffness, self.cfg.object_cfg.actuators["joints"].damping = 0.0, 0.0        
            # scale handle pos
            self.object_closed_handle_pose[:3] *= self.cfg.object_infos['scale']
            self.object_opened_handle_pose[:3] *= self.cfg.object_infos['scale']
        # scale object scale
        self.cfg.object_cfg.spawn.scale = (self.cfg.object_infos['scale'], self.cfg.object_infos['scale'], self.cfg.object_infos['scale']) # type: ignore
        # process object_cfg
        self._object_model.obj_scale = self.cfg.object_infos['scale']
        self._object_model.process_obj_sim_config(self.cfg.object_cfg, self.object_joint_handle_name)
        
        # set higher random object
        if 'higher_random_object' in self.cfg.control_params:
            self.cfg.object_cfg.init_state.pos = (self.cfg.object_cfg.init_state.pos[0], self.cfg.object_cfg.init_state.pos[1], self.cfg.object_cfg.init_state.pos[2] + self.cfg.control_params['higher_random_object'])
        
        """================ Load Stage & Ground Configs ================"""
        # record robot trajectory: make stage and ground invisible
        if self.cfg.record_trajectory: self.cfg.stage_cfg.spawn.visible, self.cfg.ground_cfg.spawn.visible = False, False # type: ignore        
        # set stage default state
        self.stage_default_root_state = torch.zeros((self.num_envs, 13)).to(self.device)
        self.stage_default_root_state[:, 2], self.stage_default_root_state[:, 3] = -0.5, 1.0
        # set stage default size
        self.cfg.stage_cfg.spawn.size = (self.cfg.room_space, self.cfg.room_space / 2, 1.0) # type: ignore
        # set stage shift_vector
        self.stage_shift_vector = torch.zeros((self.num_envs, 3)).to(self.device)
        self.stage_shift_vector[:, :2] = torch.tensor(self._object_model.obj_center_rotated[:2] * self.cfg.object_infos['scale']).to(self.device)
        self.stage_shift_vector[:, 1] += (self.cfg.stage_cfg.spawn.size[1] - self._object_model.obj_size_rotated[1] * self.cfg.object_infos['scale']) / 2 # type: ignore
        # set stage higher for windows and tabletop objects
        self.stage_default_root_state[:, 2] += self.cfg.control_params['higher_random_object']
        # load stage
        self._stage = RigidObject(self.cfg.stage_cfg)
        
        # set ground default state
        self.ground_default_root_state = torch.zeros((self.num_envs, 13)).to(self.device)
        self.ground_default_root_state[:, 2], self.ground_default_root_state[:, 3] = 0.0, 1.0
        # set ground default size
        self.cfg.ground_cfg.spawn.size = (self.cfg.room_space, self.cfg.room_space, 0.02) # type: ignore
        # set ground shift_vector
        self.ground_shift_vector = torch.zeros((self.num_envs, 3)).to(self.device)
        self.ground_shift_vector[:, 1] += (self.cfg.stage_cfg.spawn.size[1] - self.cfg.room_space) / 2 # type: ignore
        # load ground
        self._ground = RigidObject(self.cfg.ground_cfg)
        
        # load terrain with num_envs and env_spacing
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        """================ Load Robot & Object Scene ================"""
        # load robot
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot
        
        # record robot trajectory
        if self.cfg.record_trajectory:
            # load cameras
            self._camera_head = Camera(self.cfg.camera_head_cfg)
            self.scene.sensors["camera_head"] = self._camera_head
            self._camera_arm = Camera(self.cfg.camera_arm_cfg)
            self.scene.sensors["camera_arm"] = self._camera_arm
            # # remove stage and ground
            # self.stage_default_root_state[:, 2], self.ground_default_root_state[:, 2] = -1.0, -1.0
            # zero random object init
            self.cfg.control_params['zero_random_shift'], self.cfg.control_params['zero_random_height'] = True, True
            if self.cfg.object_infos['group'] in ['window']: self.cfg.control_params['zero_random_height'] = False
            # load room_model and room_infos
            self._room_model = RoomModel(self.cfg.room_id, self.cfg.record_split, self._object_model)
            self.cfg.room_infos = self._room_model.process_obj_room_infos()
            self.cfg.room_cfg.usd_path = self.cfg.room_infos['usd_path']
            self.cfg.room_cfg.scale = (self.cfg.room_infos['scale'], self.cfg.room_infos['scale'], self.cfg.room_infos['scale'])
            self._room = self.cfg.room_cfg.func("/World/Room", self.cfg.room_cfg, translation=self.cfg.room_infos['translation'], orientation=self.cfg.room_infos['orientation'])
            # load tabletop object and stage height in room
            if self.cfg.room_infos['place'] == 'tabletop':
                self.cfg.object_cfg.init_state.pos = (self.cfg.object_cfg.init_state.pos[0], self.cfg.object_cfg.init_state.pos[1], self.cfg.object_cfg.init_state.pos[2] - self.cfg.control_params['higher_random_object'] + self.cfg.room_infos['height'])
                self.stage_default_root_state[:, 2] = self.stage_default_root_state[:, 2] - self.cfg.control_params['higher_random_object'] + self.cfg.room_infos['height']
            # make terrain invisible
            UsdGeom.Imageable(omni.usd.get_context().get_stage().GetPrimAtPath("/World/ground/terrain")).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            # save scene_infos
            dump_json(osp.join(self.cfg.log_dir, 'scene_infos.json'), {'task': self.cfg.action_type, 'object': self.object_group, 'room_infos': self.cfg.room_infos})
            # init episode_trajectory
            self.episode_trajectory = {}
        
        # load object
        if self.object_class == 'Articulation':
            self._object = Articulation(self.cfg.object_cfg)
            self.scene.articulations["object"] = self._object
        elif self.object_class == 'Rigid':
            self._object = RigidObject(self.cfg.object_cfg) # type: ignore
            self.scene.rigid_objects["object"] = self._object # type: ignore
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # load light
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
    
    # pre-physics step calls
    def _pre_physics_step(self, actions: torch.Tensor):
        # world action frame: ppo
        if self.cfg.action_frame == 'world':
            # clip actions
            self.actions = torch.nan_to_num(actions, nan=0.0).clamp(-1.0, 1.0).clone()
        # base action frame: vla
        elif self.cfg.action_frame == 'base':
            # clone actions
            self.actions = torch.nan_to_num(actions, nan=0.0).clone()
            # decode local action from local base frame to world frame
            action_pos_w, action_rot_w = decode_local_action_from_local_frame(
                self.actions[:, :3], self.actions[:, 3:6], self.robot_base_pos, self.robot_base_rot_euler)
            self.actions[:, :3], self.actions[:, 3:6] = action_pos_w.clone(), action_rot_w.clone()
        
        # init robot_joint_target
        self.robot_joint_target = self.robot_default_joint_pos.clone()
        
        """================ robot_joint_target for hand ================"""
        # compute robot_joint_target for hand joints
        if len(self.robot_hand_joint_indices) == self.actions[:, 6:].shape[-1]:
            self.robot_joint_target[:, self.robot_hand_joint_indices] = unnormalize_lower_upper(
                self.actions[:, 6:] * self.cfg.control_params['action_dof_scale'], self.robot_joint_lower_limits[self.robot_hand_joint_indices], self.robot_joint_upper_limits[self.robot_hand_joint_indices])
        else:
            self.robot_joint_target[:, self.robot_hand_joint_indices] = unnormalize_lower_upper(
                self.actions[:, -1:].repeat(1, len(self.robot_hand_joint_indices)) * self.cfg.control_params['action_dof_scale'], self.robot_joint_lower_limits[self.robot_hand_joint_indices], self.robot_joint_upper_limits[self.robot_hand_joint_indices])
        
        """================ robot_joint_target for arm ================"""
        # convert actions to robot_hand_goal
        self.robot_hand_pos_goal = self.actions[:, :3] * self.cfg.control_params['action_pos_scale'] + self.robot_hand_pos
        self.robot_hand_rot_goal = quaternion_multiply(euler_angle_to_quaternion(self.actions[:, 3:6] * self.cfg.control_params['action_rot_scale']), self.robot_hand_rot)
        # convert hand_pos_goal and hand_rot_goal to robot root frame
        robot_hand_pos_goal_root, robot_hand_rot_goal_root = subtract_frame_transforms(
            self.robot_root_pos, self.robot_root_rot, self.robot_hand_pos_goal, self.robot_hand_rot_goal
        )
        # convert hand_pos and hand_rot to robot root frame
        robot_hand_pos_root, robot_hand_rot_root = subtract_frame_transforms(
            self.robot_root_pos, self.robot_root_rot, self.robot_hand_pos, self.robot_hand_rot
        )
        # set hand_pos and hand_rot command
        # self.diff_ik_controller.reset()
        self.diff_ik_controller.set_command(torch.cat([robot_hand_pos_goal_root, robot_hand_rot_goal_root], dim=-1))
        # get hand_jacobian for ik controller
        robot_hand_jacobian = self._robot.root_physx_view.get_jacobians()[:, self.robot_hand_body_index-1, :, self.robot_arm_joint_indices]
        # compute robot_joint_target for arm
        self.robot_joint_target[:, self.robot_arm_joint_indices] = self.diff_ik_controller.compute(robot_hand_pos_root, robot_hand_rot_root, robot_hand_jacobian, self.robot_joint_pos[:, self.robot_arm_joint_indices])
        
        # """================ visualize control and hand markers ================"""
        # self.control_markers.visualize(self.robot_hand_pos_goal + self.env_origins, self.robot_hand_rot_goal)
    
    
    def _apply_action(self):
        # set robot joint position target
        self.robot_joint_target = self.robot_joint_target.clamp(self.robot_joint_lower_limits, self.robot_joint_upper_limits)
        self._robot.set_joint_position_target(self.robot_joint_target)
    
    
    # post-physics step calls
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # update env times
        self.env_times += 1
        self.global_time += 1
        # update intermediate values
        self._compute_intermediate_values()
        # check terminated
        hand_range = self.dist_hand_object.squeeze(1) > 5.0
        hand_velocity = torch.norm(self.robot_palm_vel, dim=-1) > 10.0
        object_velocity = torch.norm(self.object_grasp_vel, dim=-1) > 10.0
        terminated = (hand_range * 1. + hand_velocity * 1. + object_velocity * 1.) > 0.
        # check truncated
        time_range = self.episode_length_buf >= self.max_episode_length - 1
        success_count = self.env_success_count >= 1 / (self.cfg.sim.dt * self.cfg.decimation) # keep success for 1s
        truncated = (time_range * 1. + success_count * 1.) > 0.
        
        # update env_episodes
        self.env_episodes[truncated] += 1
        # update global_episode
        self.global_episode += torch.count_nonzero(truncated)
        # update success_tracker
        self.success_tracker['episode'] += torch.count_nonzero(truncated)
        self.success_tracker['success'] += torch.sum(self.env_success[truncated])
        self.success_tracker['success_flag'] += torch.sum(self.env_success_flag[truncated])
        self.success_tracker['success_ratio'] += torch.sum(self.env_success_ratio[truncated])
        return terminated, truncated
    
    
    def _get_rewards(self) -> torch.Tensor:
        # print('run _get_rewards')
        
        # init reward_dict and reward_names
        self.reward_dict = {}
        self.reward_names = ['diff_reach_pos_action', 'diff_reach_rot_action', 'diff_reach_joint_action', 'diff_move_pos_action', 'diff_move_rot_action',
                             'diff_hand_object', 'diff_hand_body_object', 'diff_hand_thumb_object', 'diff_hand_finger_object',
                             'diff_object_goal', 'bonus_grasp_object', 'bonus_object_goal']
        
        
        """================ Global Rewards ================"""
        # difference from hand to object_vertices
        self.reward_dict['diff_hand_object'] = torch.mean(self.dist_hand_object, dim=-1)
        # difference from hand body to object_vertices
        self.reward_dict['diff_hand_body_object'] = torch.mean(self.dist_hand_body_object, dim=-1)
        # difference from hand thumb to object_vertices
        self.reward_dict['diff_hand_thumb_object'] = torch.mean(self.dist_hand_thumb_object, dim=-1)
        # difference from hand finger to object_vertices
        self.reward_dict['diff_hand_finger_object'] = torch.mean(self.dist_hand_finger_object, dim=-1)
        
        # difference from object to object_pos_goal
        self.reward_dict['diff_object_goal'] = self.object_goal_distance.clone()
        if 'normalize_diff_object_goal' in self.cfg.control_params and self.cfg.control_params['normalize_diff_object_goal']:
            self.reward_dict['diff_object_goal'] /= self.object_default_goal_distance
        
        """================ Global Flags ================"""
        # update env_success, env_success_flag and env_success_count
        self.env_success = (self.object_goal_distance < self.cfg.success_threshold)  * 1.0
        self.env_success_flag = torch.where(self.env_success == 1.0, torch.ones_like(self.env_success_flag), self.env_success_flag)
        self.env_success_ratio = (self.object_default_goal_distance - self.object_goal_distance) / self.object_default_goal_distance
        self.env_success_count += self.env_success
        
        # update grasp_object_flag
        hand_grasp_object_flag = self.reward_dict['diff_hand_object'] <= self.cfg.reward_params['dist_hand_object']
        thumb_grasp_object_flag = self.reward_dict['diff_hand_thumb_object'] <= self.cfg.reward_params['dist_thumb_object']
        finger_grasp_object_flag = self.reward_dict['diff_hand_finger_object'] <= self.cfg.reward_params['dist_finger_object']
        grasp_object_flag = (hand_grasp_object_flag * 1.0 + thumb_grasp_object_flag * 1.0 + finger_grasp_object_flag * 1.0) == 3.0
        
        
        """================ Before Grasp Rewards ================"""
        # reach_flag
        reach_action_flag = hand_grasp_object_flag
        
        # difference from hand_pos_action to target_pos_action for reaching
        robot_reach_pos_action = self.object_grasp_pos - self.robot_palm_pos
        self.reward_dict['diff_reach_pos_action'] = torch.norm(robot_reach_pos_action.clamp(-1.0, 1.0) - self.actions[:, :3], dim=-1)
        self.reward_dict['diff_reach_pos_action'] = torch.where(reach_action_flag, 0.0 * self.reward_dict['diff_reach_pos_action'], 1.0 * self.reward_dict['diff_reach_pos_action'])
        
        # difference from hand_rot_action to target_rot_action for reaching
        robot_reach_rot_action = quaternion_to_euler_angle(quaternion_between_vectors(self.object_grasp_direction, self.robot_palm_direction))
        self.reward_dict['diff_reach_rot_action'] = torch.norm(robot_reach_rot_action.clamp(-1.0, 1.0) - self.actions[:, 3:6], dim=-1)
        self.reward_dict['diff_reach_rot_action'] = torch.where(reach_action_flag, 0.0 * self.reward_dict['diff_reach_rot_action'], 1.0 * self.reward_dict['diff_reach_rot_action'])
        
        # difference from hand_joint_action to target_joint_action for reaching
        robot_reach_joint_action = self.robot_joint_pos_norm_goal[:, self.robot_hand_joint_indices]
        self.reward_dict['diff_reach_joint_action'] = torch.norm(robot_reach_joint_action.clamp(-1.0, 1.0) - self.actions[:, 6:], dim=-1)
        self.reward_dict['diff_reach_joint_action'] = torch.where(reach_action_flag, 0.0 * self.reward_dict['diff_reach_joint_action'], 1.0 * self.reward_dict['diff_reach_joint_action'])
        
        # TODO: rotate gripper/hand for ycb and refrigerator
        if self.cfg.object_infos['group'] in ['ycb', 'refrigerator'] and self.cfg.action_type == 'open' and 'diff_reach_palm_rot_action' in self.cfg.reward_params:
            robot_palm_rot_action = quaternion_to_euler_angle(quaternion_difference(self.robot_rotate_palm_rot, self.robot_palm_rot))
            self.reward_dict['diff_reach_palm_rot_action'] = torch.abs(robot_palm_rot_action.clamp(-1.0, 1.0) - self.actions[:, 3:6])[:, 1]
            self.reward_dict['diff_reach_palm_rot_action'] = torch.where(reach_action_flag, 0.0 * self.reward_dict['diff_reach_palm_rot_action'], 1.0 * self.reward_dict['diff_reach_palm_rot_action'])
            self.reward_names.append('diff_reach_palm_rot_action')
        
        """================ After Grasp Rewards ================"""
        # bonus for grasping object
        self.reward_dict['bonus_grasp_object'] = grasp_object_flag * 1.0
        
        # move_flag
        move_action_flag = grasp_object_flag
        
        # difference from hand_pos_action to target_pos_action for moving
        robot_move_pos_action = self.object_grasp_pos_goal - self.object_grasp_pos
        if 'normalize_diff_object_goal' in self.cfg.control_params and self.cfg.control_params['normalize_diff_object_goal']:
            robot_move_pos_action /= self.object_default_goal_distance.unsqueeze(-1)
        self.reward_dict['diff_move_pos_action'] = torch.norm(robot_move_pos_action.clamp(-1.0, 1.0) - self.actions[:, :3], dim=-1)
        self.reward_dict['diff_move_pos_action'] = torch.where(move_action_flag, 1.0 * self.reward_dict['diff_move_pos_action'], 0.0 * self.reward_dict['diff_move_pos_action'])
        
        # difference from hand_rot_action to target_rot_action for moving
        robot_move_rot_action = quaternion_to_euler_angle(quaternion_between_vectors(-self.object_goal_direction, self.robot_palm_direction))
        self.reward_dict['diff_move_rot_action'] = torch.norm(robot_move_rot_action.clamp(-1.0, 1.0) - self.actions[:, 3:6], dim=-1)
        self.reward_dict['diff_move_rot_action'] = torch.where(move_action_flag, 1.0 * self.reward_dict['diff_move_rot_action'], 0.0 * self.reward_dict['diff_move_rot_action'])
        
        # bonus for object reach goal
        self.reward_dict['bonus_object_goal'] = (self.object_goal_distance < 0.05)  * 1.0
        self.reward_dict['bonus_object_goal'] = torch.where(grasp_object_flag, 1.0 * self.reward_dict['bonus_object_goal'], 0.0 * self.reward_dict['bonus_object_goal'])
        
        
        """================ Compute Rewards ================"""
        # compute rewards
        rewards = torch.sum(torch.stack([self.cfg.reward_params[name] * self.reward_dict[name] for name in self.reward_names], dim=-1), dim=-1)
        
        # update reward_tracker
        if self.reward_tracker is None:
            self.reward_tracker = {'rewards': torch.mean(rewards)}
            for name in self.reward_names: self.reward_tracker[name] = torch.mean(self.reward_dict[name])
        else:
            self.reward_tracker['rewards'] += torch.mean(rewards)
            for name in self.reward_names: self.reward_tracker[name] += torch.mean(self.reward_dict[name])
        
        # log rewards
        if self.global_time % self.log_time == 0:
            log = 'Training steps: {:04d} x {:04d} x {:04d}; '.format(self.num_envs, self.global_time // self.log_time, self.log_time)
            log += 'Episode: {:04d}, '.format(self.success_tracker['episode'].to(torch.long).item())
            log += 'success: {:.3f}, '.format(self.success_tracker['success'].item() / self.success_tracker['episode'].item())
            log += 'success_flag: {:.3f}, '.format(self.success_tracker['success_flag'].item() / self.success_tracker['episode'].item())
            log += 'success_ratio: {:.3f}; '.format(self.success_tracker['success_ratio'].item() / self.success_tracker['episode'].item())
            log += 'Rewards= {:.3f}, '.format(1.0 + self.reward_tracker['rewards'] / self.log_time)
            for name in self.reward_names: log += '{}={:.3f}, '.format(name, self.reward_tracker[name] / self.log_time)
            # append and save log_tracker
            self.log_tracker.append(log)
            print(self.log_tracker[-1])
            save_list_strings(osp.join(self.log_dir, 'log.txt'), self.log_tracker)
            # reset reward_tracker
            self.reward_tracker = None
            # reset success_tracker
            self.success_tracker = {'episode': torch.tensor([1e-6], device=self.device), 
                                    'success': torch.tensor([0.0], device=self.device), 
                                    'success_flag': torch.tensor([0.0], device=self.device),
                                    'success_ratio': torch.tensor([0.0], device=self.device)}
        
        return torch.nan_to_num(rewards, nan=0.0).clamp(-10.0, 10.0).clone()
    
    
    def _reset_idx(self, env_ids: torch.Tensor | None, random=True):
        if env_ids is None: env_ids = self._robot._ALL_INDICES # type: ignore
        super()._reset_idx(env_ids) # type: ignore
        # print('run _reset_idx')
        
        # TODO: replay training trajectories
        replay = False and self.cfg.vla_mode
        if replay:
            try: replay_state_init = load_pickle(osp.join(self.log_dir.replace(self.log_dir.split('/')[-2], 'trajectories'), 'episode_{:03d}'.format(int(self.env_episodes[0] + self.cfg.record_exist)), 'state_infos.pkl'))['init']
            except: replay = False
        
        """================ Reset Room States ================"""
        # set room random root state
        if random and self.cfg.record_trajectory:
            # create random range
            random_x, random_y, random_ang = 0.2, 0.1, torch.pi * 0 / 180
            random_pos = torch.stack([torch.rand(1) * random_x - random_x / 2, torch.rand(1) * random_y - random_y / 2, torch.zeros(1)], dim=-1)
            random_rot = euler_angle_to_quaternion(torch.stack([torch.zeros(1), torch.zeros(1), torch.rand(1) * random_ang - random_ang / 2], dim=-1))
            # transfer room state
            room_pos = (random_pos + torch.tensor(self.cfg.room_infos['translation']).unsqueeze(0)).squeeze(0).numpy()
            room_rot = quaternion_to_euler_angle(quaternion_multiply(random_rot, torch.tensor(self.cfg.room_infos['orientation']).unsqueeze(0))).squeeze(0).numpy() * (180 / np.pi)
            # replay training trajectories
            if replay and random: room_pos, room_rot = replay_state_init['room']['room_pos'], replay_state_init['room']['room_rot']
            # set room state
            room_xform = UsdGeom.Xformable(omni.usd.get_context().get_stage().GetPrimAtPath("/World/Room"))
            room_xform.ClearXformOpOrder()
            room_xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(float(room_pos[0]), float(room_pos[1]), float(room_pos[2])))
            room_xform.AddRotateXYZOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(float(room_rot[0]), float(room_rot[1]), float(room_rot[2])))
            room_xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(float(self.cfg.room_infos['scale']), float(self.cfg.room_infos['scale']), float(self.cfg.room_infos['scale'])))
        
        
        """================ Reset Robot States ================"""
        # set robot random root state
        if random:
            # create random range
            random_x, random_y, random_ang = 0.5, 0.5, torch.pi * 30 / 180
            random_pos = torch.stack([torch.rand(len(env_ids)) * random_x - random_x / 2, torch.rand(len(env_ids)) * random_y, torch.zeros(len(env_ids))], dim=-1).to(self.device)
            random_rot = euler_angle_to_quaternion(torch.stack([torch.zeros(len(env_ids)), torch.zeros(len(env_ids)), torch.rand(len(env_ids)) * random_ang - random_ang / 2], dim=-1)).to(self.device)
            # transfer robot_default_root_state_w 
            robot_root_state_w = self.robot_default_root_state_w[env_ids].clone()
            robot_root_state_w[:, :3] += random_pos
            target_direction = -(robot_root_state_w[:, :3] - self.env_origins[env_ids])
            target_direction[:, -1] *= 0.0
            robot_root_state_w[:, 3:7] = quaternion_multiply(quaternion_between_vectors(target_direction, torch.tensor([0.0, 1.0, 0.0]).repeat((len(env_ids), 1)).to(self.device)), robot_root_state_w[:, 3:7])
            robot_root_state_w[:, 3:7] = quaternion_multiply(random_rot, robot_root_state_w[:, 3:7])
            # shift robot_root_state_w backward for close tasks
            if self.cfg.action_type == 'close' and self.cfg.object_infos['group'] not in ['cart', 'chair']:
                robot_root_state_w[self.object_grasp_pos_open_close_shift[env_ids, 1] < 0, :2] += self.object_grasp_pos_open_close_shift[env_ids][self.object_grasp_pos_open_close_shift[env_ids, 1] < 0, :2]
            self._robot.write_root_state_to_sim(robot_root_state_w, env_ids=env_ids) # type: ignore
        # set robot default root state
        else: self._robot.write_root_state_to_sim(self.robot_default_root_state_w[env_ids], env_ids=env_ids) # type: ignore
        
        # set robot default joint state
        self._robot.set_joint_position_target(self.robot_default_joint_pos[env_ids], env_ids=env_ids) # type: ignore
        self._robot.write_joint_state_to_sim(self.robot_default_joint_pos[env_ids], self.robot_default_joint_vel[env_ids], env_ids=env_ids) # type: ignore
        # replay training trajectories
        if replay and random:
            self._robot.write_root_state_to_sim(torch.tensor(replay_state_init['robot']['root_link_state_w']).to(self.device), env_ids=env_ids) # type: ignore
            self._robot.write_joint_state_to_sim(torch.tensor(replay_state_init['robot']['joint_pos']).to(self.device), torch.tensor(replay_state_init['robot']['joint_vel']).to(self.device), env_ids=env_ids) # type: ignore
        
        # reset robot
        self._robot.reset(env_ids=env_ids) # type: ignore
        
        
        """================ Reset Object States ================"""
        # set object random root state
        if random:
            # create random range
            random_shift = torch.rand(len(env_ids)).to(self.device) * 0.1
            random_height = torch.rand(len(env_ids)).to(self.device) * self.cfg.control_params['height_random_object']
            if 'zero_random_shift' in self.cfg.control_params and self.cfg.control_params['zero_random_shift']: random_shift *= 0.
            if 'zero_random_height' in self.cfg.control_params and self.cfg.control_params['zero_random_height']: random_height *= 0.
            # ArticulationObject: random object height
            if isinstance(self._object, Articulation):
                # transfer object_default_root_state_w with random_height
                object_root_state_w = self.object_default_root_state_w[env_ids].clone()
                object_root_state_w[:, 2] += random_height
                self._object.write_root_state_to_sim(object_root_state_w, env_ids=env_ids) # type: ignore
            # RigidObject: random object pos, rot, height
            elif isinstance(self._object, RigidObject):
                # transfer object_default_root_state_w with random_height
                object_root_state_w = self.object_default_root_state_w[env_ids].clone()
                object_root_state_w[:, 2] += random_height
                # create random rot range
                random_ang = torch.pi * 180 / 180
                random_rot = euler_angle_to_quaternion(torch.stack([torch.zeros(len(env_ids)), torch.zeros(len(env_ids)), torch.rand(len(env_ids)) * random_ang - random_ang / 2], dim=-1)).to(self.device)
                # random object rot
                object_root_state_w[:, 3:7] = quaternion_multiply(random_rot, object_root_state_w[:, 3:7])
                # create random object_root_state_w
                random_object_root_state_w = torch.zeros_like(object_root_state_w)
                random_object_root_state_w[:, 1] = 0.1
                self._object.write_root_state_to_sim(object_root_state_w + random_object_root_state_w, env_ids=env_ids) # type: ignore
                # update object grasp goal
                self.object_grasp_pos_goal[env_ids] = (object_root_state_w[:, :3] + random_object_root_state_w[:, :3]).clone() - self.env_origins[env_ids]
                # shift grasp goal with obj_points_center
                self.object_grasp_pos_goal[env_ids] += quaternion_rotate_vector(object_root_state_w[:, 3:7] + random_object_root_state_w[:, 3:7], torch.tensor(self._object_model.obj_points_center, dtype=torch.float).unsqueeze(0).repeat(len(env_ids), 1).to(self.device))
                self.object_grasp_pos_goal[env_ids, -1] += self.goal_distance
                self.object_grasp_rot_goal[env_ids] = object_root_state_w[:, 3:7].clone()
            # transfer stage_root_state_w with random_shift and random_height
            stage_root_state_w = self.stage_default_root_state[env_ids].clone()
            stage_root_state_w[:, :2] += object_root_state_w[:, :2] + self.stage_shift_vector[env_ids, :2]
            stage_root_state_w[:, 1] -= random_shift
            stage_root_state_w[:, 2] += random_height
            self._stage.write_root_state_to_sim(stage_root_state_w, env_ids=env_ids) # type: ignore
            # transfer ground_root_state_w with random_shift
            ground_root_state_w = self.ground_default_root_state[env_ids].clone()
            ground_root_state_w[:, :2] += stage_root_state_w[:, :2] + self.ground_shift_vector[env_ids, :2]
            ground_root_state_w[:, 1] -= random_shift
            self._ground.write_root_state_to_sim(ground_root_state_w, env_ids=env_ids) # type: ignore
        # set object default root state
        else: self._object.write_root_state_to_sim(self.object_default_root_state_w[env_ids], env_ids=env_ids) # type: ignore
        # replay training trajectories
        if replay and random: self._object.write_root_state_to_sim(torch.tensor(replay_state_init['object']['root_link_state_w']).to(self.device), env_ids=env_ids) # type: ignore
        
        # ArticulationObject: reset object joint
        if isinstance(self._object, Articulation):
            # set object default joint state
            self._object.write_joint_state_to_sim(self.object_default_joint_pos[env_ids], self.object_default_joint_vel[env_ids], env_ids=env_ids) # type: ignore
            # update object_grasp_rot_goal and object_grasp_pos_goal relative to default object joint state
            # combine local object grasp pose goal with global object grasp body pose
            self.object_grasp_rot_goal[env_ids], self.object_grasp_pos_goal[env_ids] = combine_transformation(
                self._object.data.body_link_quat_w[env_ids, self.object_grasp_body_idx].clone(), 
                self._object.data.body_link_pos_w[env_ids, self.object_grasp_body_idx].clone() - self.env_origins[env_ids],
                self.object_local_grasp_rot_goal[env_ids], self.object_local_grasp_pos_goal[env_ids]
            )
            
            # set object grasp_pos_goal for bucket and kettle
            if self.cfg.object_infos['group'] in ['bucket', 'kettle']:
                self.object_grasp_pos_goal[env_ids, -1] += 0.2
            
            # set object joint state for close tasks
            if self.cfg.action_type == 'close':
                # skip object close config for cart, chair
                if self.cfg.object_infos['group'] not in ['cart', 'chair']:
                    # set random joint
                    if random:
                        # create random range
                        random_ratio, random_range = self.cfg.object_infos['open_ratio'] * 2 - 1, 0.4 * 2
                        target_joint_pos = (random_ratio + torch.rand(len(env_ids)) * random_range - random_range / 2).to(self.device)
                        # update target_joint_pos
                        object_joint_pos = self.object_default_joint_pos[env_ids].clone()
                        object_joint_pos[:, self.object_grasp_joint_idx] = unnormalize_lower_upper(
                            target_joint_pos, self.object_joint_lower_limits[self.object_grasp_joint_idx], self.object_joint_upper_limits[self.object_grasp_joint_idx])
                        # update target_joint_pos for REVOLUTE
                        if self.object_grasp_joint_type == 'REVOLUTE':
                            random_angle, random_range = torch.pi * 60 / 180, torch.pi * 40 / 180
                            target_joint_pos = (random_angle + torch.rand(len(env_ids)) * random_range - random_range / 2).to(self.device)
                            object_joint_pos[:, self.object_grasp_joint_idx] = torch.min(
                                self.object_joint_lower_limits[self.object_grasp_joint_idx].repeat(len(env_ids)) + target_joint_pos, self.object_joint_upper_limits[self.object_grasp_joint_idx].repeat(len(env_ids)))
                    # set default joint
                    else:
                        # set half open joint
                        target_joint_pos = torch.zeros(len(env_ids)).to(self.device)
                        # update target_joint_pos
                        object_joint_pos = self.object_default_joint_pos[env_ids].clone()
                        object_joint_pos[:, self.object_grasp_joint_idx] = unnormalize_lower_upper(
                            target_joint_pos, self.object_joint_lower_limits[self.object_grasp_joint_idx], self.object_joint_upper_limits[self.object_grasp_joint_idx])
                    self._object.write_joint_state_to_sim(object_joint_pos, self.object_default_joint_vel[env_ids], env_ids=env_ids) # type: ignore
            # replay training trajectories
            if replay and random: self._object.write_joint_state_to_sim(torch.tensor(replay_state_init['object']['joint_pos']).to(self.device), torch.tensor(replay_state_init['object']['joint_vel']).to(self.device), env_ids=env_ids) # type: ignore
        
        # reset object
        self._object.reset(env_ids=env_ids) # type: ignore
        
        """================ Update Itermediate Params ================"""
        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self.env_times[env_ids] *= 0
        self.env_success[env_ids] *= 0
        self.env_success_flag[env_ids] *= 0
        self.env_success_ratio[env_ids] *= 0
        self.env_success_count[env_ids] *= 0
        # save and refresh episode_trajectory
        if self.cfg.record_trajectory:
            # save episode_trajectory
            if len(self.episode_trajectory) != 0 and len(self.episode_trajectory['time']) != 0:
                # concat episode_trajectory
                for key, value in self.episode_trajectory.items(): self.episode_trajectory[key] = torch.cat(self.episode_trajectory[key], dim=0).cpu().numpy() # type: ignore
                # locate save_episode_dir
                save_episode_dir = osp.join(self.log_dir, 'episode_{:03d}'.format(int(self.env_episodes[0] + self.cfg.record_exist - 1)))
                os.makedirs(save_episode_dir, exist_ok=True)
                # remove failed episode
                if np.sum(self.episode_trajectory['success']) == 0 and not self.cfg.vla_mode: # type: ignore
                    shutil.rmtree(save_episode_dir)
                    self.env_episodes[0] -= 1
                # save success episdoe
                else:
                    # save episode_image
                    save_video(osp.join(save_episode_dir, 'rgb_image_head.mp4'), self.episode_image['rgb_image_head'])
                    save_video(osp.join(save_episode_dir, 'depth_image_head.mp4'), self.episode_image['depth_image_head'])
                    save_video(osp.join(save_episode_dir, 'segment_image_head.mp4'), self.episode_image['segment_image_head'])
                    np.savez_compressed(osp.join(save_episode_dir, 'segment_label_head.npz'), array=np.stack(self.episode_image['segment_label_head'], axis=0))
                    np.savez_compressed(osp.join(save_episode_dir, 'distance_image_head.npz'), array=np.stack(self.episode_image['distance_image_head'], axis=0))
                    save_video(osp.join(save_episode_dir, 'rgb_image_arm.mp4'), self.episode_image['rgb_image_arm'])
                    save_video(osp.join(save_episode_dir, 'depth_image_arm.mp4'), self.episode_image['depth_image_arm'])
                    save_video(osp.join(save_episode_dir, 'segment_image_arm.mp4'), self.episode_image['segment_image_arm'])
                    np.savez_compressed(osp.join(save_episode_dir, 'segment_label_arm.npz'), array=np.stack(self.episode_image['segment_label_arm'], axis=0))
                    np.savez_compressed(osp.join(save_episode_dir, 'distance_image_arm.npz'), array=np.stack(self.episode_image['distance_image_arm'], axis=0))
                    # save episode_trajectory
                    self.episode_trajectory['init'] = self.episode_init # type: ignore
                    dump_pickle(osp.join(save_episode_dir, 'state_infos.pkl'), self.episode_trajectory)
                    # exit with record_episode
                    if self.env_episodes[0] + self.cfg.record_exist >= self.cfg.record_episode: self.cfg.exit_flag = True
            # refresh episode_trajectory and episode_image
            self.episode_trajectory = {'time': [], 'success': [], 'action': [], 'camera_head_pose': [], 'camera_arm_pose': [], 'object': [], 'robot_base': [], 'robot_hand': [], 'robot_body': [], 'robot_joint': [], 'robot_joint_target': []}
            self.episode_image = {'rgb_image_head': [], 'depth_image_head': [], 'distance_image_head': [], 'segment_image_head': [], 'segment_label_head': [],
                                  'rgb_image_arm': [], 'depth_image_arm': [], 'distance_image_arm': [], 'segment_image_arm': [], 'segment_label_arm': []}
            
            # refresh episode initial state
            if random:
                self.episode_init = {'robot': {}, 'object': {}, 'room': {}}
                self.episode_init['robot']['root_link_state_w'] = self._robot.data.root_link_state_w.clone().cpu().numpy()
                self.episode_init['robot']['joint_pos'] = self._robot.data.joint_pos.clone().cpu().numpy()
                self.episode_init['robot']['joint_vel'] = self._robot.data.joint_vel.clone().cpu().numpy()
                self.episode_init['object']['root_link_state_w'] = self._object.data.root_link_state_w.clone().cpu().numpy()
                # ArticulationObject: record init joint
                if isinstance(self._object, Articulation):
                    self.episode_init['object']['joint_pos'] = self._object.data.joint_pos.clone().cpu().numpy()
                    self.episode_init['object']['joint_vel'] = self._object.data.joint_vel.clone().cpu().numpy()
                self.episode_init['room']['room_pos'], self.episode_init['room']['room_rot'] = room_pos.copy(), room_rot.copy()
        
        self._compute_intermediate_values()
    
    
    def _get_observations(self) -> dict:
        # print('run _get_observations')
        obs_dict = dict()
        
        # time: (nenv, 30)
        obs_dict['time'] = time_encoding(self.env_times, 30)
        # action: (nenv, naction)
        obs_dict['action'] = self.actions
        # robot_palm: (nenv, 12)
        obs_dict['robot_palm'] = torch.cat([self.robot_palm_pos, self.robot_palm_rot_euler, self.robot_palm_vel, self.robot_palm_ang], dim=-1)
        # robot_finger: (nenv, nfinger x 12)
        obs_dict['robot_finger'] = torch.cat([self.robot_finger_pos, self.robot_finger_rot_euler, self.robot_finger_vel, self.robot_finger_ang], dim=-1).reshape(self.num_envs, -1)
        # robot_joint: (nenv, njoint_active x 3)
        obs_dict['robot_joint'] = torch.stack([self.robot_joint_pos_norm, self.robot_joint_vel, self.robot_joint_acc], dim=-1)[:, self.robot_active_joint_indices].reshape(self.num_envs, -1)
        # robot_object: (nenv, 1 + nfinger + nbody_active + 3)
        obs_dict['robot_object'] = torch.cat([self.dist_hand_object, self.dist_hand_finger_object, self.dist_hand_body_object, self.object_grasp_pos - self.robot_palm_pos], dim=-1)
        # object_goal: (nenv, 3 + 3 + 3)
        obs_dict['object'] = torch.cat([self.object_grasp_pos, self.object_grasp_rot_euler, self.object_grasp_pos_goal - self.object_grasp_pos], dim=-1)
        
        obs = torch.cat(
            (
                obs_dict['time'],
                obs_dict['action'],
                obs_dict['robot_palm'],
                obs_dict['robot_finger'],
                obs_dict['robot_joint'],
                obs_dict['robot_object'],
                obs_dict['object'],
            ),
            dim=-1,
        )
        
        # append episode_trajectory
        if self.cfg.record_trajectory and self.env_times[0] != 0 and (self.env_times[0] == 1. or self.env_times[0] % self.cfg.save_record_step == 0):
            # time: (nenv, 1)
            self.episode_trajectory['time'].append(self.env_times.unsqueeze(-1).clone())
            # success: (nenv, 1)
            self.episode_trajectory['success'].append(self.env_success_flag.unsqueeze(-1).clone())
            # actions: (nenv, naction)
            self.episode_trajectory['action'].append(self.actions.clone())
            # camera pose: (nenv, 3 + 3)
            self.episode_trajectory['camera_head_pose'].append(torch.cat([self.camera_head_pos, self.camera_head_rot_euler], dim=-1).clone())
            self.episode_trajectory['camera_arm_pose'].append(torch.cat([self.camera_arm_pos, self.camera_arm_rot_euler], dim=-1).clone())
            # object: (nenv, 3 + 3 + 3)
            self.episode_trajectory['object'].append(torch.cat([self.object_grasp_pos, self.object_grasp_rot_euler, self.object_grasp_pos_goal], dim=-1).clone())
            # robot_base: (nenv, 3 + 3 + 3 + 3)
            self.episode_trajectory['robot_base'].append(torch.cat([self.robot_base_pos, self.robot_base_rot_euler, self.robot_base_vel, self.robot_base_ang], dim=-1).clone())
            # robot_hand: (nenv, 3 + 3 + 3 + 3)
            self.episode_trajectory['robot_hand'].append(torch.cat([self.robot_hand_pos, self.robot_hand_rot_euler, self.robot_hand_vel, self.robot_hand_ang], dim=-1).clone())
            # robot_body: (nenv, nbody, 3 + 3 + 3 + 3)
            self.episode_trajectory['robot_body'].append(torch.cat([self.robot_body_pos, self.robot_body_rot_euler, self.robot_body_vel, self.robot_body_ang], dim=-1).clone())
            # robot_joint: (nenv, njoint, 1 + 1 + 1)
            self.episode_trajectory['robot_joint'].append(torch.stack([self.robot_joint_pos, self.robot_joint_vel, self.robot_joint_acc], dim=-1).clone())
            # robot_joint_target: (nenv, njoint)
            self.episode_trajectory['robot_joint_target'].append(self.robot_joint_target.clone())
            
            # assert single env
            nenv = 0
            assert self.num_envs == 1
            # update rgb, segment, distance images for camera_head
            rgb_image_head = self._camera_head.data.output['rgb']
            distance_image_head = self._camera_head.data.output['distance_to_camera']
            semantic_image_head = self._camera_head.data.output['semantic_segmentation']
            # normalize distance_image
            depth_clip = 5
            depth_image_head = distance_image_head.clone().clamp(0, depth_clip).repeat(1, 1, 1, 3)
            depth_image_head = (depth_image_head * 255 / depth_clip).to(torch.uint8)
            # format distance_image
            distance_image_head = (distance_image_head.squeeze(-1) * 1000.0).clamp(0, 65535).round().to(torch.uint16)
            # normalize semantic_image
            segment_image_head = semantic_image_head.clone() * torch.tensor([0., 0., 0., 1.]).to(self.device)
            for rgb, label in self._camera_head.data.info[nenv]['semantic_segmentation']['idToLabels'].items():
                if label['class'] == 'robot': segment_image_head[torch.norm(semantic_image_head - torch.tensor(eval(rgb), dtype=torch.float).to(self.device), dim=-1) == 0] = torch.tensor([0., 0., 255., 255.]).to(self.device)
                if label['class'] == 'object': segment_image_head[torch.norm(semantic_image_head - torch.tensor(eval(rgb), dtype=torch.float).to(self.device), dim=-1) == 0] = torch.tensor([0., 255., 0., 255.]).to(self.device)
            segment_image_head = segment_image_head[..., :3].to(dtype=torch.uint8)
            # format segment_label
            segment_label_head = segment_image_head[..., 0].clone() * 0
            segment_label_head[segment_image_head[..., 1] == 255] = 1
            segment_label_head[segment_image_head[..., 2] == 255] = 2
            # append camera images
            self.episode_image['rgb_image_head'].append(rgb_image_head[nenv].cpu().numpy())
            self.episode_image['depth_image_head'].append(depth_image_head[nenv].cpu().numpy())
            self.episode_image['distance_image_head'].append(distance_image_head[nenv].cpu().numpy())
            self.episode_image['segment_image_head'].append(segment_image_head[nenv].cpu().numpy())
            self.episode_image['segment_label_head'].append(segment_label_head[nenv].cpu().numpy())
            
            # update rgb, segment, distance images for camera_arm
            rgb_image_arm = self._camera_arm.data.output['rgb']
            distance_image_arm = self._camera_arm.data.output['distance_to_camera']
            semantic_image_arm = self._camera_arm.data.output['semantic_segmentation']
            # normalize distance_image
            depth_clip = 5
            depth_image_arm = distance_image_arm.clone().clamp(0, depth_clip).repeat(1, 1, 1, 3)
            depth_image_arm = (depth_image_arm * 255 / depth_clip).to(torch.uint8)
            # format distance_image
            distance_image_arm = (distance_image_arm.squeeze(-1) * 1000.0).clamp(0, 65535).round().to(torch.uint16)
            # normalize semantic_image
            segment_image_arm = semantic_image_arm.clone() * torch.tensor([0., 0., 0., 1.]).to(self.device)
            for rgb, label in self._camera_arm.data.info[nenv]['semantic_segmentation']['idToLabels'].items():
                if label['class'] == 'robot': segment_image_arm[torch.norm(semantic_image_arm - torch.tensor(eval(rgb), dtype=torch.float).to(self.device), dim=-1) == 0] = torch.tensor([0., 0., 255., 255.]).to(self.device)
                if label['class'] == 'object': segment_image_arm[torch.norm(semantic_image_arm - torch.tensor(eval(rgb), dtype=torch.float).to(self.device), dim=-1) == 0] = torch.tensor([0., 255., 0., 255.]).to(self.device)
            segment_image_arm = segment_image_arm[..., :3].to(dtype=torch.uint8)
            # format segment_label
            segment_label_arm = segment_image_arm[..., 0].clone() * 0
            segment_label_arm[segment_image_arm[..., 1] == 255] = 1
            segment_label_arm[segment_image_arm[..., 2] == 255] = 2
            # append camera images
            self.episode_image['rgb_image_arm'].append(rgb_image_arm[nenv].cpu().numpy())
            self.episode_image['depth_image_arm'].append(depth_image_arm[nenv].cpu().numpy())
            self.episode_image['distance_image_arm'].append(distance_image_arm[nenv].cpu().numpy())
            self.episode_image['segment_image_arm'].append(segment_image_arm[nenv].cpu().numpy())
            self.episode_image['segment_label_arm'].append(segment_label_arm[nenv].cpu().numpy())
        
        # vla mode
        if self.cfg.vla_mode: return {"policy": torch.nan_to_num(obs, nan=0.0).clamp(-10.0, 10.0).clone(), 'render': self.episode_image, 'trajectory': self.episode_trajectory}
        
        return {"policy": torch.nan_to_num(obs, nan=0.0).clamp(-10.0, 10.0).clone()}
