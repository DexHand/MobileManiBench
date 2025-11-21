import pybullet
import pybullet_data
from xml.etree import ElementTree as ET
from unimanip.utils.general_utils import *

# locate unidoor dataset
unidoor_dir = osp.join(ASSET_DIR, 'unidoor')
unidoor_analysis = load_yaml(osp.join(unidoor_dir, 'analysis.yaml'))
# locate all skip category objects
Skip_Cat_Objects = {}
# locate all pick category objects
Pick_Cat_Objects = {}

# preprocess unidoor urdf
def preprocess_unidoor_urdf(urdf_dir, save_dir):
    # load urdf
    tree = ET.parse(urdf_dir)
    root = tree.getroot()
    # iterate all origin tags
    for origin in root.findall(".//origin"):
        # switch y and z
        xyz = origin.attrib.get("xyz")
        if xyz:
            parts = xyz.split()
            parts[1], parts[2] = parts[2], parts[1]
            origin.set("xyz", " ".join(parts))
    # iterate all joint tags
    for joint in root.findall("joint"):
        # change joint_0 origin rpy
        if joint.attrib.get("name") == "joint_0":
            origin = joint.find("origin")
            if origin is not None: origin.set("rpy", "0 0 0")
        # change joint_1 axis xyz
        if joint.attrib.get("name") == "joint_1" and joint.attrib.get("type") == "revolute":
            axis = joint.find("axis")
            if axis is not None: axis.set("xyz", "0 0 -1")
        # change joint_2, joint_3 axis xyz, limit upper, origin xyz
        if joint.attrib.get("name") in ["joint_2", "joint_3"] and joint.attrib.get("type") == "revolute":
            axis = joint.find("axis")
            if axis is not None: axis.set("xyz", "0 -1 0")
            limit = joint.find("limit")
            if limit is not None:
                limit.set("lower", "0"); limit.set("upper", "0.01")
            origin = joint.find("origin")
            if origin is not None:
                xyz = origin.attrib.get("xyz")
                if xyz:
                    parts = xyz.split()
                    parts[1] = "-" + parts[1]
                    origin.set("xyz", " ".join(parts))
    # save urdf
    tree.write(save_dir, encoding="utf-8", xml_declaration=True)
    

class UniDoorObjModel:
    def __init__(self, obj_id, obj_cat, num_envs=1, num_surface_samples=100, target_open_ratio=0.6, target_handle_direction=np.array([0., -1., 0.]), device='cuda:0'):
        """
        Create a Kinematic Model from unidoor URDF File using pybulet
        
        Parameters
        ----------
        obj_id: str, object id
        obj_cat: str, object category
        num_envs: int, number of envs
        num_surface_samples: int, number of surface samples per area
        device: str, torch.Device, device for torch tensors
        """
        
        # init basic
        self.device = device
        self.num_envs = num_envs
        self.num_surface_samples = num_surface_samples
        self.target_open_ratio = target_open_ratio
        # init object id and category
        self.obj_id = obj_id
        self.obj_cat = obj_cat
        self.obj_type = 'unidoor'
        self.obj_scale = 1.0
        # locate object urdf, usd, config, visualize dir
        preprocess_unidoor_urdf(osp.join(unidoor_dir, 'dataset', self.obj_cat, self.obj_id, 'mobility.urdf'), osp.join(unidoor_dir, 'dataset', self.obj_cat, self.obj_id, 'mobility_convert.urdf'))
        self.obj_urdf_dir = osp.join(unidoor_dir, 'dataset', self.obj_cat, self.obj_id, 'mobility_convert.urdf')        
        self.obj_usd_dir = osp.join(unidoor_dir, 'process', self.obj_cat, self.obj_id, 'mobility.usd')
        self.obj_visualize_dir = osp.join(unidoor_dir, 'process', self.obj_cat, self.obj_id, 'Visualize')
        # if osp.exists(self.obj_visualize_dir): shutil.rmtree(self.obj_visualize_dir)
        os.makedirs(self.obj_visualize_dir, exist_ok=True)
        
        # init pybullet
        pybullet.connect(pybullet.DIRECT)
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        # load object urdf model
        self.obj_model = pybullet.loadURDF(self.obj_urdf_dir, useFixedBase=True)
        self.obj_dofs = pybullet.getNumJoints(self.obj_model)
        
        # load object joint_link_meshes
        self.load_joint_link_meshes()
        # set joint_pos as lower
        joint_pos = unnormalize_lower_upper(-np.ones(self.obj_dofs), np.array(self.obj_joint_lower), np.array(self.obj_joint_upper))
        self.forward_joint_link_meshes(joint_pos, save=False)
        # get object canonical points
        self.obj_points_canonical = self.obj_points.copy()
        self.obj_highest = np.max(self.obj_points_canonical, axis=0)
        self.obj_lowest = np.min(self.obj_points_canonical, axis=0)
        # get object bbox, center, size
        self.obj_bbox = trimesh.points.PointCloud(self.obj_points_canonical).bounding_box.bounds
        self.obj_center = (self.obj_bbox[1] + self.obj_bbox[0]) / 2
        self.obj_size = self.obj_bbox[1] - self.obj_bbox[0]
        # process object handles with target_handle_direction
        self.target_handle_direction = target_handle_direction
        self.load_joint_handles()
    
    
    # load object joint, link, meshes
    def load_joint_link_meshes(self):
        # init object joint_link_infos
        self.obj_joint_names = []
        self.obj_joint_types = []
        self.obj_joint_links = []
        self.obj_joint_lower = []
        self.obj_joint_upper = []
        self.obj_joint_handle_names = []
        self.obj_joint_link_infos = {}
        # process all joints and links
        for joint_index in range(self.obj_dofs):
            # Get joint info
            joint_info = pybullet.getJointInfo(self.obj_model, joint_index)
            joint_name, joint_type, joint_lower, joint_upper, joint_link = \
                joint_info[1].decode("utf-8"), joint_info[2], joint_info[8], joint_info[9], joint_info[12].decode("utf-8")
            # Get joint type
            if joint_type == 0: joint_type = 'REVOLUTE'
            elif joint_type == 1: joint_type = 'PRISMATIC'
            else: joint_type = None
            # Get link info
            link_state = pybullet.getLinkState(self.obj_model, joint_index, computeForwardKinematics=True)
            
            # Assign joint_link_infos
            joint_link_infos = {'name': joint_name, 'type': joint_type, 'lower': joint_lower, 'upper': joint_upper, 'link': joint_link,
                                'pos': np.array(link_state[4]), 'rot':np.array(link_state[5]), 
                                'body_mesh': [], 'handle_mesh': [], 'body_points': [], 'handle_points': [], 'handle_name': []}
            # Get collision shape data
            collision_data = pybullet.getCollisionShapeData(self.obj_model, joint_index)
            # Process joint_link collision shape data
            for c_data in collision_data:
                # load mesh
                c_mesh_dir = c_data[4].decode("utf-8")
                c_mesh = trimesh.load(c_mesh_dir)
                c_mesh.apply_scale(c_data[3])
                # transform origin mesh using collisionFramePosition and collisionFrameOrientation 
                transform_matrix = trimesh.transformations.quaternion_matrix(np.array(c_data[-1])[[-1, 0, 1, 2]]) # (quaternion: w, x, y, z)
                transform_matrix[:3, 3] = np.array(c_data[-2])
                c_mesh.apply_transform(transform_matrix) # type: ignore
                # check mesh type
                mesh_type = 'body_mesh'
                # append mesh list
                if isinstance(c_mesh, trimesh.Scene): joint_link_infos[mesh_type].append(trimesh.util.concatenate([mesh for mesh in c_mesh.dump()]))
                else: joint_link_infos[mesh_type].append(c_mesh)
            
            # combine all body_mesh, sample body_points
            if len(joint_link_infos['body_mesh']) > 0:
                body_mesh_combined = trimesh.util.concatenate(joint_link_infos['body_mesh'])
                body_points_combined, _ = trimesh.sample.sample_surface_even(body_mesh_combined, max(10, int(body_mesh_combined.area * self.num_surface_samples)))
                joint_link_infos['body_mesh'] = [body_mesh_combined]
                joint_link_infos['body_points'] = [trimesh.points.PointCloud(body_points_combined)]
            
            # assign model_meshes
            self.obj_joint_names.append(joint_link_infos['name'])
            self.obj_joint_types.append(joint_link_infos['type'])
            self.obj_joint_links.append(joint_link_infos['link'])
            self.obj_joint_lower.append(joint_link_infos['lower'])
            self.obj_joint_upper.append(joint_link_infos['upper'])
            self.obj_joint_handle_names.append(joint_link_infos['handle_name'])
            self.obj_joint_link_infos[joint_link_infos['name']] = joint_link_infos
        # process object dofs
        self.obj_dof_joint_names = [self.obj_joint_names[njoint] for njoint in range(len(self.obj_joint_names)) if self.obj_joint_types[njoint] is not None]
        
        # locate door_index and handle_index
        for joint_index in range(self.obj_dofs):
            if self.obj_joint_types[joint_index] == 'REVOLUTE': self.obj_joint_door_index = joint_index; break
        for joint_index in range(self.obj_dofs):
            if self.obj_joint_types[joint_index] == 'REVOLUTE' and joint_index != self.obj_joint_door_index: self.obj_joint_handle_index = joint_index; break
        
        # resample handle_points
        body_points_combined, _ = trimesh.sample.sample_surface_even(self.obj_joint_link_infos[self.obj_joint_names[self.obj_joint_handle_index]]['body_mesh'][0], max(10, int(body_mesh_combined.area * self.num_surface_samples**2)))
        self.obj_joint_link_infos[self.obj_joint_names[self.obj_joint_handle_index]]['body_points'] = [trimesh.points.PointCloud(body_points_combined)]
    
    
    # forward joint link meshes with joint_pos
    def forward_joint_link_meshes(self, joint_pos, save=False):
        assert len(joint_pos) == self.obj_dofs, "Joint pose length must match number of DOFs"
        
        # forward joint_pos
        for joint_index in range(self.obj_dofs):
            pybullet.resetJointState(self.obj_model, joint_index, joint_pos[joint_index])
        
        # forward joint_link_meshes
        self.obj_points = []
        for joint_index in range(self.obj_dofs):
            # get joint and link state
            joint_name = self.obj_joint_names[joint_index]
            link_state = pybullet.getLinkState(self.obj_model, joint_index, computeForwardKinematics=True)
            link_pos = np.array(link_state[4])  # World position (x, y, z)
            link_rot = np.array(link_state[5])  # World orientation (quaternion: x, y, z, w)
            # get link transformation
            transform_matrix = trimesh.transformations.quaternion_matrix(link_rot[[-1, 0, 1, 2]]) # (quaternion: w, x, y, z)
            transform_matrix[:3, 3] = link_pos  # Set translation
            # apply transformation on body_mesh, body_points
            if len(self.obj_joint_link_infos[joint_name]['body_mesh']) > 0:
                self.obj_joint_link_infos[joint_name]['body_mesh_deformed'] = []
                self.obj_joint_link_infos[joint_name]['body_points_deformed'] = []
                for nbody in range(len(self.obj_joint_link_infos[joint_name]['body_mesh'])):
                    body_mesh_deformed = self.obj_joint_link_infos[joint_name]['body_mesh'][nbody].copy()
                    body_mesh_deformed.apply_transform(transform_matrix)
                    body_points_deformed = self.obj_joint_link_infos[joint_name]['body_points'][nbody].copy()
                    body_points_deformed.apply_transform(transform_matrix)
                    if save: body_points_deformed.export(osp.join(self.obj_visualize_dir, '{}_body.ply'.format(self.obj_joint_link_infos[joint_name]['name'])))
                    self.obj_points.append(body_points_deformed.vertices)
                    self.obj_joint_link_infos[joint_name]['body_mesh_deformed'].append(body_mesh_deformed.copy())
                    self.obj_joint_link_infos[joint_name]['body_points_deformed'].append(body_points_deformed.copy())
        
        # concat obj_points
        self.obj_points = np.concatenate(self.obj_points, axis=0)
    
    
    
    # load all joint_handles
    def load_joint_handles(self):
        # init obj_joint_handle_infos
        self.obj_joint_handle_infos = {}
        # locate joint_name and joint_link_type
        joint_index = self.obj_joint_door_index
        handle_index = self.obj_joint_handle_index
        joint_name = self.obj_joint_names[joint_index]
        joint_type = self.obj_joint_types[joint_index]
        joint_link = self.obj_joint_links[joint_index]
        
        # get handle_name
        handle_name = self.obj_joint_names[handle_index]
        # get handle dimension
        handle_bbox = self.obj_joint_link_infos[handle_name]['body_mesh_deformed'][0].bounding_box.bounds
        handle_size = handle_bbox[1] - handle_bbox[0]
        
        # get opened handle: 60% for PRISMATIC, 60 degrees for REVOLUTE
        opened_handle_ratio = self.target_open_ratio
        joint_pos = unnormalize_lower_upper(-np.ones(self.obj_dofs), np.array(self.obj_joint_lower), np.array(self.obj_joint_upper))
        joint_pos[joint_index] = unnormalize_lower_upper(opened_handle_ratio * 2 - 1, self.obj_joint_lower[joint_index], self.obj_joint_upper[joint_index])
        if joint_type == 'REVOLUTE': joint_pos[joint_index] = min(self.obj_joint_lower[joint_index] + np.pi * 60 / 180, self.obj_joint_upper[joint_index])
        self.forward_joint_link_meshes(joint_pos)
        opened_handle_points = self.obj_joint_link_infos[handle_name]['body_points_deformed'][0].vertices.copy()
        opened_handle_pos = np.mean(self.obj_joint_link_infos[handle_name]['body_points_deformed'][0].vertices, axis=0)
        # trimesh.points.PointCloud(opened_handle_points).export(osp.join(self.obj_visualize_dir, 'handle_opened.ply'))
        # trimesh.points.PointCloud(self.obj_points).export(osp.join(self.obj_visualize_dir, 'opened.ply'))
        
        # get disturbed handle
        joint_pos = unnormalize_lower_upper(-np.ones(self.obj_dofs), np.array(self.obj_joint_lower), np.array(self.obj_joint_upper))
        joint_pos[joint_index] = unnormalize_lower_upper(-1.0 + 0.1, self.obj_joint_lower[joint_index], self.obj_joint_upper[joint_index])
        self.forward_joint_link_meshes(joint_pos)
        disturb_handle_points = self.obj_joint_link_infos[handle_name]['body_points_deformed'][0].vertices.copy()
        disturb_handle_pos = np.mean(self.obj_joint_link_infos[handle_name]['body_points_deformed'][0].vertices, axis=0)
        
        # get closed handle
        joint_pos = unnormalize_lower_upper(-np.ones(self.obj_dofs), np.array(self.obj_joint_lower), np.array(self.obj_joint_upper))
        self.forward_joint_link_meshes(joint_pos)
        closed_handle_points = self.obj_joint_link_infos[handle_name]['body_points_deformed'][0].vertices.copy()
        closed_handle_pos = np.mean(self.obj_joint_link_infos[handle_name]['body_points_deformed'][0].vertices, axis=0)
        # trimesh.points.PointCloud(closed_handle_points).export(osp.join(self.obj_visualize_dir, 'handle_closed.ply'))
        # trimesh.points.PointCloud(self.obj_points).export(osp.join(self.obj_visualize_dir, 'closed.ply'))
        
        # get handle_direction
        handle_direction = np.zeros(3)
        disturbed_handle_diff = disturb_handle_pos - closed_handle_pos
        if np.abs(disturbed_handle_diff[0]) > np.abs(disturbed_handle_diff[1]): handle_direction[0] = 1.0 * np.sign(disturbed_handle_diff[0])
        else:  handle_direction[1] = 1.0 * np.sign(disturbed_handle_diff[1])
        # get handle_rotation: align hand_direction to target axis
        handle_rotation = quaternion_between_vectors(torch.tensor(self.target_handle_direction).unsqueeze(0), torch.tensor(handle_direction).unsqueeze(0)).squeeze(0).numpy()
        
        # locate handle_point and handle_indices for safe
        if self.obj_cat in ['safe']:
            # locate lower_indices
            center_point = np.mean(closed_handle_points, axis=0)
            lower_indices = closed_handle_points[:, -1] < center_point[-1]
            # locate handle_point and handle_indices, filter lower_indices
            handle_point = closed_handle_points[np.argmax(np.linalg.norm(opened_handle_points - closed_handle_points, axis=1))]
            handle_indices = np.abs(closed_handle_points - handle_point)[:, np.argmax(np.abs(handle_direction[[1, 0, 2]]))] < 0.05
            handle_indices[lower_indices] = False
            # update handle_pos
            opened_handle_pos = np.mean(opened_handle_points[handle_indices], axis=0)
            disturb_handle_pos = np.mean(disturb_handle_points[handle_indices], axis=0)
            closed_handle_pos = np.mean(closed_handle_points[handle_indices], axis=0)
            # filter handle_points
            self.obj_joint_link_infos[handle_name]['body_points_deformed'][0] = trimesh.points.PointCloud(self.obj_joint_link_infos[handle_name]['body_points_deformed'][0].vertices[handle_indices])
            # trimesh.points.PointCloud(self.obj_joint_link_infos[handle_name]['body_points_deformed'][0]).export(osp.join(self.obj_visualize_dir, 'handle_filtered.ply'))
        
        # frame handle_pos and handle_direction relative to link_pos and link_rot
        link_state = pybullet.getLinkState(self.obj_model, joint_index, computeForwardKinematics=True)
        link_pos = np.array(link_state[4])  # World position (x, y, z)
        link_rot = np.array(link_state[5])  # World orientation (quaternion: x, y, z, w)
        closed_handle_pos = quaternion_frame_vector(torch.tensor(link_rot[[-1, 0, 1, 2]]).unsqueeze(0), torch.tensor(closed_handle_pos - link_pos).unsqueeze(0)).squeeze(0).numpy()
        opened_handle_pos = quaternion_frame_vector(torch.tensor(link_rot[[-1, 0, 1, 2]]).unsqueeze(0), torch.tensor(opened_handle_pos - link_pos).unsqueeze(0)).squeeze(0).numpy()
        handle_direction = quaternion_frame_vector(torch.tensor(link_rot[[-1, 0, 1, 2]]).unsqueeze(0), torch.tensor(handle_direction).unsqueeze(0)).squeeze(0).numpy()
        
        # save handle_infos
        self.obj_joint_handle_infos['{}_handle_{}'.format(joint_name, handle_index)] = {
            'joint_name': joint_name,
            'joint_type': joint_type,
            'joint_link': joint_link,
            'handle_name': handle_name,
            'handle_index': handle_index,
            'handle_bbox': handle_bbox,
            'handle_size': handle_size,
            'handle_rotation': handle_rotation,
            'handle_direction': handle_direction,
            'closed_handle_pos': closed_handle_pos,
            'opened_handle_pos': opened_handle_pos,
            'handle_trajectory': self.load_handle_trajectory(),
        }
    
    # load handle_trajectory in world frame
    def load_handle_trajectory(self, num_steps=10):
        # get handle trajectory
        handle_traj_pos_w = []
        for nstep in range(num_steps + 1):
            # get opened handle: 0.8 * 2. - 1. = 0.6
            joint_pos = unnormalize_lower_upper(-np.ones(self.obj_dofs), np.array(self.obj_joint_lower), np.array(self.obj_joint_upper))
            joint_pos[self.obj_joint_door_index] = unnormalize_lower_upper(-1.0 + nstep * (2.0 / num_steps), self.obj_joint_lower[self.obj_joint_door_index], self.obj_joint_upper[self.obj_joint_door_index])
            self.forward_joint_link_meshes(joint_pos)
            handle_traj_pos_w.append(np.mean(self.obj_joint_link_infos[self.obj_joint_names[self.obj_joint_handle_index]]['body_points_deformed'][0].vertices, axis=0))
        handle_traj_pos_w = np.stack(handle_traj_pos_w, axis=0)
        return handle_traj_pos_w
    
    # process IsaacLab ArticulationCfg
    def process_obj_sim_config(self, obj_art_config, obj_joint_handle_name):        
        # InitialStateCfg
        obj_art_config.init_state.pos = (obj_art_config.init_state.pos[0], obj_art_config.init_state.pos[1], 0.01 - self.obj_lowest[-1] * self.obj_scale)
        obj_art_config.init_state.rot = self.obj_joint_handle_infos[obj_joint_handle_name]['handle_rotation'].tolist()
        obj_art_config.init_state.joint_pos = {dof_joint_name: self.obj_joint_link_infos[dof_joint_name]['lower'] + 0.001 for dof_joint_name in self.obj_dof_joint_names}
        # ImplicitActuatorCfg
        obj_art_config.actuators['joints'].joint_names_expr = [dof_joint_name for dof_joint_name in self.obj_dof_joint_names]
        # Rotate object in Simulation
        self.obj_points_rotated = quaternion_rotate_vector(torch.tensor(self.obj_joint_handle_infos[obj_joint_handle_name]['handle_rotation']).unsqueeze(0), torch.tensor(self.obj_points_canonical)).numpy()
        # Update object bbox, center, size
        self.obj_bbox_rotated = trimesh.points.PointCloud(self.obj_points_rotated).bounding_box.bounds
        self.obj_center_rotated = (self.obj_bbox_rotated[1] + self.obj_bbox_rotated[0]) / 2
        self.obj_size_rotated = self.obj_bbox_rotated[1] - self.obj_bbox_rotated[0]


if __name__ == '__main__':
    # add argparse arguments
    parser = argparse.ArgumentParser(description="Utility to process unidoor objects.")
    parser.add_argument("--category", type=str, default="Cabinet", help="Process unidoor object category.")
    # parse the arguments
    args = parser.parse_args()
    
    # locate category and obj_ids
    obj_cat = args.category
    obj_cat_ids = list(unidoor_analysis[obj_cat].keys())
    
    # locate obj_with_handle
    obj_with_handle = {'objects': {}, 'infos': {}}
    # process all object ids
    for obj_id in obj_cat_ids:
        if obj_cat in Skip_Cat_Objects and obj_id in Skip_Cat_Objects[obj_cat]: continue
        if obj_cat in Pick_Cat_Objects and obj_id not in Pick_Cat_Objects[obj_cat]: continue
        try:
            # init object model
            obj_model = UniDoorObjModel(obj_id, obj_cat)
            # loate all object handles
            if len(obj_model.obj_joint_handle_infos) > 0:
                for joint_handle_name, joint_handle_info in obj_model.obj_joint_handle_infos.items():
                    obj_with_handle['objects']['{}/{}/{}/{}/handle_{}'.format(obj_id, joint_handle_info['joint_name'], joint_handle_info['joint_type'], joint_handle_info['joint_link'], joint_handle_info['handle_index'])] = {}
                obj_with_handle['infos'][obj_id] = {'handle': obj_model.obj_joint_handle_infos, 'bbox': np.stack([obj_model.obj_lowest, obj_model.obj_highest], axis=0)}
        except: pass
    dump_pickle(osp.join(unidoor_dir, 'process', obj_cat, 'analysis.pkl'), obj_with_handle)
    dump_yaml(osp.join(unidoor_dir, 'process', obj_cat, 'analysis.yaml'), {'objects': obj_with_handle['objects']})