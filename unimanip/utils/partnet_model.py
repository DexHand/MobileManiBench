import pybullet
import pybullet_data
from unimanip.utils.general_utils import *

# locate partnet dataset
partnet_dir = osp.join(ASSET_DIR, 'partnet')
partnet_analysis = load_yaml(osp.join(partnet_dir, 'analysis.yaml'))
# locate all skip category objects
Skip_Cat_Objects = {
    'chair': ['35063', '39551', '39988'],
    'trashcan': ['102163', '102193', '103007'],
    'refrigerator': ['11846', '12036', '12050', '12059'],
    'toilet': ['102675', '102703'], 'washingmachine': ['103490', '103508', '103528', '103776', '103781'],
    'laptop': ['10125', '10238', '10280', '10305', '11406', '10626', '11030', '11778', '11854', '11876', '11945', '9996'],
    'table': ['20985', '21467', '23782', '26899', '27267', '30341', '32174', '32259', '32601', '32625', '33457', '33810', '33930'],
    'dishwasher': ['11661', '12065', '12085', '12530', '12536', '12552', '12563', '12565', '12579', '12592', '12594', '12597', '12617'],
    'faucet': ['1386', '1435', '1488', '167', '1712', '1721', '1785', '1788', '1823', '1941', '2017', '2082', '2084', '2095', '2113', '862', '866', '912', '920'],
    'storage_furniture': ['41086', '41452', '41529', '45087', '45091', '45134', '45177', '45247', '45297', '45385', '45403', '45415', '45419', '45503', '45504', '45600', '45621', '45633', 
                          '45667', '45671', '45693', '45699', '45717', '45747', '45779', '45855', '45915', '45922', '45963', '46044', '46107', '46117', '46127', '46427', '46430', '46744', 
                          '46787', '47133', '47180', '47315', '47391', '47419', '47601', '47632', '47729', '48023', '48036', '48167', '48169', '48243', '48467', '48490', '48492', '49038'],
}
# locate all pick category objects
Pick_Cat_Objects = {
    'box': ['100141', '100174', '100189', '100191', '100214', '100221', '100243', '100247', '100664', '100671', '100685', '102377', '102379', '102456'],
    'cart': ['100491', '100498', '100501', '100508', '100852', '100853', '100856', '100860', '101066', '101081', '101083', '101086', '101090', '102347', '102551', '102552'],
}

class PartNetObjModel:
    def __init__(self, obj_id, obj_cat, num_envs=1, num_surface_samples=100, target_open_ratio=0.6, target_handle_direction=np.array([0., -1., 0.]), device='cuda:0'):
        """
        Create a Kinematic Model from partnet URDF File using pybulet
        
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
        self.obj_type = 'partnet'
        self.obj_scale = 1.0
        # locate object urdf, usd, config, visualize dir
        self.obj_urdf_dir = osp.join(partnet_dir, 'dataset', self.obj_id, 'mobility.urdf')
        self.obj_usd_dir = osp.join(partnet_dir, 'process', self.obj_cat, self.obj_id, 'mobility.usd')
        self.obj_config_dir = osp.join(partnet_dir, 'process', self.obj_cat, self.obj_id, 'config.yaml')
        self.obj_visualize_dir = osp.join(partnet_dir, 'process', self.obj_cat, self.obj_id, 'Visualize')
        # if osp.exists(self.obj_visualize_dir): shutil.rmtree(self.obj_visualize_dir)
        os.makedirs(self.obj_visualize_dir, exist_ok=True)
        
        # load object config
        self.obj_config = load_yaml(self.obj_config_dir)
        # load object joint infos
        self.obj_joint_infos = []
        self.load_joint_infos(load_json(self.obj_urdf_dir.replace('mobility.urdf', 'result.json'))[0], self.obj_joint_infos)
        # locate object handles
        self.obj_handles = []
        for info in self.obj_joint_infos:
            # locate handles
            if 'handle' in info: self.obj_handles.extend(info[-1])
            # locate lids for box
            if self.obj_cat in ['box'] and 'rotation_lid' in info: self.obj_handles.extend(info[-1])
            # locate lids for toilet
            if self.obj_cat in ['toilet'] and 'lid' in info: self.obj_handles.extend(info[-1])
            # locate lids for trashcan
            if self.obj_cat in ['trashcan'] and 'lid' in info: self.obj_handles.extend(info[-1])
            # locate screen for laptop
            if self.obj_cat in ['laptop'] and 'screen' in info: self.obj_handles.extend(info[-1])
            # locate switch for faucet
            if self.obj_cat in ['faucet'] and 'switch' in info: self.obj_handles.extend(info[-1])
            # locate door for washingmachine
            if self.obj_cat in ['washingmachine'] and 'door' in info: self.obj_handles.extend(info[-1])
        
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
        if self.obj_cat in ['cart', 'chair']: self.load_label_handles(save=False)
        else: self.load_joint_handles()
    
    
    # recursively process the nested joint_infos
    def load_joint_infos(self, config, config_infos):
        # append node_infos
        config_infos.append([config['name'], config['id']])
        if 'objs' in config: config_infos[-1].append(config['objs'])
        
        # Recursively process children if they exist
        if 'children' in config:
            for child in config['children']:
                self.load_joint_infos(child, config_infos)
    
    
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
                c_mesh = load_trimesh(c_mesh_dir)
                c_mesh.apply_scale(c_data[3])
                # transform origin mesh using collisionFramePosition and collisionFrameOrientation 
                transform_matrix = trimesh.transformations.quaternion_matrix(np.array(c_data[-1])[[-1, 0, 1, 2]]) # (quaternion: w, x, y, z)
                transform_matrix[:3, 3] = np.array(c_data[-2])
                c_mesh.apply_transform(transform_matrix) # type: ignore
                # check mesh type
                mesh_type = 'body_mesh'
                if c_mesh_dir.split('/')[-1].split('.')[0] in self.obj_handles: 
                    mesh_type = 'handle_mesh'
                    joint_link_infos['handle_name'].append(c_mesh_dir.split('/')[-1].split('.')[0])
                # append mesh list
                if isinstance(c_mesh, trimesh.Scene): joint_link_infos[mesh_type].append(trimesh.util.concatenate([mesh for mesh in c_mesh.dump()]))
                else: joint_link_infos[mesh_type].append(c_mesh)
            
            # combine all body_mesh, sample body_points
            if len(joint_link_infos['body_mesh']) > 0:
                body_mesh_combined = trimesh.util.concatenate(joint_link_infos['body_mesh'])
                body_points_combined, _ = trimesh.sample.sample_surface_even(body_mesh_combined, max(10, int(body_mesh_combined.area * self.num_surface_samples)))
                joint_link_infos['body_mesh'] = [body_mesh_combined]
                joint_link_infos['body_points'] = [trimesh.points.PointCloud(body_points_combined)]
            
            # process each handle_mesh, sample handle_points
            if len(joint_link_infos['handle_mesh']) > 0:
                for handle_mesh in joint_link_infos['handle_mesh']:
                    handle_points, _ = trimesh.sample.sample_surface_even(handle_mesh, max(10, int(handle_mesh.area * self.num_surface_samples**2)))
                    joint_link_infos['handle_points'].append(trimesh.points.PointCloud(handle_points))
            
            # group near handle_mesh
            if len(joint_link_infos['handle_mesh']) > 1:
                # get group_indices
                handle_name_grouped, handle_mesh_grouped = [], []
                group_indices = group_mesh_points(joint_link_infos['handle_points'])
                # group handle_name and handle_mesh
                for group_index in group_indices:
                    handle_name_grouped.append([joint_link_infos['handle_name'][index] for index in group_index])
                    handle_mesh_grouped.append(trimesh.util.concatenate([joint_link_infos['handle_mesh'][index] for index in group_index]))
                # reprocess handle_mesh, sample body_points
                joint_link_infos['handle_name'] = handle_name_grouped.copy()
                joint_link_infos['handle_mesh'] = handle_mesh_grouped.copy()
                joint_link_infos['handle_points'] = []
                for handle_mesh in joint_link_infos['handle_mesh']:
                    handle_points, _ = trimesh.sample.sample_surface_even(handle_mesh, max(10, int(handle_mesh.area * self.num_surface_samples**2)))
                    joint_link_infos['handle_points'].append(trimesh.points.PointCloud(handle_points))
            
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
            
            # apply transformation on handle_mesh, handle_points
            if len(self.obj_joint_link_infos[joint_name]['handle_mesh']) > 0:
                self.obj_joint_link_infos[joint_name]['handle_mesh_deformed'] = []
                self.obj_joint_link_infos[joint_name]['handle_points_deformed'] = []
                for nhandle in range(len(self.obj_joint_link_infos[joint_name]['handle_mesh'])):
                    handle_mesh_deformed = self.obj_joint_link_infos[joint_name]['handle_mesh'][nhandle].copy()
                    handle_mesh_deformed.apply_transform(transform_matrix)
                    handle_points_deformed = self.obj_joint_link_infos[joint_name]['handle_points'][nhandle].copy()
                    handle_points_deformed.apply_transform(transform_matrix)
                    if save: handle_points_deformed.export(osp.join(self.obj_visualize_dir, '{}_handle_{}.ply'.format(self.obj_joint_link_infos[joint_name]['name'], nhandle)))
                    self.obj_points.append(handle_points_deformed.vertices)
                    self.obj_joint_link_infos[joint_name]['handle_mesh_deformed'].append(handle_mesh_deformed.copy())
                    self.obj_joint_link_infos[joint_name]['handle_points_deformed'].append(handle_points_deformed.copy())
        
        # concat obj_points
        self.obj_points = np.concatenate(self.obj_points, axis=0)
    
    
    # load label handles
    def load_label_handles(self, save=False):
        # init obj_joint_handle_infos
        self.obj_joint_handle_infos = {}
        
        # label handle for cart, chair: heighest point
        if self.obj_cat in ['cart', 'chair']:
            # locate object handle points
            self.obj_handle_points = self.obj_points[self.obj_points[:, -1] > (self.obj_highest[-1] - 0.1)] # type: ignore
            # locate object handle joint_link: min distance from joint_link
            obj_joint_link_handle_distances = [np.inf for joint_index in range(self.obj_dofs)]
            for joint_index in range(self.obj_dofs):
                # locate joint_link_points 
                joint_link_points = []
                for npoint in range(len(self.obj_joint_link_infos[self.obj_joint_names[joint_index]]['body_points'])):
                    joint_link_points.append(self.obj_joint_link_infos[self.obj_joint_names[joint_index]]['body_points_deformed'][npoint].vertices)
                for npoint in range(len(self.obj_joint_link_infos[self.obj_joint_names[joint_index]]['handle_points'])):
                    joint_link_points.append(self.obj_joint_link_infos[self.obj_joint_names[joint_index]]['handle_points_deformed'][npoint].vertices)
                if len(joint_link_points) == 0: continue
                joint_link_points = np.concatenate(joint_link_points, axis=0)
                # append nearest distance
                obj_joint_link_handle_distances[joint_index] = nearest_distance_between_points(joint_link_points, self.obj_handle_points)
            # get joint index, name, type, link
            joint_index = np.argmin(obj_joint_link_handle_distances)
            joint_name = self.obj_joint_names[joint_index]
            joint_type = self.obj_joint_types[joint_index]
            joint_link = self.obj_joint_links[joint_index]
            # get handle_name, handle_index
            handle_name, handle_index = 'highest', 0
            # get handle_bbox, handle_size
            handle_bbox = trimesh.points.PointCloud(self.obj_handle_points).bounding_box.bounds
            handle_size = handle_bbox[1] - handle_bbox[0]
            # get handle_direction
            if handle_size[0] > handle_size[1]: handle_direction = np.array([0., 1., 0.])
            else: handle_direction = np.array([1., 0., 0.])
            # get handle_rotation: align hand_direction to target axis
            handle_rotation = quaternion_between_vectors(torch.tensor(self.target_handle_direction).unsqueeze(0), torch.tensor(handle_direction).unsqueeze(0)).squeeze(0).numpy()
            if self.obj_cat in ['chair']: handle_rotation = np.array([0.95371695, 0, 0, -0.30070580])
            # get closed_handle_pos, opened_handle_pos
            closed_handle_pos = np.mean(self.obj_handle_points, axis=0)
            opened_handle_pos = closed_handle_pos + handle_direction * self.target_open_ratio
            # get handle_trajectory
            handle_trajectory = np.stack([closed_handle_pos + 0.1 * handle_direction * n for n in range(11)], axis=0)
            # save object_handle_points
            if save: trimesh.points.PointCloud(self.obj_handle_points).export(osp.join(self.obj_visualize_dir, '{}_handle_0.ply'.format(joint_name)))
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
            'handle_trajectory': handle_trajectory,
        }
    
    
    # load all joint_handles
    def load_joint_handles(self):
        # init obj_joint_handle_infos
        self.obj_joint_handle_infos = {}
        # process all joint_handles
        for joint_index in range(self.obj_dofs):
            # locate joint_name and joint_link_type
            joint_name = self.obj_joint_names[joint_index]
            joint_type = self.obj_joint_types[joint_index]
            joint_link = self.obj_joint_links[joint_index]
            # locate joint_handle_names
            joint_handle_names = self.obj_joint_handle_names[joint_index]
            # assert joint is REVOLUTE and PRISMATIC
            if joint_type is None: continue
            # assert joint_handle exists
            if len(joint_handle_names) == 0: continue
            
            # skip object joints
            if self.obj_id == '103351' and joint_name == 'joint_9': continue
            if self.obj_id == '103521' and joint_name == 'joint_3': continue
            if self.obj_id == '19898' and joint_name == 'joint_4': continue
            
            # assert joint is REVOLUTE for box lid, toilet lid, trashcan lid, laptop screen, faucet switch, washingmachine door
            if self.obj_cat in ['box', 'toilet', 'trashcan', 'laptop', 'faucet', 'washingmachine'] and joint_type != 'REVOLUTE': continue
            
            # process all handles
            for handle_index in range(len(joint_handle_names)):
                
                # skip object joint handles
                if self.obj_id == '12484' and joint_name == 'joint_3' and handle_index == 1: continue
                
                # get handle_name
                handle_name = joint_handle_names[handle_index]
                # get handle dimension
                handle_bbox = self.obj_joint_link_infos[joint_name]['handle_mesh_deformed'][handle_index].bounding_box.bounds
                handle_size = handle_bbox[1] - handle_bbox[0]
                
                # get opened handle: 60% for PRISMATIC, 60 degrees for REVOLUTE
                opened_handle_ratio = self.target_open_ratio
                joint_pos = unnormalize_lower_upper(-np.ones(self.obj_dofs), np.array(self.obj_joint_lower), np.array(self.obj_joint_upper))
                joint_pos[joint_index] = unnormalize_lower_upper(opened_handle_ratio * 2 - 1, self.obj_joint_lower[joint_index], self.obj_joint_upper[joint_index])
                if joint_type == 'REVOLUTE': joint_pos[joint_index] = min(self.obj_joint_lower[joint_index] + np.pi * 60 / 180, self.obj_joint_upper[joint_index])
                self.forward_joint_link_meshes(joint_pos)
                opened_handle_points = self.obj_joint_link_infos[joint_name]['handle_points_deformed'][handle_index].vertices.copy()
                opened_handle_pos = np.mean(self.obj_joint_link_infos[joint_name]['handle_points_deformed'][handle_index].vertices, axis=0)
                
                # get disturbed handle
                joint_pos = unnormalize_lower_upper(-np.ones(self.obj_dofs), np.array(self.obj_joint_lower), np.array(self.obj_joint_upper))
                joint_pos[joint_index] = unnormalize_lower_upper(-1.0 + 0.1, self.obj_joint_lower[joint_index], self.obj_joint_upper[joint_index])
                self.forward_joint_link_meshes(joint_pos)
                disturb_handle_points = self.obj_joint_link_infos[joint_name]['handle_points_deformed'][handle_index].vertices.copy()
                disturb_handle_pos = np.mean(self.obj_joint_link_infos[joint_name]['handle_points_deformed'][handle_index].vertices, axis=0)
                
                # get closed handle
                joint_pos = unnormalize_lower_upper(-np.ones(self.obj_dofs), np.array(self.obj_joint_lower), np.array(self.obj_joint_upper))
                self.forward_joint_link_meshes(joint_pos)
                closed_handle_points = self.obj_joint_link_infos[joint_name]['handle_points_deformed'][handle_index].vertices.copy()
                closed_handle_pos = np.mean(self.obj_joint_link_infos[joint_name]['handle_points_deformed'][handle_index].vertices, axis=0)
                
                # get handle_direction
                handle_direction = np.zeros(3)
                disturbed_handle_diff = disturb_handle_pos - closed_handle_pos
                if self.obj_cat in ['box', 'toilet', 'laptop']: disturbed_handle_diff *= -1.
                if self.obj_cat in ['faucet', 'trashcan']: disturbed_handle_diff = np.array([-1, 0, 0])
                if np.abs(disturbed_handle_diff[0]) > np.abs(disturbed_handle_diff[1]): handle_direction[0] = 1.0 * np.sign(disturbed_handle_diff[0])
                else:  handle_direction[1] = 1.0 * np.sign(disturbed_handle_diff[1])
                # get handle_rotation: align hand_direction to target axis
                handle_rotation = quaternion_between_vectors(torch.tensor(self.target_handle_direction).unsqueeze(0), torch.tensor(handle_direction).unsqueeze(0)).squeeze(0).numpy()
                
                # locate handle_point and handle_indices for box, toilet, trashcan, laptop, faucet, washingmachine, bucket, kettle
                if self.obj_cat in ['box', 'toilet', 'trashcan', 'laptop', 'faucet', 'washingmachine', 'bucket', 'kettle']:
                    # locate handle_point and handle_indices
                    handle_point = closed_handle_points[np.argmax(np.linalg.norm(opened_handle_points - closed_handle_points, axis=1))]
                    handle_indices = np.abs(closed_handle_points - handle_point)[:, np.argmax(np.abs(handle_direction))] < 0.1
                    if self.obj_cat in ['faucet', 'washingmachine']: handle_indices = np.linalg.norm(closed_handle_points - handle_point, axis=1) < 0.1
                    # update handle_pos
                    opened_handle_pos = np.mean(opened_handle_points[handle_indices], axis=0)
                    disturb_handle_pos = np.mean(disturb_handle_points[handle_indices], axis=0)
                    closed_handle_pos = np.mean(closed_handle_points[handle_indices], axis=0)
                    # filter handle_points
                    self.obj_joint_link_infos[joint_name]['handle_points'][handle_index] = trimesh.points.PointCloud(self.obj_joint_link_infos[joint_name]['handle_points'][handle_index].vertices[handle_indices])
                    # trimesh.points.PointCloud(self.obj_joint_link_infos[joint_name]['handle_points_deformed'][handle_index].vertices[handle_indices]).export(osp.join(self.obj_visualize_dir, '{}_handle_{}_filtered.ply'.format(joint_name, handle_index)))
                
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
                    'handle_trajectory': self.load_handle_trajectory(joint_name, handle_index),
                }
    
    # load handle_trajectory in world frame
    def load_handle_trajectory(self, joint_name, handle_index, num_steps=10):
        # process all joints
        for joint_index in range(self.obj_dofs):
            # check joint_name
            if joint_name != self.obj_joint_names[joint_index]: continue
            
            # get handle trajectory
            handle_traj_pos_w = []
            for nstep in range(num_steps + 1):
                # get opened handle: 0.8 * 2. - 1. = 0.6
                joint_pos = unnormalize_lower_upper(-np.ones(self.obj_dofs), np.array(self.obj_joint_lower), np.array(self.obj_joint_upper))
                joint_pos[joint_index] = unnormalize_lower_upper(-1.0 + nstep * (2.0 / num_steps), self.obj_joint_lower[joint_index], self.obj_joint_upper[joint_index])
                self.forward_joint_link_meshes(joint_pos)
                handle_traj_pos_w.append(np.mean(self.obj_joint_link_infos[joint_name]['handle_points_deformed'][handle_index].vertices, axis=0))
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
    parser = argparse.ArgumentParser(description="Utility to process partnet objects.")
    parser.add_argument("--category", type=str, default="table", help="Process partnet object category.")
    # parse the arguments
    args = parser.parse_args()
    
    # locate category and obj_ids
    obj_cat = args.category
    obj_cat_ids = list(partnet_analysis[obj_cat].keys())
    
    # locate obj_with_handle
    obj_with_handle = {'objects': {}, 'infos': {}}
    # process all object ids
    for obj_id in obj_cat_ids:
        if obj_cat in Skip_Cat_Objects and obj_id in Skip_Cat_Objects[obj_cat]: continue
        if obj_cat in Pick_Cat_Objects and obj_id not in Pick_Cat_Objects[obj_cat]: continue
        try:
            # init object model
            obj_model = PartNetObjModel(obj_id, obj_cat)
            # loate all object handles
            if len(obj_model.obj_joint_handle_infos) > 0:
                for joint_handle_name, joint_handle_info in obj_model.obj_joint_handle_infos.items():
                    obj_with_handle['objects']['{}/{}/{}/{}/handle_{}'.format(obj_id, joint_handle_info['joint_name'], joint_handle_info['joint_type'], joint_handle_info['joint_link'], joint_handle_info['handle_index'])] = {}
                obj_with_handle['infos'][obj_id] = {'handle': obj_model.obj_joint_handle_infos, 'bbox': np.stack([obj_model.obj_lowest, obj_model.obj_highest], axis=0)}
        except: pass
    dump_pickle(osp.join(partnet_dir, 'process', obj_cat, 'analysis.pkl'), obj_with_handle)
    dump_yaml(osp.join(partnet_dir, 'process', obj_cat, 'analysis.yaml'), {'objects': obj_with_handle['objects']})