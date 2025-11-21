from unimanip.utils.general_utils import *

# locate ycb dataset
ycb_dir = osp.join(ASSET_DIR, 'ycb')
ycb_analysis = load_yaml(osp.join(ycb_dir, 'analysis.yaml'))

class YCBObjModel:
    def __init__(self, obj_id, obj_cat, num_envs=1, num_surface_samples=100, target_open_ratio=0.6, target_handle_direction=np.array([0., -1., 0.]), device='cuda:0'):
        """
        Create an Object Model from ycb OBJ File
        
        Parameters
        ----------
        obj_id: str, object id
        obj_cat: str, object category
        num_envs: int, number of envs
        device: str, torch.Device, device for torch tensors
        """
        
        # init basic
        self.device = device
        self.num_envs = num_envs
        # init object id and category
        self.obj_id = obj_id
        self.obj_cat = obj_cat
        self.obj_type = 'ycb'
        self.obj_scale = 1.0
        # locate object usd, obj dir
        self.obj_usd_dir = osp.join(ycb_dir, self.obj_id, 'google_16k/textured.usd')
        self.obj_mesh_dir = osp.join(ycb_dir, self.obj_id, 'google_16k/textured.obj')
        self.obj_points = trimesh.load_mesh(self.obj_mesh_dir).vertices
        self.obj_points_center = np.mean(self.obj_points, axis=0)
        
        # get object canonical points
        self.obj_points_canonical = self.obj_points.copy()
        self.obj_highest = np.max(self.obj_points_canonical, axis=0)
        self.obj_lowest = np.min(self.obj_points_canonical, axis=0)
        # get object bbox, center, size
        self.obj_bbox = trimesh.points.PointCloud(self.obj_points_canonical).bounding_box.bounds
        self.obj_center = (self.obj_bbox[1] + self.obj_bbox[0]) / 2
        self.obj_size = self.obj_bbox[1] - self.obj_bbox[0]
    
    # process IsaacLab RigidObjectCfg
    def process_obj_sim_config(self, obj_rig_config, obj_joint_handle_name=None):        
        # InitialStateCfg
        obj_rig_config.init_state.pos = (obj_rig_config.init_state.pos[0], obj_rig_config.init_state.pos[1], 0.02 - self.obj_lowest[-1] * self.obj_scale)
        obj_rig_config.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        # Scale object points center
        self.obj_points_center *= self.obj_scale
        # Rotate object in Simulation
        self.obj_points_rotated = self.obj_points.copy()
        # Update object bbox, center, size
        self.obj_bbox_rotated = trimesh.points.PointCloud(self.obj_points_rotated).bounding_box.bounds
        self.obj_center_rotated = (self.obj_bbox_rotated[1] + self.obj_bbox_rotated[0]) / 2
        self.obj_size_rotated = self.obj_bbox_rotated[1] - self.obj_bbox_rotated[0]


if __name__ == '__main__':
    pass