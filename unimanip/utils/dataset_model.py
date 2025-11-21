from torch.utils.data import Dataset
from unimanip.utils.general_utils import *


class UniManipTrajectoryDataset(Dataset):
    
    def __init__(self, text_tokenizer, image_transform, prompt_builder_fn,
                 action_past_window_size=0, action_future_window_size=0, image_past_window_size=0, image_future_window_size=0,
                 image_augmentation=False, predict_stop_token=False, normalization=False, processor=None,
                 dataset_dir=osp.join(LOG_DIR, 'MobileManiDataset/G1_Robot/Best_0'), dataset_mode='entire_depth'):
        # init dataset info
        self.dataset_dir = dataset_dir
        self.dataset_mode = dataset_mode
        # init post transformer
        self.text_tokenizer = text_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        # init hyper params
        self.action_past_window_size = action_past_window_size
        self.action_future_window_size = action_future_window_size
        self.image_past_window_size = image_past_window_size
        self.image_future_window_size = image_future_window_size
        # init processer
        self.image_augmentation = image_augmentation
        self.predict_stop_token = predict_stop_token
        self.normalization = normalization
        self.processor = processor
        
        # # save trajectory paths
        # self.save_trajectory_paths()
        # load trajectory paths
        self.load_trajectory_paths()
        self.generate_random_traj_frame_indices()
    
    
    # save trajectory paths
    def save_trajectory_paths(self, save=True):
        # locate all traj_paths: self.dataset_dir/object_task/object_type/object_group/object_id/object_name/object_train_id/trajectories/traj_000/episode_000/state_infos.pkl
        traj_paths = sorted(glob.glob(osp.join(self.dataset_dir, '*/*/*/*/*/*/trajectories/traj_*/episode_*/state_infos.pkl')))
        traj_paths = [traj_path.replace(self.dataset_dir, '')[1:] for traj_path in traj_paths]
        
        # load traj_frame_nums
        object_task_type_group_traj_info = {}
        for traj_path in tqdm.tqdm(traj_paths, desc="Saving trajectory paths"):
            # locate object_task_type_group from traj_path
            object_task_type_group = '{}/{}/{}'.format(traj_path.split('/')[0], traj_path.split('/')[1], traj_path.split('/')[2])
            if object_task_type_group not in object_task_type_group_traj_info: object_task_type_group_traj_info[object_task_type_group] = []
            # load traj_frame_nums from traj_path
            traj_frame_nums = load_pickle(osp.join(self.dataset_dir, traj_path))['action'].shape[0]
            # append object_task_type_group_traj_info[object_task_type_group]
            object_task_type_group_traj_info[object_task_type_group].append('{:03d} {}'.format(traj_frame_nums, traj_path))
        
        # return if not save
        if not save: return
        # save object_task_type_group_traj_info
        for object_task_type_group, traj_infos in object_task_type_group_traj_info.items():
            save_list_strings(osp.join(self.dataset_dir, object_task_type_group, 'trajectory_info.txt'), traj_infos)
    
    # load existing trajectory
    def load_trajectory_paths(self, save=True):
        # locate all traj_info_paths: self.dataset_dir/object_task/object_type/object_group/trajectory_info.txt
        self.traj_info_paths = sorted(glob.glob(osp.join(self.dataset_dir, '*/*/*/trajectory_info.txt')))
        # load all traj_frames and paths
        self.traj_frame_nums, self.traj_paths = [], []
        for traj_info_path in self.traj_info_paths:
            for traj_info in load_list_strings(traj_info_path):
                self.traj_frame_nums.append(int(traj_info.split(' ')[0]))
                self.traj_paths.append(traj_info.split(' ')[1].replace('\n', ''))
        # get total trajectory numbers
        self.traj_nums = len(self.traj_paths)
        # # plot distribution of trajectory_frames
        # self.plot_trajectory_frames()
    
    # generate random traj_frame_indices with [ntraj, nframe]
    def generate_random_traj_frame_indices(self, shuffle=True):
        # stack random_traj_frame_indices with [ntraj, nframe]
        self.random_traj_frame_indices = np.vstack([np.array([ntraj, nframe]) for ntraj in range(self.traj_nums) for nframe in range(self.traj_frame_nums[ntraj])])
        # shuffle random_traj_frame_indices
        if shuffle: np.random.shuffle(self.random_traj_frame_indices)
        # get total frame numbers
        self.frame_nums = self.random_traj_frame_indices.shape[0]
    
    
    def __len__(self):
        # return sampled frame_nums
        return self.frame_nums
    
    def __getitem__(self, idx):
        # locate traj_frame
        traj_frame = self.random_traj_frame_indices[idx]
        try: return self._load_trajectory_frame(traj_frame[0], traj_frame[1])
        except Exception as e:
            print("Missing file:", traj_frame[0], traj_frame[1])
            new_idx = (idx + 1) % len(self)  # Try next index
            return self.__getitem__(new_idx)
    
    
    # load trajectory frame: multi_view images + state + action
    def _load_trajectory_frame(self, ntraj, nframe):
        """================ Load Trajectory ================"""
        # locate traj_path
        traj_path = osp.join(self.dataset_dir, self.traj_paths[ntraj])
        # load episode_infos
        episode_infos = load_pickle(traj_path)
        # load episode_action: np.array, (N, 7)
        episode_action = episode_infos['action'].copy()
        episode_action = np.concatenate([episode_action[1:, :], episode_action[-1:, :]], axis=0)
        # load episode_base_pose: np.array, (N, 6)
        episode_base_pose = episode_infos['robot_base'][:, :6].copy()
        
        
        """================ Load Frame Prompt ================"""
        # 'Close/partnet/box/0000/100141-joint_0-REVOLUTE-link_0-handle_0/train_0/trajectories/traj_001/episode_006/state_infos.pkl'
        # 'Open/ycb/ycb/0001/003_cracker_box/train_0/trajectories/traj_001/episode_006/state_infos.pkl'
        # unpack trajectory info
        obj_action, obj_type, obj_group, obj_id, obj_name = self.traj_paths[ntraj].split('/')[0:5]
        frame_prompt = load_object_prompt(obj_action.lower(), obj_type, obj_group, obj_name)
        
        
        """================ Encode Frame Images ================"""
        # load rgb_image_head: np.array, (520, 520, 3)
        frame_rgb_image_head, _ = load_video(traj_path.replace('state_infos.pkl', 'rgb_image_head.mp4'), frame_index=[nframe])
        frame_rgb_image_head = frame_rgb_image_head[0]
        # load rgb_image_arm: np.array, (520, 520, 3)
        frame_rgb_image_arm, _ = load_video(traj_path.replace('state_infos.pkl', 'rgb_image_arm.mp4'), frame_index=[nframe])
        frame_rgb_image_arm = frame_rgb_image_arm[0]
        # load depth_image_head: np.array, (520, 520, 3)
        frame_depth_image_head, _ = load_video(traj_path.replace('state_infos.pkl', 'depth_image_head.mp4'), frame_index=[nframe])
        frame_depth_image_head = frame_depth_image_head[0]
        # load depth_image_arm: np.array, (520, 520, 3)
        frame_depth_image_arm, _ = load_video(traj_path.replace('state_infos.pkl', 'depth_image_arm.mp4'), frame_index=[nframe])
        frame_depth_image_arm = frame_depth_image_arm[0]
        # multi-view RGBD
        imgs = [Image.fromarray(frame_rgb_image_head), Image.fromarray(frame_depth_image_head), Image.fromarray(frame_rgb_image_arm), Image.fromarray(frame_depth_image_arm)]
        # post transform pixel_values and input_ids
        model_inputs = self.processor(text='<image>'*len(imgs) + frame_prompt, images=imgs, return_tensors="pt").to(torch.float32) # type: ignore
        input_ids = model_inputs["input_ids"]
        pixel_values = model_inputs["pixel_values"]
        input_ids = input_ids.squeeze(0) 
        
        
        """================ Encode Frame State ================"""
        # load episode_frame_hand_pose: np.array, (1, 6)
        episode_frame_hand_pose = episode_infos['robot_hand'][nframe:nframe+1, :6].copy()
        
        # encode world hand_pose to local base frame
        episode_frame_hand_pos_l, episode_frame_hand_rot_l = encode_world_pose_to_local_frame(
            torch.tensor(episode_frame_hand_pose[:, :3]), torch.tensor(episode_frame_hand_pose[:, 3:6]), torch.tensor(episode_base_pose[nframe:nframe+1, :3]), torch.tensor(episode_base_pose[nframe:nframe+1, 3:6]))
        # post transform current_state (6,) and current_state_mask (6,)
        current_state = torch.cat([episode_frame_hand_pos_l, episode_frame_hand_rot_l], dim=-1).squeeze(0).to(torch.float32)
        current_state_mask = torch.tensor(np.asarray([True]), dtype=torch.bool).repeat(current_state.shape[-1])
        
        
        """================ Encode Episode Action ================"""
        # encode world action to local base frame
        episode_action_pos_l, episode_action_rot_l = encode_world_action_to_local_frame(
            torch.tensor(episode_action[:, :3]), torch.tensor(episode_action[:, 3:6]), torch.tensor(episode_base_pose[:, :3]), torch.tensor(episode_base_pose[:, 3:6]))
        episode_action[:, :3], episode_action[:, 3:6] = episode_action_pos_l.numpy().copy(), episode_action_rot_l.numpy().copy()
        
        # post transform action_list (16, 7) and action_mask (16, 7)
        action_list, action_mask = [], []
        for idx in range(nframe - self.action_past_window_size, nframe + self.action_future_window_size + 1):
            if idx < 0:
                action_list.append(np.zeros_like(episode_action[0]))
                action_mask.append(False)
            elif idx >= len(episode_action):
                action_list.append(np.zeros_like(episode_action[-1]))
                action_mask.append(True)
            else:
                action_list.append(episode_action[idx])
                action_mask.append(True)
        action_list = torch.tensor(np.asarray(action_list), dtype=torch.float32)
        action_mask = torch.tensor(np.asarray(action_mask), dtype=torch.bool).unsqueeze(-1).repeat(1, action_list.shape[-1])
        
        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=None,
            actions=action_list,
            action_masks=action_mask,
            current_state=current_state,
            current_state_mask=current_state_mask,
        )


# UniManipDataset = UniManipTrajectoryDataset(text_tokenizer=None, image_transform=None, prompt_builder_fn=None)