import os
import sys
import glob
import json
import tqdm
import torch
import decord
import ffmpeg
import shutil
import random
import trimesh
import datetime
import argparse
import cv2 as cv
import numpy as np
import pandas as pd
import open3d as o3d
from PIL import Image
import os.path as osp
import matplotlib.pyplot as plt
from isaaclab.utils.io import *
from scipy.spatial import cKDTree # type: ignore

# locate project and asset folders
PROJECT_DIR =  osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
CONTAINER_DIR = osp.join(osp.dirname(PROJECT_DIR), 'Container')
ASSET_DIR = osp.join(osp.dirname(PROJECT_DIR), 'Assets')
LOG_DIR = osp.join(osp.dirname(PROJECT_DIR), 'Logs')
SERVER = False

# locate server folders
if not osp.exists(ASSET_DIR):
    ASSET_DIR = '/mnt/blob/Desktop/Assets/'
    LOG_DIR = '/mnt/blob/Desktop/Logs/'
    SERVER = True

# global definitions 
SCALE2STR = {0.06: '006', 0.08: '008', 0.10: '010', 0.12: '012', 0.15: '015'}
STR2SCALE = {'006': 0.06, '008': 0.08, '010': 0.10, '012': 0.12, '015': 0.15}
# global analysis
SCENE_ANALYSIS = load_yaml(osp.join(PROJECT_DIR, 'unimanip/configs/data/analysis_scene.yaml'))
PARTNET_ANALYSIS = load_yaml(osp.join(PROJECT_DIR, 'unimanip/configs/data/analysis_partnet.yaml'))
UNIDOOR_ANALYSIS = load_yaml(osp.join(PROJECT_DIR, 'unimanip/configs/data/analysis_unidoor.yaml'))
YCB_ANALYSIS = load_yaml(osp.join(PROJECT_DIR, 'unimanip/configs/data/analysis_ycb.yaml'))

# global object groups
OBJECT_GROUPS = {
    'partnet': ["box", "cart", "chair", "dishwasher", "faucet", "laptop", "microwave", "refrigerator", "storage_furniture", "table", "toilet", "trashcan", "washingmachine", "bucket", "kettle"],
    'unidoor': ["cabinet", "car", "fridge", "lever_door", "round_door", "window", "safe"],
}

"""================ Load, Save Files, Images ================"""

# load json file
def load_json(path):
    with open(path, 'r') as file: 
        data = json.load(file)
    return data

# dump json file
def dump_json(path, data):
    with open(path, 'w') as file: 
        json.dump(data, file)
        

# load image as numpy array
def load_image(img_dir):
    return np.array(Image.open(img_dir))

# save numpy array image
def save_image(img_dir, img):
    Image.fromarray(img).save(img_dir)

# load video length
def load_video_length(video_dir):
    video_reader = decord.VideoReader(video_dir)
    num_frames = len(video_reader)
    del video_reader
    return num_frames

# load video as numpy array
def load_video(video_dir, frame_index=None, sample_ratio=1.):
    # load video
    video_reader = decord.VideoReader(video_dir)
    frame_number = len(video_reader)
    # load frame_index
    if frame_index is None:
        frame_index = list(range(frame_number))
        if sample_ratio != 1.: frame_index = np.random.choice(frame_index, replace=False, size=max(1, int(frame_number * sample_ratio)))
    # load frame image
    video = video_reader.get_batch(frame_index).asnumpy()
    del video_reader
    return video, frame_index

# save image list as mp4 video
def save_video(video_dir, images, crf=18):
    # render the images
    height, width, _ = images[0].shape
    out = (
        ffmpeg
        .input('pipe:0', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
        .output(video_dir, reset_timestamps=1, **{'preset': 'medium', 'b:v': '0', 'c:v':'libx264', 'crf': str(crf)})
        .overwrite_output()
        .run_async(quiet=True, pipe_stdin=True, pipe_stderr=True)
    )
    for frame in images: out.stdin.write(frame.tobytes())
    out.stdin.close()
    out.wait()


# load a list of strings from txt
def load_list_strings(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
    return lines

# save a list of strings into txt
def save_list_strings(filename, data):
    with open(filename, "w") as file:
        for string in data:
            file.write(string + "\n")


# load trimesh
def load_trimesh(mesh_dir):
    return trimesh.load_mesh(mesh_dir)

# save trimesh
def save_trimesh(mesh_dir, mesh):
    mesh.export(mesh_dir)


"""================ Translation, Rotation, Transformation ================"""

def quaternion_conjugate(q):
    """
    Compute the conjugate of a batch of quaternions.
    
    Args:
        q (torch.Tensor): Tensor of shape (batch_size, 4), where each row is (w, x, y, z).
    
    Returns:
        torch.Tensor: Tensor of shape (batch_size, 4), conjugated quaternions.
    """
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack((w, -x, -y, -z), dim=-1)

def quaternion_multiply(q1, q2):
    """
    Perform quaternion multiplication for a batch of quaternions for vector components.
    Args:
        q1: Tensor of shape (batch_size, 4) representing the first quaternion batch.
        q2: Tensor of shape (batch_size, 4) representing the second quaternion batch.
    Returns:
        Tensor of shape (batch_size, 4) representing the resulting quaternion batch.
    """
    # Separate scalar (w) and vector (x, y, z) parts
    w1, v1 = q1[..., 0], q1[..., 1:]
    w2, v2 = q2[..., 0], q2[..., 1:]
    # Compute the scalar (w) part
    w_r = w1 * w2 - torch.sum(v1 * v2, dim=-1)
    # Compute the vector (x, y, z) part
    v_r = w1.unsqueeze(-1) * v2 + w2.unsqueeze(-1) * v1 + torch.cross(v1, v2, dim=-1)
    # Combine scalar and vector parts
    return torch.cat((w_r.unsqueeze(-1), v_r), dim=-1)

def quaternion_difference(q_target, q_current):
    """
    Compute the quaternion difference between target and current orientations: q_target*q_current^-1
    
    Args: 
        q_target (torch.Tensor): Tensor of shape (batch_size, 4) for target quaternion orientations (w, x, y, z).
        q_current (torch.Tensor): Tensor of shape (batch_size, 4) for current quaternion orientations (w, x, y, z).
    """
    return quaternion_multiply(q_target, quaternion_conjugate(q_current))

def quaternion_relative(q, q_base):
    """
    Compute the quaternion q relative to q_base: q_base^-1*q
    
    Args: 
        q (torch.Tensor): Tensor of shape (batch_size, 4) for quaternion orientations (w, x, y, z).
        q_base (torch.Tensor): Tensor of shape (batch_size, 4) for base quaternion orientations (w, x, y, z).
    """
    return quaternion_multiply(quaternion_conjugate(q_base), q)

def quaternion_rotate_vector(q, v):
    """
    Rotate vector with quaternion: q*v*q^-1
    
    Args:
        q (torch.Tensor): Tensor of shape (batch_size, 4) for quaternion orientations (w, x, y, z).
        v (torch.Tensor): Tensor of shape (batch_size, 3) for vector.
    
    Returns:
        torch.Tensor: Tensors of shape (batch_size, 3) for new vector.
    """
    # convert vectors to quaternion form (w=0 for vectors)
    v = torch.cat((torch.zeros(v.shape[0], 1, device=v.device), v), dim=-1)
    # Rotate vector: q_conjugate * forces_q * q
    return quaternion_multiply(quaternion_multiply(q, v), quaternion_conjugate(q))[:, 1:]  # Extract vector part

def quaternion_frame_vector(q, v):
    """
    Map vector(force, torque) into quaternion frame: q^-1*v*q
    
    Args:
        v (torch.Tensor): Tensor of shape (batch_size, 3) for vector.
        q (torch.Tensor): Tensor of shape (batch_size, 4) for quaternion orientations (w, x, y, z).
    
    Returns:
        torch.Tensor: Tensors of shape (batch_size, 3) for new vector.
    """
    # convert vectors to quaternion form (w=0 for vectors)
    v = torch.cat((torch.zeros(v.shape[0], 1, device=v.device), v), dim=-1)
    # Rotate vector: q_conjugate * forces_q * q
    return quaternion_multiply(quaternion_multiply(quaternion_conjugate(q), v), q)[:, 1:]  # Extract vector part

def quaternion_between_vectors(v_target, v_current):
    """
    Compute the quaternion that rotates v_current to v_target in batch mode.
    
    Args:
        v_target: Tensor of shape (batch, 3) - target vectors (must be normalized)
        v_current: Tensor of shape (batch, 3) - current vectors (must be normalized)
    
    Returns:
        Quaternion tensor of shape (batch, 4) in [w, x, y, z] format.
    """
    v_target = torch.nn.functional.normalize(v_target, dim=1)
    v_current = torch.nn.functional.normalize(v_current, dim=1)
    axis = torch.cross(v_current, v_target, dim=1)
    angle = torch.acos(torch.clamp(torch.sum(v_current * v_target, dim=1, keepdim=True), -1.0, 1.0))
    axis_norm = torch.norm(axis, dim=1)
    
    # Handle near-zero axis cases (nearly identical vectors)
    zero_axis_mask = axis_norm < 1e-6
    axis[zero_axis_mask] = torch.tensor([1.0, 0.0, 0.0], dtype=axis.dtype, device=axis.device)  # Default axis
    
    # Handle opposite vectors (180-degree rotation)
    opposite_mask = torch.abs(torch.sum(v_current * v_target, dim=1) + 1.0) < 1e-6
    # **Automatically select a valid perpendicular axis**
    arbitrary_vector = torch.tensor([1.0, 0.0, 0.0], dtype=axis.dtype, device=axis.device).expand_as(v_current)
    use_alternate_vector = torch.norm(torch.cross(arbitrary_vector, v_current, dim=1)) < 1e-6
    # If v_current is parallel to (1,0,0), choose (0,1,0) instead
    arbitrary_vector[use_alternate_vector] = torch.tensor([0.0, 1.0, 0.0], dtype=axis.dtype, device=axis.device)
    default_axis = torch.nn.functional.normalize(torch.cross(v_current, arbitrary_vector, dim=1), dim=1)
    # Apply default axis when vectors are opposite
    axis[opposite_mask] = default_axis[opposite_mask]
    
    axis = torch.nn.functional.normalize(axis, dim=1)
    q_w = torch.cos(angle / 2)
    q_xyz = axis * torch.sin(angle / 2)
    return torch.cat([q_w, q_xyz], dim=1)

def combine_transformation(q1, v1, q2, v2):
    """
    Combine two transformations (q1, v1) and (q2, v2).
    
    Parameters:
        q1, q2: Quaternion (batch, 4) -> (w, x, y, z)
        v1, v2: Translation vector (batch, 3) -> (x, y, z)

    Returns:
        q: Combined quaternion (batch, 4)
        v: Combined translation (batch, 3)
    """
    q = quaternion_multiply(q1, q2)  # q = q2 * q1
    v = v1 + quaternion_rotate_vector(q1, v2)  # v = v2 + q2 * v1
    return q, v


def axis_angle_to_quaternion(axis, angle):
    """
    Convert a batch of axis-angle representations to quaternions.
    
    Args:
        axis (torch.Tensor): A tensor of shape (batch_size, 3) where each row is an axis (u_x, u_y, u_z).
        angle (torch.Tensor): A tensor of shape (batch_size,) representing angles in radians.
        
    Returns:
        torch.Tensor: A tensor of shape (batch_size, 4) where each row is the quaternion (w, x, y, z).
    """
    # Normalize the axis to ensure it's a unit vector
    axis = axis / axis.norm(dim=-1, keepdim=True)
    # Compute quaternion components
    w = torch.cos(angle / 2.0)
    x = axis[:, 0] * torch.sin(angle / 2.0)
    y = axis[:, 1] * torch.sin(angle / 2.0)
    z = axis[:, 2] * torch.sin(angle / 2.0)
    # Stack the results to form the quaternion
    quaternions = torch.stack((w, x, y, z), dim=-1)
    return quaternions

def quaternion_to_axis_angle(quaternions):
    """
    Convert a batch of quaternions to axis-angle representation.
    
    Args:
        quaternions (torch.Tensor): Tensor of shape (batch_size, 4), where each row is (w, x, y, z).
        
    Returns:
        axes (torch.Tensor): Tensor of shape (batch_size, 3) representing rotation axes.
        angles (torch.Tensor): Tensor of shape (batch_size,) representing rotation angles in radians.
    """
    # Split quaternion components
    w, x, y, z = quaternions.unbind(dim=-1)
    # Compute rotation angle
    angles = 2.0 * torch.arccos(w.clamp(-1.0, 1.0))  # Clamp w to avoid numerical errors outside [-1, 1]
    # Compute rotation axis
    sin_theta_half = torch.sqrt(1.0 - w**2).clamp(min=1e-8)  # Avoid division by zero
    axes = torch.stack((x, y, z), dim=-1) / sin_theta_half.unsqueeze(-1)
    # Handle zero rotation (angle close to 0)
    axes[angles < 1e-8] = torch.tensor([0.0, 0.0, 0.0], device=quaternions.device)
    return axes, angles


def euler_angle_to_quaternion(euler_angle):
    """
    Convert a batch of euler-angle representations to quaternions.
    
    Args:
        euler_angle (torch.Tensor): A tensor of shape (batch_size, 3).
        
    Returns:
        torch.Tensor: A tensor of shape (batch_size, 4) where each row is the quaternion (w, x, y, z).
    """
    # Unpack roll, pitch, yaw
    roll, pitch, yaw = euler_angle[:, 0] / 2, euler_angle[:, 1] / 2, euler_angle[:, 2] / 2
    c1, s1 = torch.cos(yaw), torch.sin(yaw)
    c2, s2 = torch.cos(pitch), torch.sin(pitch)
    c3, s3 = torch.cos(roll), torch.sin(roll)
    # Compute quaternion components
    x = s3 * c2 * c1 - c3 * s2 * s1
    y = c3 * s2 * c1 + s3 * c2 * s1
    z = c3 * c2 * s1 - s3 * s2 * c1
    w = c3 * c2 * c1 + s3 * s2 * s1
    return torch.stack((w, x, y, z), dim=-1)

def quaternion_to_euler_angle(quaternions):
    """
    Convert a batch of quaternions to Euler angles.

    Args:
        quaternions (torch.Tensor): Tensor of shape (batch_size, 4) in (w, x, y, z) format.
    Returns:
        euler_angles (torch.Tensor): Tensor of shape (batch_size, 3), representing Euler angles (roll, pitch, yaw) in radians.
    """
    # Extract components
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]

    # Compute Euler angles
    # Roll (X-axis rotation)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    # Pitch (Y-axis rotation) - Clamp to avoid NaNs
    sinp = 2 * (w * y - z * x)
    pitch = torch.asin(sinp.clamp(-1.0, 1.0))
    # Yaw (Z-axis rotation)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

    return torch.stack((roll, pitch, yaw), dim=-1)  # Shape (batch_size, 3)


def encode_world_action_to_local_frame(action_pos_w, action_rot_w, local_pos_w, local_rot_w):
    """
    Convert a 6-DoF delta action (position + euler rot) in world frame
    to the local frame defined by (local_pos_w, local_rot_w).
    Args:
        action_pos_w: (N, 3) position delta in world frame
        action_rot_w: (N, 3) rotation delta in world frame (Euler)
        local_pos_w:  (N, 3) local frame origin in world
        local_rot_w:  (N, 3) local frame rotation in world (Euler)
    Returns:
        action_pos_l: (N, 3) position delta in local frame
        action_rot_l: (N, 3) rotation delta in local frame (Euler)
    """
    # Convert to quaternion
    q_action_w = euler_angle_to_quaternion(action_rot_w)  # (N, 4)
    q_local_w  = euler_angle_to_quaternion(local_rot_w)   # (N, 4)
    # Transform position from world to local frame
    action_pos_l = quaternion_frame_vector(q_local_w, action_pos_w)  # (N, 3)
    # Transform rotation from world to local frame: q_rel = q_frame^-1 * q_action
    action_rot_l = quaternion_to_euler_angle(quaternion_relative(q_action_w, q_local_w))  # (N, 3)
    return action_pos_l, action_rot_l

def decode_local_action_from_local_frame(action_pos_l, action_rot_l, local_pos_w, local_rot_w):
    """
    Convert a 6-DoF delta action (position + euler rot) in local frame
    to the world frame using the local frame's world pose (local_pos_w, local_rot_w).
    Args:
        action_pos_l: (N, 3) delta position in local frame
        action_rot_l: (N, 3) delta euler rotation in local frame
        local_pos_w:  (N, 3) local frame origin in world
        local_rot_w:  (N, 3) local frame rotation in world
    Returns:
        action_pos_w: (N, 3) delta position in world frame
        action_rot_w: (N, 3) delta rotation in world frame (Euler)
    """
    # Convert to quaternion
    q_action_l = euler_angle_to_quaternion(action_rot_l)  # (N, 4)
    q_local_w  = euler_angle_to_quaternion(local_rot_w)   # (N, 4)
    # Transform position into world frame
    action_pos_w = quaternion_rotate_vector(q_local_w, action_pos_l)  # (N, 3)
    # Transform rotation into world frame: q_world = q_frame * q_local
    action_rot_w = quaternion_to_euler_angle(quaternion_multiply(q_local_w, q_action_l))
    return action_pos_w, action_rot_w


def encode_world_pose_to_local_frame(pose_pos_w, pose_rot_w, local_pos_w, local_rot_w):
    """
    Convert a 6-DoF absolute pose (position + euler rot) in the world frame
    to the local frame defined by (local_pos_w, local_rot_w).
    Args:
        pose_pos_w:  (N, 3) absolute position in world frame
        pose_rot_w:  (N, 3) absolute euler rotation in world frame
        local_pos_w: (N, 3) local frame origin in world
        local_rot_w: (N, 3) local frame rotation in world (Euler)
    Returns:
        pose_pos_l: (N, 3) absolute position in local frame coordinates
        pose_rot_l: (N, 3) absolute euler rotation in local frame coordinates
    """
    # Convert to quaternions
    q_pose_w  = euler_angle_to_quaternion(pose_rot_w)   # (N, 4)
    q_local_w = euler_angle_to_quaternion(local_rot_w)  # (N, 4)
    # Transform position from world to local frame: translate into local frame origin, then rotate by local frame^{-1}
    pose_pos_l = quaternion_frame_vector(q_local_w, pose_pos_w - local_pos_w)  # (N, 3)
    # Transform rotation from world to local frame: q_l = q_local^{-1} * q_pose
    pose_rot_l = quaternion_to_euler_angle(quaternion_relative(q_pose_w, q_local_w))  # (N, 3)
    return pose_pos_l, pose_rot_l

def decode_local_pose_to_world_frame(pose_pos_l, pose_rot_l, local_pos_w, local_rot_w):
    """
    Convert a 6-DoF absolute pose (position + euler rot) in the local frame
    back to the world frame using the local frame's world pose (local_pos_w, local_rot_w).
    Args:
        pose_pos_l:  (N, 3) absolute position in local frame coordinates
        pose_rot_l:  (N, 3) absolute euler rotation in local frame coordinates
        local_pos_w: (N, 3) local frame origin in world
        local_rot_w: (N, 3) local frame rotation in world (Euler)
    Returns:
        pose_pos_w: (N, 3) absolute position in world frame
        pose_rot_w: (N, 3) absolute euler rotation in world frame
    """
    # Convert to quaternions
    q_pose_l  = euler_angle_to_quaternion(pose_rot_l)   # (N, 4)
    q_local_w = euler_angle_to_quaternion(local_rot_w)  # (N, 4)
    # Transform position from world to local frame: rotate local coords into world, then translate by local frame origin
    pose_pos_w = local_pos_w + quaternion_rotate_vector(q_local_w, pose_pos_l)
    # Transform rotation from world to local frame: q_w = q_local * q_l
    pose_rot_w = quaternion_to_euler_angle(quaternion_multiply(q_local_w, q_pose_l))  # (N, 3)
    return pose_pos_w, pose_rot_w


def radius_between_vectors(vector0, vector1):
    """
    Compute the angle radius between two vectors.
    
    Parameters:
        vector0 (torch.Tensor): Shape (N, V, 3).
        vector1 (torch.Tensor): Shape (N, V, 3).
        
    Returns:
        angle_radius (torch.Tensor): Shape (N, V).
    """
    return torch.acos(torch.clamp(torch.sum(vector0 * vector1, dim=-1) / (torch.norm(vector0, dim=-1) * torch.norm(vector1, dim=-1)), -1.0, 1.0))


"""================ Compute ================"""

# normalize value(batch, size) from (lower, upper) to (-1, 1)
def normalize_lower_upper(value, lower, upper):
    return ((value - lower) / (upper - lower)) * 2 -1

# unnormalize value(batch, size) from (-1, 1) to (lower, upper)
def unnormalize_lower_upper(value, lower, upper):
    return ((value + 1) * (upper - lower) / 2) + lower


@torch.jit.script
def batch_quat_apply(a, b) -> torch.Tensor:
    # unsqueeze a(Nenv, 1, 4)
    shape = b.shape
    a = a.unsqueeze(1)
    # extract the xyz component of quaternion a
    xyz = a[:, :, 1:]
    # compute the cross product t
    t = torch.cross(xyz, b, dim=-1) * 2
    # compute the final result and reshape it to the original shape
    return (b + a[:, :, :1] * t + torch.cross(xyz, t, dim=-1)).view(shape)

@torch.jit.script
# compute sided distance from sources(Nenv, Ns, 3) to targets(Nenv, Nt, 3)
def batch_sided_distance(sources, targets):
    # pairwise_distances: (Nenv, Ns, Nt)
    pairwise_distances = torch.cdist(sources, targets)
    # find the minimum distances
    distances, _ = torch.min(pairwise_distances, dim=-1)
    return distances

# simplify trimesh vertices
def simplify_trimesh(mesh, ratio=0.1, min_faces=None):
    # # simplify trimesh
    # mesh = mesh.simplify_quadric_decimation(1)
    # init open3d mesh
    temp_mesh = o3d.geometry.TriangleMesh()
    temp_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    temp_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    # use open3d to simplify mesh
    num_faces = int(len(temp_mesh.triangles)*ratio)
    if min_faces is not None: num_faces = max(min_faces, num_faces)
    temp_mesh = temp_mesh.simplify_quadric_decimation(target_number_of_triangles=num_faces)
    # return trimesh
    return trimesh.Trimesh(vertices=np.asarray(temp_mesh.vertices), faces=np.asarray(temp_mesh.triangles), process=True)

# nearest distance between two point clouds
def nearest_distance_between_points(points1, points2):
    dist1, _ = cKDTree(points1).query(points2)  # Nearest from points2 to points1
    dist2, _ = cKDTree(points2).query(points1)  # Nearest from points1 to points2
    return max(np.min(dist1), np.min(dist2))

# group mesh_points where the nearest distance is below a threshold
def group_mesh_points(mesh_points, threshold=0.1):
    # Adjacency matrix for grouping
    num_meshes = len(mesh_points)
    adjacency_matrix = np.zeros((num_meshes, num_meshes), dtype=bool)
    # Distance between mesh_points
    for i in range(num_meshes):
        for j in range(i + 1, num_meshes):
            dist = nearest_distance_between_points(mesh_points[i].vertices, mesh_points[j].vertices)
            if dist < threshold: adjacency_matrix[i, j] = adjacency_matrix[j, i] = True
    # Find connected components
    groups, visited = [], set()
    def dfs(idx, group):
        """Depth-First Search to find connected components."""
        visited.add(idx)
        group.append(idx)
        for j in range(num_meshes):
            if adjacency_matrix[idx, j] and j not in visited: dfs(j, group)
    # Group connected components
    for i in range(num_meshes):
        if i not in visited:
            group = []
            dfs(i, group)
            groups.append(group)
    return groups

# compute encoding vector (nenv, dimension) for time (nenv, )
def time_encoding(time, dimension):
    # Create a tensor for dimension indices: [0, 1, 2, ..., dimension-1]
    div_term = torch.arange(0, dimension, 2, dtype=torch.float32) * -(torch.log(torch.tensor(10000.0)) / dimension)
    div_term = torch.exp(div_term).unsqueeze(0).to(time.device)  # Shape: (1, dimension/2)
    # Apply sin to even indices in the array; 2i
    encoding = torch.zeros(time.shape[0], dimension).to(time.device)
    encoding[:, 0::2] = torch.sin(time.unsqueeze(1) * div_term)
    # Apply cos to odd indices in the array; 2i+1
    encoding[:, 1::2] = torch.cos(time.unsqueeze(1) * div_term)
    return encoding


"""================ Plot ================"""

# plot current and target values (batch, step)
def plot_current_target(current, target, save_name=None):
    # get batch and step
    batch, step = current.shape
    x = np.arange(step)
    # plot current and target for each batch
    for nb in range(batch):
        c, t = current[nb], target[nb]
        plt.figure(figsize=(8, 4))
        plt.plot(x, t, label="target", color="r", linewidth=2)
        plt.plot(x, c, label="current", color="b", linewidth=2)
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title("Current_Target_{}".format(nb))
        plt.legend()
        if save_name is not None: plt.savefig(save_name + '_{}.png'.format(nb))


# compare successes
def compare_successes(current_dir, previous_dir):
    # load current and previous success.txt
    current_success = load_list_strings(current_dir)
    previous_success = load_list_strings(previous_dir)
    # init currrent and previous successes
    current_success_lines = []
    previous_success_lines = []
    current_success_successes = []
    previous_success_successes = []
    # compare current with previous successes
    for current_line in current_success[:-2]:
        current_line_object = current_line.split(',')[-2]
        for previous_line in previous_success:
            if current_line_object in previous_line:
                current_success_lines.append(current_line.split('\n')[0])
                previous_success_lines.append(previous_line.split('\n')[0])
                current_success_successes.append(float(current_line.split(',')[2]))
                previous_success_successes.append(float(previous_line.split(',')[2]))
                break
    print('current_success:', np.mean(current_success_successes))
    print('previous_success:', np.mean(previous_success_successes))


# load vla train jsonl
def load_vla_train_jsonl(jsonl_dir):
    # read jsonl
    df = pd.read_json(jsonl_dir, lines=True)
    df = df.sort_values("VLA Train/Step").reset_index(drop=True)
    df["VLA Train/Loss_EMA"] = df["VLA Train/Loss"].ewm(span=200, adjust=False).mean()
    # plot jsonl
    fig, ax = plt.subplots(figsize=(15, 5))
    df.plot(x="VLA Train/Step", y="VLA Train/Loss", ax=ax, alpha=0.35, label="Loss (Raw)")
    df.plot(x="VLA Train/Step", y="VLA Train/Loss_EMA", ax=ax, label=f"Loss (EMA)", linewidth=2)
    ax.set_xlabel("VLA Train Step")
    ax.set_ylabel("VLA Train Loss")
    ax.set_title("VLA Train Loss (EMA Smoothed)")
    fig.savefig(jsonl_dir.replace('jsonl', 'png'), dpi=300)


# load vla test success
def load_vla_test_success(obj_folder, test_name='', obj_range=[], vla_name='vla_trajectories_-01', traj_range=[], episode_range=[]):
    # locate objects
    if len(obj_range) == 0: obj_names = sorted(os.listdir(obj_folder))
    else: obj_names = ['{:04d}'.format(obj) for obj in obj_range]
    # init vla result
    vla_result_dict = {'object': [], 'success': []}
    # process all objects
    for obj_name in obj_names:
        # locate all object vla results
        if len(traj_range) == 0 and len(episode_range) == 0:
            vla_result_fns = sorted(glob.glob(osp.join(obj_folder, obj_name, '*/*/{}/*/*/state_infos.pkl'.format(vla_name))))
        # filter traj and episode
        else:
            vla_result_fns = []
            for traj in traj_range:
                for episode in episode_range:
                    vla_result_fns += sorted(glob.glob(osp.join(obj_folder, obj_name, '*/*/{}/traj_{:03d}/episode_{:03d}/state_infos.pkl'.format(vla_name, traj, episode))))
        if len(vla_result_fns) == 0: continue
        # load vla successes
        vla_result_successes = np.asarray([(np.sum(load_pickle(vla_result_fn)['success']) > 0) * 1. for vla_result_fn in vla_result_fns])
        # append vla_result_dict
        vla_result_dict['success'].append(np.mean(vla_result_successes))
        vla_result_dict['object'].append('{} {:.3f}'.format(obj_name, vla_result_dict['success'][-1]))
    # save result
    save_list_strings(osp.join(obj_folder, '{}_{}.txt'.format(test_name, vla_name)), vla_result_dict['object'])
    # print result
    print('object name: {}, num: {}, success_mean: {:.3f}, success_std: {:.3f}'.format(obj_folder.split('/')[-1], len(vla_result_dict['object']), np.mean(vla_result_dict['success']), np.std(vla_result_dict['success'])))
    return vla_result_dict


# load existing trajectory info
def load_trajectory_info(config_name, traj_size=16, episode_size=10):
    # init traj_info
    traj_info = {}
    print('====== Existing Trajectory Info: {} ======'.format(config_name))
    # load train_info from config
    train_info = load_yaml(osp.join(PROJECT_DIR, 'unimanip/configs/train', '{}.yaml'.format(config_name)))
    train_folder, train_task = train_info['Infos']['name'], train_info['Agents']['name']
    # read all object_groups
    for object_type, object_groups in OBJECT_GROUPS.items():
        traj_info[object_type] = {}
        for object_group in object_groups:
            # locate group_info
            group_info = load_yaml(osp.join(ASSET_DIR, object_type, 'process', object_group, 'analysis.yaml'))
            group_folder = osp.join(LOG_DIR, train_folder, train_task, object_type, object_group)
            group_object_num, group_traj_target = len(group_info['objects']), len(group_info['objects']) * traj_size * episode_size
            # non existing group_folder
            if not osp.exists(group_folder): group_traj_paths = []
            # read all traj*_episode*
            else: group_traj_paths = sorted(glob.glob(osp.join(group_folder, '*/*/*/trajectories/traj_*/episode_*/state_infos.pkl')))
            # save existing trajectory infos
            group_traj_process = len(group_traj_paths)
            traj_info[object_type][object_group] = {'paths': group_traj_paths, 'process': group_traj_process, 'target': group_traj_target, 'object': group_object_num}
            print('{:10} || {:20} || object {:3} || processed {:6} / {:6} = {:.3f}'.format(object_type, object_group, group_object_num, group_traj_process, group_traj_target, group_traj_process / group_traj_target))
    return traj_info


# load object prompt: <action> <object>
def load_object_prompt(obj_action, obj_type, obj_group, obj_name):
    # update obj_action
    if obj_group in ['cart', 'chair']:
        if obj_action == 'open': obj_action = 'pull'
        elif obj_action == 'close': obj_action = 'push'
    elif obj_group in ['ycb']:
        if obj_action == 'open': obj_action = 'pick'
    # load obj_info
    if obj_type == 'partnet': obj_info = PARTNET_ANALYSIS[obj_group][obj_name.replace('-', '/')]
    elif obj_type == 'unidoor': obj_info = UNIDOOR_ANALYSIS[obj_group][obj_name.replace('-', '/')]
    elif obj_type == 'ycb': obj_info = YCB_ANALYSIS[obj_group][obj_name]
    # load obj_prompt
    if len(obj_info.split('/')) == 2: obj_prompt = obj_info.split('/')[1]
    elif len(obj_info.split('/')) > 2: obj_prompt = '{} at {}'.format(obj_info.split('/')[1], obj_info.split('/')[2])
    # assign frame_prompt
    frame_prompt = '{} {}'.format(obj_action.lower(), obj_prompt.replace('_', ' '))
    return frame_prompt
