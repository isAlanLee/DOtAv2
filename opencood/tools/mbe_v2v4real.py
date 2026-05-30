import argparse
import math
import os

import numpy as np
import open3d as o3d
import scipy
from scipy.spatial import ConvexHull, Delaunay
from tqdm import tqdm

from opencood.hypes_yaml.yaml_utils import load_yaml


def pcd_to_np(pcd_file):
    pcd = o3d.io.read_point_cloud(pcd_file)
    xyz = np.asarray(pcd.points)
    intensity = np.expand_dims(np.asarray(pcd.colors)[:, 0], -1)
    return np.asarray(np.hstack((xyz, intensity)), dtype=np.float32)


def pose_to_matrix(pose):
    pose = np.asarray(pose)
    if pose.shape == (4, 4):
        return pose

    x, y, z, roll, yaw, pitch = pose[:]
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))

    matrix = np.identity(4)
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix


def get_registration_angle(mat):
    cos_theta, sin_theta = mat[0, 0], mat[1, 0]
    cos_theta = np.clip(cos_theta, -1, 1)
    theta_cos = np.arccos(cos_theta)
    return theta_cos if sin_theta >= 0 else 2 * np.pi - theta_cos


def points_to_world(points, pose):
    pose_mat = pose_to_matrix(pose)
    points_h = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))
    points_w = np.dot(pose_mat, points_h.T).T
    if points.shape[1] > 3:
        return np.hstack((points_w[:, :3], points[:, 3:]))
    return points_w[:, :3]


def boxes_to_world(boxes, ego_pose):
    if boxes.shape[0] == 0:
        return boxes.copy()

    pose_mat = pose_to_matrix(ego_pose)
    boxes_w = boxes.copy()
    centers_h = np.hstack((boxes[:, :3], np.ones((boxes.shape[0], 1))))
    centers_w = np.dot(pose_mat, centers_h.T).T
    boxes_w[:, :3] = centers_w[:, :3]
    boxes_w[:, 6] = boxes[:, 6] + get_registration_angle(pose_mat)
    return boxes_w


def boxes_to_corners_3d(boxes3d):
    template = np.array((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6] * template[None, :, :]
    corners3d = rotate_points_along_z(
        corners3d.reshape(-1, 8, 3), boxes3d[:, 6]).reshape(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]
    return corners3d


def rotate_points_along_z(points, angle):
    cosa = np.cos(angle)
    sina = np.sin(angle)
    zeros = np.zeros_like(angle)
    ones = np.ones_like(angle)
    rot_matrix = np.stack((
        cosa, sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), axis=1).reshape(-1, 3, 3).astype(float)
    points_rot = np.matmul(points[:, :, 0:3], rot_matrix)
    return np.concatenate((points_rot, points[:, :, 3:]), axis=-1)


def in_hull(points, hull):
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    try:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        return hull.find_simplex(points) >= 0
    except scipy.spatial.qhull.QhullError:
        return np.zeros(points.shape[0], dtype=bool)


def safe_ratio(a, b):
    if abs(b) < 1e-6:
        return 0.0
    return a / b


def classify_state(inter_points_number_total, convex_hull_number_total,
                   distance_total):
    if not inter_points_number_total or sum(distance_total) < 1e-6:
        return 0

    c1 = 0.0
    c2 = 0.0
    for i in range(len(inter_points_number_total)):
        inter = inter_points_number_total[i]
        hull = convex_hull_number_total[i]
        score_r_1 = safe_ratio(inter[0] - inter[1], inter[1])
        score_r_2 = safe_ratio(inter[1] - inter[2], inter[2])
        score_r = (score_r_1 + score_r_2) / 2
        score_0_1 = safe_ratio(hull[2] - hull[3], hull[2])
        score_0_2 = safe_ratio(hull[3] - hull[4], hull[3])
        score_0 = (score_0_1 + score_0_2) / 2
        score_d = distance_total[i] / sum(distance_total)
        c1 += score_r * score_d
        c2 += score_0 * score_d

    return 1 if c1 < 0.1 and c2 > 0.7 else 0


def box_filter(pseudo_labels_world, multi_agent_point, cav_positions,
               frame_idx):
    if pseudo_labels_world.shape[0] == 0:
        return np.zeros((0,), dtype=bool)

    keep = []
    scale_var = [1.5, 1.2, 1.0, 0.8, 0.5]
    for box in pseudo_labels_world:
        distance_total = [
            np.linalg.norm(box[:2] - cav_position[:2])
            for cav_position in cav_positions
        ]
        inter_points_number_total = []
        convex_hull_number_total = []
        for cav_idx in range(len(cav_positions)):
            inter_counts = []
            hull_counts = []
            points = multi_agent_point[cav_idx][frame_idx][:, :3]
            for scale in scale_var:
                scale_box = np.ones(7)
                scale_box[:3] = box[:3]
                scale_box[3:6] = box[3:6] * scale
                scale_box[6] = box[6]
                mask = in_hull(
                    points,
                    boxes_to_corners_3d(scale_box.reshape(-1, 7)).reshape(-1, 3))
                inter_points = points[mask]
                inter_counts.append(inter_points.shape[0])
                if inter_points.shape[0] >= 4:
                    try:
                        hull_counts.append(
                            inter_points[ConvexHull(inter_points).vertices].shape[0])
                    except scipy.spatial.qhull.QhullError:
                        hull_counts.append(0)
                else:
                    hull_counts.append(0)

            inter_points_number_total.append(inter_counts)
            convex_hull_number_total.append(hull_counts)

        keep.append(
            classify_state(inter_points_number_total,
                           convex_hull_number_total,
                           distance_total) == 1)
    return np.array(keep, dtype=bool)


def get_transform(box):
    x, y, z, l, w, h, yaw = box[:7]
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    trans = np.eye(4, dtype=np.float32)
    trans[0, 0] = cos_yaw
    trans[0, 1] = -sin_yaw
    trans[0, 3] = x
    trans[1, 0] = sin_yaw
    trans[1, 1] = cos_yaw
    trans[1, 3] = y
    trans[2, 3] = z
    return np.linalg.inv(trans)


def score_box(points, box):
    if points.shape[0] == 0:
        return 0.0

    mask = in_hull(points[:, :3],
                   boxes_to_corners_3d(box[:7].reshape(-1, 7)).reshape(-1, 3))
    foreground = points[:, :3][mask]
    if foreground.shape[0] == 0:
        return 0.0

    xy = foreground[:, :2]
    if xy.shape[0] >= 3:
        try:
            xy = xy[ConvexHull(xy).vertices]
        except scipy.spatial.qhull.QhullError:
            pass

    points_h = np.hstack((xy, np.zeros((xy.shape[0], 1)),
                          np.ones((xy.shape[0], 1))))
    local = np.dot(points_h, get_transform(box).T)[:, :3]
    scale = np.abs(local[:, :2]) / (box[3:5] * 0.5)
    return float(np.max(scale, axis=1).mean())


def append_scores(local_boxes, world_boxes, dense_points):
    if local_boxes.shape[0] == 0:
        return np.empty((0, 8), dtype=np.float32)

    scores = [score_box(dense_points, box) for box in world_boxes]
    return np.hstack((local_boxes, np.array(scores).reshape(-1, 1)))


def scenario_timestamps(scenario_folder, cav_list):
    first_cav_path = os.path.join(scenario_folder, cav_list[0])
    yaml_files = sorted([
        x for x in os.listdir(first_cav_path)
        if x.endswith('.yaml') and 'additional' not in x
    ])
    return [x.replace('.yaml', '') for x in yaml_files]


def load_scenario_points_and_poses(scenario_folder, cav_list, timestamps):
    multi_agent_point = []
    poses = []
    for cav_id in cav_list:
        cav_path = os.path.join(scenario_folder, cav_id)
        cav_points = []
        cav_poses = []
        for timestamp in timestamps:
            pcd_path = os.path.join(cav_path, timestamp + '.pcd')
            yaml_path = os.path.join(cav_path, timestamp + '.yaml')
            pose = load_yaml(yaml_path)['lidar_pose']
            points_world = points_to_world(pcd_to_np(pcd_path), pose)
            cav_points.append(points_world)
            cav_poses.append(pose)
        multi_agent_point.append(cav_points)
        poses.append(cav_poses)
    return multi_agent_point, poses


def main():
    parser = argparse.ArgumentParser(
        description='Run MBE filtering and score generation for V2V4Real.')
    parser.add_argument('--data_dir', required=True,
                        help='V2V4Real OPV2V-format train directory.')
    parser.add_argument('--pred_dir',
                        default='pseduo_label_val/pre_box_test_full',
                        help='Directory containing pre_{idx}.npy.')
    parser.add_argument('--out_dir', default='/root/autodl-tmp/out_v2v4real',
                        help='Directory for filtered pseudo labels.')
    parser.add_argument('--window', type=int, default=25,
                        help='Temporal window radius for scoring.')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    scenario_folders = sorted([
        os.path.join(args.data_dir, x)
        for x in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, x))
    ])

    global_idx = 0
    for scenario_folder in tqdm(scenario_folders, desc='scenarios'):
        cav_list = sorted([
            x for x in os.listdir(scenario_folder)
            if os.path.isdir(os.path.join(scenario_folder, x))
        ])
        timestamps = scenario_timestamps(scenario_folder, cav_list)
        multi_agent_point, poses = load_scenario_points_and_poses(
            scenario_folder, cav_list, timestamps)

        num_frames = len(timestamps)
        for frame_idx in tqdm(range(num_frames), desc=os.path.basename(scenario_folder),
                              leave=False):
            pred_path = os.path.join(args.pred_dir, 'pre_%d.npy' % global_idx)
            local_boxes = np.load(pred_path)
            if local_boxes.shape[0] == 0:
                keep = np.zeros((0,), dtype=bool)
                world_boxes = local_boxes.copy()
            else:
                world_boxes = boxes_to_world(local_boxes, poses[0][frame_idx])
                cav_positions = [
                    pose_to_matrix(poses[cav_idx][frame_idx])[:3, 3]
                    for cav_idx in range(len(cav_list))
                ]
                keep = box_filter(world_boxes, multi_agent_point,
                                  cav_positions, frame_idx)

            noise = np.logical_not(keep)
            filtered_local = local_boxes[keep]
            noise_local = local_boxes[noise]
            filtered_world = world_boxes[keep]
            noise_world = world_boxes[noise]

            start = max(0, frame_idx - args.window)
            end = min(num_frames, frame_idx + args.window)
            dense_points = []
            for dense_frame in range(start, end):
                for cav_idx in range(len(cav_list)):
                    dense_points.append(multi_agent_point[cav_idx][dense_frame])
            dense_points = np.concatenate(dense_points, axis=0) \
                if dense_points else np.empty((0, 4))

            filtered_with_score = append_scores(
                filtered_local, filtered_world, dense_points)
            noise_with_score = append_scores(
                noise_local, noise_world, dense_points)

            np.save(os.path.join(
                args.out_dir, 'out_pseduo_labels_v1_%d.npy' % global_idx),
                filtered_local)
            np.save(os.path.join(
                args.out_dir, 'out_pseduo_labels_noise_v1_%d.npy' % global_idx),
                noise_local)
            np.save(os.path.join(
                args.out_dir,
                'out_pseduo_labels_with_score_v4_%d.npy' % global_idx),
                filtered_with_score)
            np.save(os.path.join(
                args.out_dir,
                'out_pseduo_labels_noise_with_score_v4_%d.npy' % global_idx),
                noise_with_score)
            global_idx += 1


if __name__ == '__main__':
    main()
