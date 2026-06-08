#!/usr/bin/env python3
import argparse
import bisect
import glob
import os
import re

import numpy as np
import scipy
from scipy.spatial import ConvexHull, Delaunay
try:
    from scipy.spatial import QhullError
except ImportError:
    from scipy.spatial.qhull import QhullError


SCALE_VAR = [1.5, 1.2, 1.0, 0.8, 0.5]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose why MBE rejects pseudo labels.")
    parser.add_argument("--pseudo-root",
                        default="/root/autodl-tmp/out_pseudo_lables")
    parser.add_argument("--mbe-root", default="/root/autodl-tmp/out_mbe")
    parser.add_argument("--data-root", default="/root/autodl-tmp/opv2v/train")
    parser.add_argument("--sample-frames", type=int, default=120)
    parser.add_argument("--max-boxes-per-frame", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--phi-r", type=float, default=0.1)
    parser.add_argument("--phi-o", type=float, default=0.7)
    return parser.parse_args()


def index_from_name(path):
    match = re.search(r"_(\d+)\.npy$", os.path.basename(path))
    return int(match.group(1)) if match else None


def sorted_npy(pattern):
    paths = glob.glob(pattern)
    pairs = []
    for path in paths:
        idx = index_from_name(path)
        if idx is not None:
            pairs.append((idx, path))
    return dict(sorted(pairs))


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
    points_rot = np.concatenate((points_rot, points[:, :, 3:]), axis=-1)
    return points_rot


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


def in_hull(p, hull):
    if p.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    try:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        return hull.find_simplex(p) >= 0
    except (QhullError, ValueError):
        return np.zeros(p.shape[0], dtype=bool)


def x_to_world(pose):
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


def pc_2_world(points, pose):
    if points.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float32)
    point_homogeneous = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))
    return np.dot(x_to_world(pose), point_homogeneous.T).T


def get_registration_angle(mat):
    cos_theta, sin_theta = mat[0, 0], mat[1, 0]
    cos_theta = np.clip(cos_theta, -1, 1)
    theta_cos = np.arccos(cos_theta)
    return theta_cos if sin_theta >= 0 else 2 * np.pi - theta_cos


def scenario_ranges(data_root):
    scenarios = sorted([
        os.path.join(data_root, x)
        for x in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, x))
    ])
    ends = []
    lengths = []
    total = 0
    for scenario in scenarios:
        cavs = sorted([
            x for x in os.listdir(scenario)
            if os.path.isdir(os.path.join(scenario, x))
        ])
        if not cavs:
            lengths.append(0)
            ends.append(total)
            continue
        ego = os.path.join(scenario, cavs[0])
        yamls = sorted([
            x for x in os.listdir(ego)
            if x.endswith(".yaml") and "additional" not in x
        ])
        total += len(yamls)
        lengths.append(len(yamls))
        ends.append(total)
    return scenarios, lengths, ends


def safe_ratio(numerator, denominator):
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def distance_weights(distances, mode):
    distances = np.asarray(distances, dtype=np.float64)
    if mode == "inverse":
        weights = 1.0 / np.maximum(distances ** 2, 1e-6)
    elif mode == "linear":
        weights = distances.copy()
    elif mode == "uniform":
        weights = np.ones_like(distances)
    else:
        raise ValueError(mode)
    weight_sum = np.sum(weights)
    if weight_sum <= 0:
        return np.ones_like(distances) / max(len(distances), 1)
    return weights / weight_sum


def compute_c1_c2(inter_counts, hull_counts, distances, mode):
    weights = distance_weights(distances, mode)
    c1 = 0.0
    c2 = 0.0
    for i in range(len(inter_counts)):
        score_r_1 = safe_ratio(inter_counts[i][0] - inter_counts[i][1],
                               inter_counts[i][1])
        score_r_2 = safe_ratio(inter_counts[i][1] - inter_counts[i][2],
                               inter_counts[i][2])
        score_r = (score_r_1 + score_r_2) / 2.0
        score_o_1 = safe_ratio(hull_counts[i][2] - hull_counts[i][3],
                               hull_counts[i][2])
        score_o_2 = safe_ratio(hull_counts[i][3] - hull_counts[i][4],
                               hull_counts[i][3])
        score_o = (score_o_1 + score_o_2) / 2.0
        c1 += score_r * weights[i]
        c2 += score_o * weights[i]
    return c1, c2


def box_metrics(box, multi_agent_point, poses, local_idx):
    distances = []
    inter_counts = []
    hull_counts = []
    for cav_idx in range(len(poses)):
        pose_xyz = np.asarray(poses[cav_idx][local_idx])[:3]
        distances.append(np.linalg.norm(box[:2] - pose_xyz[:2]))

        points_this = multi_agent_point[cav_idx][local_idx]
        inter_scale_counts = []
        hull_scale_counts = []
        for scale in SCALE_VAR:
            scale_box = np.ones(7)
            scale_box[:3] = box[:3]
            scale_box[3:6] = box[3:6] * scale
            scale_box[6] = box[6]
            if points_this.shape[0] == 0:
                inter_scale_counts.append(0)
                hull_scale_counts.append(0)
                continue
            mask = in_hull(points_this[:, :3],
                           boxes_to_corners_3d(scale_box.reshape(-1, 7)).reshape(-1, 3))
            inter_points = points_this[:, :3][mask]
            inter_scale_counts.append(inter_points.shape[0])
            if inter_points.shape[0] < 4:
                hull_scale_counts.append(0)
                continue
            try:
                hull_scale_counts.append(ConvexHull(inter_points).vertices.shape[0])
            except (QhullError, ValueError):
                hull_scale_counts.append(0)
        inter_counts.append(inter_scale_counts)
        hull_counts.append(hull_scale_counts)
    return inter_counts, hull_counts, distances


def pct(name, values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        print(f"{name}: empty")
        return
    result = {
        "0": float(np.percentile(values, 0)),
        "5": float(np.percentile(values, 5)),
        "50": float(np.percentile(values, 50)),
        "95": float(np.percentile(values, 95)),
        "100": float(np.percentile(values, 100)),
    }
    print(f"{name}: {result}")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    pre_box = sorted_npy(os.path.join(
        args.pseudo_root, "pre_box_test_full", "pre_*.npy"))
    pre_score = sorted_npy(os.path.join(
        args.pseudo_root, "pre_score_test_full", "score_*.npy"))
    mbe_pos = sorted_npy(os.path.join(
        args.mbe_root, "out_pseduo_labels_v1_*.npy"))
    scenarios, lengths, ends = scenario_ranges(args.data_root)

    all_indices = sorted(set(pre_box) & set(pre_score) & set(mbe_pos))
    if len(all_indices) > args.sample_frames:
        frame_indices = sorted(rng.choice(
            all_indices, size=args.sample_frames, replace=False).tolist())
    else:
        frame_indices = all_indices

    print(f"sampled_frames: {len(frame_indices)}")
    print(f"max_boxes_per_frame: {args.max_boxes_per_frame}")
    print(f"phi_r: {args.phi_r}")
    print(f"phi_o: {args.phi_o}")

    mode_stats = {
        "inverse": {"accepted": 0, "c1": [], "c2": [], "pass_c1": 0, "pass_c2": 0},
        "linear": {"accepted": 0, "c1": [], "c2": [], "pass_c1": 0, "pass_c2": 0},
        "uniform": {"accepted": 0, "c1": [], "c2": [], "pass_c1": 0, "pass_c2": 0},
    }
    score_seen = []
    score_by_inverse_accept = {"accepted": [], "rejected": []}
    boxes_seen = 0
    saved_mbe_counts = []

    cache = {}
    for global_idx in frame_indices:
        scenario_idx = bisect.bisect_right(ends, global_idx)
        scenario_start = 0 if scenario_idx == 0 else ends[scenario_idx - 1]
        local_idx = global_idx - scenario_start
        if scenario_idx not in cache:
            point_path = os.path.join(
                args.mbe_root, "multi_agent_point_remove_ground",
                f"multi_agent_point{scenario_idx}.npy")
            pose_path = os.path.join(
                args.mbe_root, "multi_agent_point_pose",
                f"multi_agent_point_pose{scenario_idx}.npy")
            cache[scenario_idx] = (
                np.load(point_path, allow_pickle=True),
                np.load(pose_path, allow_pickle=True),
            )
        multi_agent_point, poses = cache[scenario_idx]

        boxes = np.load(pre_box[global_idx], allow_pickle=True)
        scores = np.load(pre_score[global_idx], allow_pickle=True).reshape(-1)
        saved_mbe_counts.append(np.load(mbe_pos[global_idx],
                                        allow_pickle=True).shape[0])
        if boxes.shape[0] > args.max_boxes_per_frame:
            chosen = sorted(rng.choice(boxes.shape[0],
                                       size=args.max_boxes_per_frame,
                                       replace=False).tolist())
        else:
            chosen = list(range(boxes.shape[0]))

        boxes_world = boxes.copy()
        center_world = pc_2_world(boxes_world[:, :3].copy(),
                                  poses[0][local_idx])
        dif_ang = get_registration_angle(x_to_world(poses[0][local_idx]))
        boxes_world[:, :3] = center_world[:, :3]
        boxes_world[:, 6] = boxes_world[:, 6] + dif_ang

        for box_idx in chosen:
            boxes_seen += 1
            score_seen.append(float(scores[box_idx]))
            inter_counts, hull_counts, distances = box_metrics(
                boxes_world[box_idx], multi_agent_point, poses, local_idx)
            inverse_accepted = False
            for mode in mode_stats:
                c1, c2 = compute_c1_c2(inter_counts, hull_counts,
                                       distances, mode)
                mode_stats[mode]["c1"].append(c1)
                mode_stats[mode]["c2"].append(c2)
                if c1 < args.phi_r:
                    mode_stats[mode]["pass_c1"] += 1
                if c2 > args.phi_o:
                    mode_stats[mode]["pass_c2"] += 1
                if c1 < args.phi_r and c2 > args.phi_o:
                    mode_stats[mode]["accepted"] += 1
                    if mode == "inverse":
                        inverse_accepted = True
            if inverse_accepted:
                score_by_inverse_accept["accepted"].append(float(scores[box_idx]))
            else:
                score_by_inverse_accept["rejected"].append(float(scores[box_idx]))

    print(f"boxes_evaluated: {boxes_seen}")
    print(f"saved_mbe_accepted_in_sampled_frames: {int(np.sum(saved_mbe_counts))}")
    pct("pre_score_seen", score_seen)

    for mode, stats in mode_stats.items():
        accepted = stats["accepted"]
        print()
        print(f"[{mode}]")
        print(f"accepted_by_recomputed_mbe: {accepted} / {boxes_seen} "
              f"({accepted / max(boxes_seen, 1):.6f})")
        print(f"pass_c1_only: {stats['pass_c1']} / {boxes_seen} "
              f"({stats['pass_c1'] / max(boxes_seen, 1):.6f})")
        print(f"pass_c2_only: {stats['pass_c2']} / {boxes_seen} "
              f"({stats['pass_c2'] / max(boxes_seen, 1):.6f})")
        pct("c1_percentiles", stats["c1"])
        pct("c2_percentiles", stats["c2"])

    print()
    pct("pre_score_inverse_accepted", score_by_inverse_accept["accepted"])
    pct("pre_score_inverse_rejected", score_by_inverse_accept["rejected"])


if __name__ == "__main__":
    main()
