#!/usr/bin/env python3
import argparse
import bisect
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from diagnose_mbe_thresholds import (  # noqa: E402
    box_metrics,
    compute_c1_c2,
    get_registration_angle,
    pc_2_world,
    scenario_ranges,
    sorted_npy,
    x_to_world,
)
from opencood.data_utils.datasets import build_dataset  # noqa: E402
from opencood.hypes_yaml import yaml_utils  # noqa: E402
from opencood.utils import box_utils, eval_utils  # noqa: E402


def parse_float_list(text):
    return [float(x) for x in text.split(",") if x]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate MBE threshold recall/AP without rewriting MBE npy files.")
    parser.add_argument("--hypes-yaml",
                        default="opencood/hypes_yaml/point_pillar_intermediate_fusion_lable_free.yaml")
    parser.add_argument("--pseudo-root",
                        default="/root/autodl-tmp/out_pseudo_lables")
    parser.add_argument("--mbe-root", default="/root/autodl-tmp/out_mbe")
    parser.add_argument("--data-root", default="/root/autodl-tmp/opv2v/train")
    parser.add_argument("--split", choices=["train", "validate"], default="train")
    parser.add_argument("--sample-frames", type=int, default=250)
    parser.add_argument("--max-boxes-per-frame", type=int, default=-1,
                        help="Use -1 to evaluate all pseudo boxes in sampled frames.")
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--mode", choices=["inverse", "linear", "uniform"],
                        default="inverse")
    parser.add_argument("--phi-r-values", default="0.1")
    parser.add_argument("--phi-o-values", default="0.4,0.5,0.6,0.7")
    return parser.parse_args()


def make_result_stat():
    return {
        0.30: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.50: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.70: {"tp": [], "fp": [], "gt": 0, "score": []},
    }


def final_recall(result_stat, iou):
    stat = result_stat[iou]
    if stat["gt"] == 0 or len(stat["tp"]) == 0:
        return 0.0
    return float(np.sum(stat["tp"])) / float(stat["gt"])


def add_eval(boxes, scores, gt_box_tensor, result_stat):
    if boxes.shape[0] == 0:
        det_box_tensor = None
        det_score_tensor = None
    else:
        det_corners = box_utils.boxes_to_corners_3d(boxes[:, :7], order="lwh")
        det_box_tensor = torch.from_numpy(det_corners).float()
        det_score_tensor = torch.from_numpy(scores).float()

    for iou in (0.30, 0.50, 0.70):
        eval_utils.caluclate_tp_fp(det_box_tensor,
                                   det_score_tensor,
                                   gt_box_tensor,
                                   result_stat,
                                   iou)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    phi_r_values = parse_float_list(args.phi_r_values)
    phi_o_values = parse_float_list(args.phi_o_values)
    threshold_keys = [(r, o) for r in phi_r_values for o in phi_o_values]

    pre_box = sorted_npy(os.path.join(
        args.pseudo_root, "pre_box_test_full", "pre_*.npy"))
    pre_score = sorted_npy(os.path.join(
        args.pseudo_root, "pre_score_test_full", "score_*.npy"))
    scenarios, _, ends = scenario_ranges(args.data_root)
    all_indices = sorted(set(pre_box) & set(pre_score))
    if args.sample_frames > 0 and len(all_indices) > args.sample_frames:
        frame_indices = sorted(rng.choice(
            all_indices, size=args.sample_frames, replace=False).tolist())
    else:
        frame_indices = all_indices

    hypes = yaml_utils.load_yaml(args.hypes_yaml)
    hypes["lable_free"] = False
    hypes["iterative_training"] = False
    if args.split == "train":
        hypes["validate_dir"] = args.data_root
    else:
        hypes["validate_dir"] = args.data_root.replace("/train", "/validate")
    dataset = build_dataset(hypes, visualize=False, train=False)

    stats = {key: make_result_stat() for key in threshold_keys}
    counts = {key: [] for key in threshold_keys}
    empty_counts = {key: 0 for key in threshold_keys}
    gt_counts = []
    boxes_seen = 0
    cache = {}

    print(f"mode: {args.mode}")
    print(f"sampled_frames: {len(frame_indices)}")
    print(f"max_boxes_per_frame: {args.max_boxes_per_frame}")
    print(f"phi_r_values: {phi_r_values}")
    print(f"phi_o_values: {phi_o_values}")

    for global_idx in tqdm(frame_indices):
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
        if boxes.ndim != 2 or boxes.shape[1] < 7 or boxes.shape[0] != scores.shape[0]:
            continue
        if args.max_boxes_per_frame > 0 and boxes.shape[0] > args.max_boxes_per_frame:
            chosen = sorted(rng.choice(boxes.shape[0],
                                       size=args.max_boxes_per_frame,
                                       replace=False).tolist())
        else:
            chosen = list(range(boxes.shape[0]))

        data_dict = dataset.collate_batch_test([dataset[global_idx]])
        gt_box_tensor = dataset.post_processor.generate_gt_bbx(data_dict)
        gt_counts.append(int(gt_box_tensor.shape[0]))

        boxes_world = boxes.copy()
        center_world = pc_2_world(boxes_world[:, :3].copy(), poses[0][local_idx])
        dif_ang = get_registration_angle(x_to_world(poses[0][local_idx]))
        boxes_world[:, :3] = center_world[:, :3]
        boxes_world[:, 6] = boxes_world[:, 6] + dif_ang

        c_pairs = []
        for box_idx in chosen:
            boxes_seen += 1
            inter_counts, hull_counts, distances = box_metrics(
                boxes_world[box_idx], multi_agent_point, poses, local_idx)
            c_pairs.append(compute_c1_c2(inter_counts, hull_counts,
                                         distances, args.mode))

        c_pairs = np.asarray(c_pairs, dtype=np.float64)
        chosen_boxes = boxes[chosen, :7]
        chosen_scores = scores[chosen].astype(np.float32)
        for key in threshold_keys:
            phi_r, phi_o = key
            if c_pairs.size == 0:
                keep = np.zeros((0,), dtype=bool)
            else:
                keep = np.logical_and(c_pairs[:, 0] < phi_r, c_pairs[:, 1] > phi_o)
            counts[key].append(int(np.sum(keep)))
            if not np.any(keep):
                empty_counts[key] += 1
            add_eval(chosen_boxes[keep], chosen_scores[keep],
                     gt_box_tensor, stats[key])

    print(f"boxes_evaluated: {boxes_seen}")
    print(f"gt_count_mean: {float(np.mean(gt_counts)) if gt_counts else 0:.4f}")
    print(f"gt_count_median: {float(np.median(gt_counts)) if gt_counts else 0:.4f}")

    for key in threshold_keys:
        phi_r, phi_o = key
        frame_counts = np.asarray(counts[key], dtype=np.float64)
        print()
        print(f"[phi_r={phi_r}, phi_o={phi_o}]")
        if frame_counts.size:
            print(f"pred_count_mean: {float(frame_counts.mean()):.4f}")
            print(f"pred_count_median: {float(np.median(frame_counts)):.4f}")
            print(f"pred_empty_frames: {empty_counts[key]}")
        for iou in (0.30, 0.50, 0.70):
            ap, _, _ = eval_utils.calculate_ap(stats[key], iou, True)
            print(f"iou_{iou:.2f}: recall={final_recall(stats[key], iou):.4f}, ap={ap:.4f}")


if __name__ == "__main__":
    main()
