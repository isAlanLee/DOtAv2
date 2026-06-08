#!/usr/bin/env python3
import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml import yaml_utils
from opencood.utils import box_utils
from opencood.utils import eval_utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate saved label npy files against full GT.")
    parser.add_argument("--hypes-yaml",
                        default="opencood/hypes_yaml/point_pillar_intermediate_fusion_lable_free.yaml")
    parser.add_argument("--pseudo-root",
                        default="/root/autodl-tmp/out_pseudo_lables")
    parser.add_argument("--mbe-root", default="/root/autodl-tmp/out_mbe")
    parser.add_argument("--data-root", default="/root/autodl-tmp/opv2v/train")
    parser.add_argument("--split", choices=["train", "validate"], default="train")
    parser.add_argument("--source", choices=["pseudo", "mbe", "mbe-score"],
                        default="mbe")
    parser.add_argument("--sample", type=int, default=1000,
                        help="Number of frames to sample. Use -1 for all.")
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--score-thresholds", default="0.0",
                        help="Comma-separated thresholds for the file score column.")
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


def load_labels(args, idx):
    if args.source == "pseudo":
        box_path = os.path.join(args.pseudo_root, "pre_box_test_full",
                                f"pre_{idx}.npy")
        score_path = os.path.join(args.pseudo_root, "pre_score_test_full",
                                  f"score_{idx}.npy")
        if not os.path.exists(box_path) or not os.path.exists(score_path):
            return None, None, False
        boxes = np.load(box_path, allow_pickle=True)
        scores = np.load(score_path, allow_pickle=True).reshape(-1)
        return boxes, scores, True

    if args.source == "mbe":
        box_path = os.path.join(args.mbe_root,
                                f"out_pseduo_labels_v1_{idx}.npy")
        if not os.path.exists(box_path):
            return None, None, False
        boxes = np.load(box_path, allow_pickle=True)
        scores = np.ones((boxes.shape[0],), dtype=np.float32)
        return boxes, scores, True

    box_path = os.path.join(args.mbe_root, "score",
                            f"out_pseduo_labels_with_score_v4_{idx}.npy")
    if not os.path.exists(box_path):
        return None, None, False
    arr = np.load(box_path, allow_pickle=True)
    if arr.ndim != 2 or arr.shape[1] < 8:
        return arr, np.empty((0,), dtype=np.float32), True
    return arr[:, :7], arr[:, 7].astype(np.float32), True


def main():
    args = parse_args()
    thresholds = [float(x) for x in args.score_thresholds.split(",") if x]

    hypes = yaml_utils.load_yaml(args.hypes_yaml)
    hypes["lable_free"] = False
    hypes["iterative_training"] = False
    if args.split == "train":
        hypes["validate_dir"] = args.data_root
    else:
        hypes["validate_dir"] = args.data_root.replace("/train", "/validate")

    dataset = build_dataset(hypes, visualize=False, train=False)
    total = len(dataset)
    if args.sample < 0 or args.sample >= total:
        indices = list(range(total))
    else:
        rng = np.random.default_rng(args.seed)
        indices = sorted(rng.choice(total, size=args.sample, replace=False).tolist())

    stats = {thr: make_result_stat() for thr in thresholds}
    counts = {thr: [] for thr in thresholds}
    gt_counts = []
    missing = 0
    bad = 0

    print(f"source: {args.source}")
    print(f"split: {args.split}")
    print(f"dataset_len: {total}")
    print(f"sampled_frames: {len(indices)}")
    print(f"thresholds: {thresholds}")

    for idx in tqdm(indices):
        boxes, scores, exists = load_labels(args, idx)
        if not exists:
            missing += 1
            continue

        data_dict = dataset.collate_batch_test([dataset[idx]])
        gt_box_tensor = dataset.post_processor.generate_gt_bbx(data_dict)
        gt_counts.append(int(gt_box_tensor.shape[0]))

        if (boxes.ndim != 2 or boxes.shape[1] < 7 or
                boxes.shape[0] != scores.shape[0] or
                not np.isfinite(boxes[:, :7]).all() or
                not np.isfinite(scores).all()):
            bad += 1
            continue

        for thr in thresholds:
            keep = scores >= thr
            counts[thr].append(int(np.sum(keep)))
            if np.sum(keep) == 0:
                det_box_tensor = None
                det_score_tensor = None
            else:
                det_corners = box_utils.boxes_to_corners_3d(
                    boxes[keep, :7], order="lwh")
                det_box_tensor = torch.from_numpy(det_corners).float()
                det_score_tensor = torch.from_numpy(scores[keep]).float()

            for iou in (0.30, 0.50, 0.70):
                eval_utils.caluclate_tp_fp(det_box_tensor,
                                           det_score_tensor,
                                           gt_box_tensor,
                                           stats[thr],
                                           iou)

    print(f"missing_files: {missing}")
    print(f"bad_files: {bad}")
    print(f"gt_count_mean: {float(np.mean(gt_counts)) if gt_counts else 0:.4f}")
    print(f"gt_count_median: {float(np.median(gt_counts)) if gt_counts else 0:.4f}")

    for thr in thresholds:
        print()
        print(f"[score >= {thr}]")
        frame_counts = np.asarray(counts[thr], dtype=np.float64)
        if frame_counts.size:
            print(f"pred_count_mean: {float(frame_counts.mean()):.4f}")
            print(f"pred_count_median: {float(np.median(frame_counts)):.4f}")
            print(f"pred_empty_frames: {int(np.sum(frame_counts == 0))}")
        for iou in (0.30, 0.50, 0.70):
            ap, _, _ = eval_utils.calculate_ap(stats[thr], iou, True)
            print(f"iou_{iou:.2f}: recall={final_recall(stats[thr], iou):.4f}, ap={ap:.4f}")


if __name__ == "__main__":
    main()
