#!/usr/bin/env python3
import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose DOtA pseudo-label and MBE artifact health.")
    parser.add_argument("--pseudo-root",
                        default="/root/autodl-tmp/out_pseudo_lables")
    parser.add_argument("--mbe-root",
                        default="/root/autodl-tmp/out_mbe")
    parser.add_argument("--data-root",
                        default="/root/autodl-tmp/opv2v/train")
    parser.add_argument("--final-model-dir", default="")
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260608)
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


def load_npy(path):
    return np.load(path, allow_pickle=True)


def numeric(arr):
    arr = np.asarray(arr)
    if arr.dtype == object:
        arr = arr.astype(np.float64)
    return arr


def pct(values, q=(0, 1, 5, 50, 95, 99, 100)):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return "empty"
    return {str(x): float(np.percentile(values, x)) for x in q}


def print_header(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def summarize_counts(name, counts):
    counts = np.asarray(counts, dtype=np.float64)
    if counts.size == 0:
        print(f"{name}: empty")
        return
    print(f"{name}: total={int(counts.sum())}, files={counts.size}, "
          f"empty_files={int((counts == 0).sum())}, "
          f"min={int(counts.min())}, median={float(np.median(counts)):.2f}, "
          f"mean={float(counts.mean()):.2f}, max={int(counts.max())}")


def summarize_box_dims(name, boxes):
    if not boxes:
        print(f"{name}: no boxes")
        return
    boxes = numeric(np.concatenate(boxes, axis=0))
    if boxes.size == 0 or boxes.shape[1] < 6:
        print(f"{name}: invalid boxes shape {boxes.shape}")
        return
    dims = boxes[:, 3:6]
    med = np.median(dims, axis=0)
    mean = np.mean(dims, axis=0)
    if med[0] > med[2]:
        guess = "lwh-like before dataset swap"
    elif med[2] > med[0]:
        guess = "hwl-like before dataset swap"
    else:
        guess = "ambiguous"
    print(f"{name}: dim_mean[col3,col4,col5]={mean.tolist()}")
    print(f"{name}: dim_median[col3,col4,col5]={med.tolist()} -> {guess}")
    bad = np.sum(np.any(dims <= 0, axis=1))
    print(f"{name}: non_positive_dims={int(bad)} / {dims.shape[0]}")


def sample_indices(indices, n, seed):
    indices = list(indices)
    if len(indices) <= n:
        return indices
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(indices, size=n, replace=False).tolist())


def inspect_yaml(path):
    try:
        import yaml
    except ImportError:
        print("PyYAML not installed; skip yaml inspection")
        return
    if not path or not os.path.exists(path):
        print(f"missing yaml: {path}")
        return
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    print(f"yaml: {path}")
    for key in ("root_dir", "validate_dir", "lable_free",
                "iterative_training", "pseudo_lable_path"):
        print(f"  {key}: {data.get(key)}")
    post = data.get("postprocess", {})
    print(f"  postprocess.order: {post.get('order')}")
    target = post.get("target_args", {})
    print(f"  score_threshold: {target.get('score_threshold')}")
    print(f"  pos_threshold: {target.get('pos_threshold')}")
    print(f"  neg_threshold: {target.get('neg_threshold')}")


def inspect_dataset(data_root):
    print_header("Dataset Index")
    scenarios = sorted([
        os.path.join(data_root, x)
        for x in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, x))
    ]) if os.path.isdir(data_root) else []
    frame_total = 0
    scenario_lengths = []
    missing_pcd = 0
    for scenario in scenarios:
        cavs = sorted([
            x for x in os.listdir(scenario)
            if os.path.isdir(os.path.join(scenario, x))
        ])
        if not cavs:
            continue
        ego = os.path.join(scenario, cavs[0])
        yamls = sorted([
            x for x in os.listdir(ego)
            if x.endswith(".yaml") and "additional" not in x
        ])
        timestamps = [x[:-5] for x in yamls]
        frame_total += len(timestamps)
        scenario_lengths.append(len(timestamps))
        for cav in cavs:
            cav_path = os.path.join(scenario, cav)
            for ts in timestamps:
                if not os.path.exists(os.path.join(cav_path, ts + ".pcd")):
                    missing_pcd += 1
    print(f"data_root: {data_root}")
    print(f"scenario_count: {len(scenarios)}")
    print(f"frame_total_from_first_cav_yaml: {frame_total}")
    summarize_counts("scenario_frame_counts", scenario_lengths)
    print(f"missing_pcd_against_first_cav_timestamps: {missing_pcd}")


def main():
    args = parse_args()
    np.random.seed(args.seed)

    pre_box = sorted_npy(os.path.join(
        args.pseudo_root, "pre_box_test_full", "pre_*.npy"))
    pre_score = sorted_npy(os.path.join(
        args.pseudo_root, "pre_score_test_full", "score_*.npy"))
    mbe_pos = sorted_npy(os.path.join(
        args.mbe_root, "out_pseduo_labels_v1_*.npy"))
    mbe_neg = sorted_npy(os.path.join(
        args.mbe_root, "out_pseduo_labels_noise_v1_*.npy"))
    score_pos = sorted_npy(os.path.join(
        args.mbe_root, "score", "out_pseduo_labels_with_score_v4_*.npy"))
    score_neg = sorted_npy(os.path.join(
        args.mbe_root, "score",
        "out_pseduo_labels_noise_with_score_v4_*.npy"))

    print_header("Artifact Counts")
    print(f"pre_box_files: {len(pre_box)}")
    print(f"pre_score_files: {len(pre_score)}")
    print(f"mbe_accepted_files: {len(mbe_pos)}")
    print(f"mbe_noise_files: {len(mbe_neg)}")
    print(f"scored_accepted_files: {len(score_pos)}")
    print(f"scored_noise_files: {len(score_neg)}")
    print(f"missing_pre_score_for_pre_box: "
          f"{len(set(pre_box) - set(pre_score))}")
    print(f"missing_mbe_for_pre_box: {len(set(pre_box) - set(mbe_pos))}")
    print(f"missing_scored_for_mbe: {len(set(mbe_pos) - set(score_pos))}")

    inspect_dataset(args.data_root)

    print_header("Pseudo Label Statistics")
    common = sample_indices(set(pre_box) & set(pre_score), args.sample,
                            args.seed)
    pre_counts = []
    pre_boxes_for_dims = []
    pre_scores = []
    bad = defaultdict(int)
    for idx in common:
        boxes = load_npy(pre_box[idx])
        scores = load_npy(pre_score[idx])
        pre_counts.append(boxes.shape[0])
        if boxes.shape[0] > 0:
            pre_boxes_for_dims.append(boxes)
        pre_scores.append(np.asarray(scores).reshape(-1))
        if boxes.ndim != 2 or boxes.shape[1] != 7:
            bad["pre_box_shape"] += 1
        if scores.shape[0] != boxes.shape[0]:
            bad["pre_score_count_mismatch"] += 1
        if not np.isfinite(numeric(boxes)).all():
            bad["pre_box_nonfinite"] += 1
        if not np.isfinite(numeric(scores)).all():
            bad["pre_score_nonfinite"] += 1
    print(f"sampled_pre_indices: {len(common)}")
    summarize_counts("pre_box_counts", pre_counts)
    print(f"pre_score_percentiles: {pct(np.concatenate(pre_scores)) if pre_scores else 'empty'}")
    summarize_box_dims("pre_box_dims", pre_boxes_for_dims)
    print(f"pre_bad: {dict(bad)}")

    print_header("MBE Filter Statistics")
    common_mbe = sample_indices(set(mbe_pos) & set(mbe_neg), args.sample,
                                args.seed)
    pos_counts, neg_counts, ratios = [], [], []
    pos_boxes_for_dims = []
    for idx in common_mbe:
        pos = load_npy(mbe_pos[idx])
        neg = load_npy(mbe_neg[idx])
        pos_counts.append(pos.shape[0])
        neg_counts.append(neg.shape[0])
        denom = pos.shape[0] + neg.shape[0]
        ratios.append(pos.shape[0] / denom if denom else np.nan)
        if pos.shape[0] > 0:
            pos_boxes_for_dims.append(pos)
    summarize_counts("mbe_accepted_counts", pos_counts)
    summarize_counts("mbe_noise_counts", neg_counts)
    print(f"mbe_accept_ratio_percentiles: {pct(ratios)}")
    summarize_box_dims("mbe_accepted_dims", pos_boxes_for_dims)

    print_header("MBE Score Statistics")
    common_score = sample_indices(set(score_pos) & set(score_neg),
                                  args.sample, args.seed)
    score_pos_counts, score_neg_counts = [], []
    pos_score_cols, neg_score_cols = [], []
    score_boxes_for_dims = []
    score_bad = defaultdict(int)
    for idx in common_score:
        pos = load_npy(score_pos[idx])
        neg = load_npy(score_neg[idx])
        score_pos_counts.append(pos.shape[0])
        score_neg_counts.append(neg.shape[0])
        for name, arr in (("pos", pos), ("neg", neg)):
            if arr.ndim != 2 or arr.shape[1] != 8:
                score_bad[f"{name}_shape"] += 1
            if not np.isfinite(numeric(arr)).all():
                score_bad[f"{name}_nonfinite"] += 1
            if arr.dtype == object:
                score_bad[f"{name}_object_dtype"] += 1
        if pos.shape[0] > 0 and pos.shape[1] >= 8:
            score_boxes_for_dims.append(pos[:, :7])
            pos_score_cols.append(numeric(pos[:, 7]))
        if neg.shape[0] > 0 and neg.shape[1] >= 8:
            neg_score_cols.append(numeric(neg[:, 7]))
    summarize_counts("scored_accepted_counts", score_pos_counts)
    summarize_counts("scored_noise_counts", score_neg_counts)
    print(f"accepted_score_col_percentiles: "
          f"{pct(np.concatenate(pos_score_cols)) if pos_score_cols else 'empty'}")
    print(f"noise_score_col_percentiles: "
          f"{pct(np.concatenate(neg_score_cols)) if neg_score_cols else 'empty'}")
    summarize_box_dims("scored_accepted_dims", score_boxes_for_dims)
    print(f"score_bad: {dict(score_bad)}")

    print_header("Config Inspection")
    if args.final_model_dir:
        inspect_yaml(os.path.join(args.final_model_dir, "config.yaml"))
    else:
        print("final model dir not provided; pass --final-model-dir to inspect config.yaml")

    print_header("High-Risk Checks")
    if len(pre_box) and len(score_pos) and len(pre_box) != len(score_pos):
        print("WARN: scored accepted file count differs from pre_box count.")
    if ratios:
        finite_ratios = np.asarray(ratios, dtype=np.float64)
        finite_ratios = finite_ratios[np.isfinite(finite_ratios)]
        if finite_ratios.size:
            mean_ratio = float(finite_ratios.mean())
            if mean_ratio < 0.05:
                print("WARN: MBE keeps very few boxes; pseudo labels may be too sparse.")
            if mean_ratio > 0.95:
                print("WARN: MBE rejects very few boxes; noisy labels may dominate.")
    if pos_score_cols:
        scores = np.concatenate(pos_score_cols)
        if np.nanmean(scores) <= 0.05:
            print("WARN: accepted score weights are near zero; regression loss may be almost disabled.")
        if np.nanmax(scores) > 5:
            print("WARN: accepted score weights exceed 5; regression loss may be over-weighted.")
    print("done")


if __name__ == "__main__":
    main()
