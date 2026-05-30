import argparse
import os

import numpy as np
from tqdm import tqdm

from opencood.tools.mbe_v2v4real_utils import (
    append_scores,
    boxes_to_world,
    load_scenario_points_and_poses,
    scenario_timestamps,
)


def dense_points_for_frame(multi_agent_point, frame_idx, num_frames, window):
    start = max(0, frame_idx - window)
    end = min(num_frames, frame_idx + window)
    dense_points = []
    for dense_frame in range(start, end):
        for cav_idx in range(len(multi_agent_point)):
            dense_points.append(multi_agent_point[cav_idx][dense_frame])
    return np.concatenate(dense_points, axis=0) \
        if dense_points else np.empty((0, 4))


def main():
    parser = argparse.ArgumentParser(
        description='Stage-2 box scoring for V2V4Real MBE labels.')
    parser.add_argument('--data_dir', required=True,
                        help='V2V4Real OPV2V-format train directory.')
    parser.add_argument('--mbe_dir', default='/root/autodl-tmp/out_v2v4real',
                        help='Directory containing MBE v1 outputs.')
    parser.add_argument('--out_dir', default='/root/autodl-tmp/out_v2v4real',
                        help='Directory for scored pseudo labels.')
    parser.add_argument('--window', type=int, default=25,
                        help='Temporal window radius for scoring.')
    parser.add_argument('--max_scenarios', type=int, default=None,
                        help='Debug option: process only first N scenarios.')
    parser.add_argument('--keep_ground', action='store_true',
                        help='Keep ground points instead of original MBE removal.')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    scenario_folders = sorted([
        os.path.join(args.data_dir, x)
        for x in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, x))
    ])
    if args.max_scenarios is not None:
        scenario_folders = scenario_folders[:args.max_scenarios]

    global_idx = 0
    for scenario_folder in tqdm(scenario_folders, desc='scenarios'):
        cav_list = sorted([
            x for x in os.listdir(scenario_folder)
            if os.path.isdir(os.path.join(scenario_folder, x))
        ])
        timestamps = scenario_timestamps(scenario_folder, cav_list)
        multi_agent_point, poses = load_scenario_points_and_poses(
            scenario_folder, cav_list, timestamps,
            remove_ground=not args.keep_ground)
        num_frames = len(timestamps)

        for frame_idx in tqdm(range(num_frames),
                              desc=os.path.basename(scenario_folder),
                              leave=False):
            labels_path = os.path.join(
                args.mbe_dir, 'out_pseduo_labels_v1_%d.npy' % global_idx)
            noise_path = os.path.join(
                args.mbe_dir,
                'out_pseduo_labels_noise_v1_%d.npy' % global_idx)
            labels = np.load(labels_path)
            noise = np.load(noise_path)

            labels_world = boxes_to_world(labels, poses[0][frame_idx])
            noise_world = boxes_to_world(noise, poses[0][frame_idx])
            dense_points = dense_points_for_frame(
                multi_agent_point, frame_idx, num_frames, args.window)

            labels_with_score = append_scores(
                labels, labels_world, dense_points)
            noise_with_score = append_scores(
                noise, noise_world, dense_points)

            np.save(os.path.join(
                args.out_dir,
                'out_pseduo_labels_with_score_v4_%d.npy' % global_idx),
                labels_with_score)
            np.save(os.path.join(
                args.out_dir,
                'out_pseduo_labels_noise_with_score_v4_%d.npy' % global_idx),
                noise_with_score)
            global_idx += 1


if __name__ == '__main__':
    main()
