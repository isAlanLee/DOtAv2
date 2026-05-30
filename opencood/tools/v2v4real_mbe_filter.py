import argparse
import os

import numpy as np
from tqdm import tqdm

from opencood.tools.mbe_v2v4real_utils import (
    boxes_to_world,
    load_scenario_points_and_poses,
    mbe_box_filter,
    pose_to_matrix,
    scenario_timestamps,
)


def main():
    parser = argparse.ArgumentParser(
        description='Stage-1 MBE filtering for V2V4Real.')
    parser.add_argument('--data_dir', required=True,
                        help='V2V4Real OPV2V-format train directory.')
    parser.add_argument('--pred_dir',
                        default='pseduo_label_val/pre_box_test_full',
                        help='Directory containing pre_{idx}.npy.')
    parser.add_argument('--out_dir', default='/root/autodl-tmp/out_v2v4real',
                        help='Directory for MBE v1 pseudo labels.')
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

        for frame_idx in tqdm(range(len(timestamps)),
                              desc=os.path.basename(scenario_folder),
                              leave=False):
            pred_path = os.path.join(args.pred_dir, 'pre_%d.npy' % global_idx)
            local_boxes = np.load(pred_path)
            world_boxes = boxes_to_world(local_boxes, poses[0][frame_idx])
            cav_positions = [
                pose_to_matrix(poses[cav_idx][frame_idx])[:3, 3]
                for cav_idx in range(len(cav_list))
            ]
            keep = mbe_box_filter(world_boxes, multi_agent_point,
                                  cav_positions, frame_idx)
            noise = np.logical_not(keep)

            np.save(os.path.join(
                args.out_dir, 'out_pseduo_labels_v1_%d.npy' % global_idx),
                local_boxes[keep])
            np.save(os.path.join(
                args.out_dir, 'out_pseduo_labels_noise_v1_%d.npy' % global_idx),
                local_boxes[noise])
            global_idx += 1


if __name__ == '__main__':
    main()
