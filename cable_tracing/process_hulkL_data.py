from utils.utils import *
# from tracers.mle_cont_trace import trace
# from tracers.mle_dot_dfs_trace import trace
from tracers.simple_uncertain_trace import trace
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)
import os

data_dirs = ['/Users/kaushikshivakumar/Documents/cable/untangling_long_cables_clean/hulkL_seg/train/hulkL_detectron_fail_anal_new_aug21',
             '/Users/kaushikshivakumar/Documents/cable/untangling_long_cables_clean/hulkL_seg/train/hulkL_detectron_more_complex_preRSS_aug29',
             '/Users/kaushikshivakumar/Documents/cable/untangling_long_cables_clean/hulkL_seg/train/hulkL_live_rollout_image_bank_sep2']
output_dir = 'hulkL_seg_traced'
thresh = 100

all_inputs_file_paths = []
for data_dir in data_dirs:
    for file in os.listdir(data_dir):
        if '.npy' not in file:
            continue
        all_inputs_file_paths.append(os.path.join(data_dir, file))         

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    plt.set_loglevel(level="info")
    
    for i, input_file in enumerate(sorted(all_inputs_file_paths)):
        out_path = os.path.join(output_dir, f'{i:03d}.npy')
        input_pickle = np.load(input_file, allow_pickle=True)

        img, annots = input_pickle.item()['img'], input_pickle.item()['annots']

        if len(annots) % 8 != 0:
            continue
            
        for annot in range(0, len(annots), 8):
            cur_img_cpy = img.copy()
            cur_img_cpy = np.where(cur_img_cpy > thresh, 255, 0).astype(np.uint8)

            bbox_corners = annots[annot: annot+2]
            min_x, min_y, max_x, max_y = np.min(bbox_corners[:, 0]), np.min(bbox_corners[:, 1]), np.max(bbox_corners[:, 0]), np.max(bbox_corners[:, 1])

            mask = np.zeros_like(img)
            mask[min_y:max_y, min_x:max_x] = 1
            cur_img_cpy *= mask

            trace_start_point_bbox = annots[annot+2: annot+4]
            trace_start_point_mask = np.zeros_like(img)
            cond_min_x, cond_min_y, cond_max_x, cond_max_y = np.min(trace_start_point_bbox[:, 0]), np.min(trace_start_point_bbox[:, 1]), np.max(trace_start_point_bbox[:, 0]), np.max(trace_start_point_bbox[:, 1])
            trace_start_point_mask[cond_min_y:cond_max_y, cond_min_x:cond_max_x] = 1
            condition_points = cur_img_cpy * trace_start_point_mask
            all_valid_cond_points = np.argwhere(condition_points[:, :, 0])
            
            border_points = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) - mask
            all_border_points = np.argwhere(border_points[:, :, 0])

            dist_matrix = np.linalg.norm(all_valid_cond_points[:, None, :] - all_border_points[None, :, :], axis=-1)

            min_dist_border_pt = dist_matrix.min(axis=1)
            closest_cond_pt_to_border = all_valid_cond_points[np.argmin(min_dist_border_pt, axis=0)]

            fake_depth_channel = np.zeros_like(img[:, :, :1])

            full_img = np.concatenate([cur_img_cpy, fake_depth_channel], axis=-1)

            path, paths = trace(full_img, closest_cond_pt_to_border, None, viz=True, viz_iter=1, timeout=30, termination_map=1-mask)

            print(path, paths)
            for path in paths:
                plt.imshow(visualize_path(img, path))
                plt.show()
            
            assert False