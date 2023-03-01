from logging.config import valid_ident
import numpy as np
import time
from mpl_toolkits import mplot3d
import os, sys

import numpy as np
import matplotlib.pyplot as plt
import cv2
import colorsys
import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# import plotly.express as px
from collections import deque, OrderedDict
import pandas as pd
from cable_tracing.utils.utils import *
import logging
import itertools
from itertools import product

STEP_SIZES = np.array([12, 18, 24, 32]) # 10 and 20 #np.arange(3.5, 25, 10)
DEPTH_THRESH = 0.0030
COS_THRESH_SIMILAR = 0.96 #0.94
COS_THRESH_FWD = 0.0    #TODO: why does decreasing this sometimes make fewer paths?
WIDTH_THRESH = 0
NUM_POINTS_BEFORE_DIR = 1
NUM_POINTS_TO_CONSIDER_BEFORE_RET = 20
IDEAL_IMG_DIM = 1032

step_path_time_sum = 0
step_path_time_count = 0
# dedup_path_time_sum = 0
# dedup_path_time_count = 0
# similar_check_time = 0
# get_updated_traversed_set_time = 0

step_cache = {}

logger = logging.getLogger("Untangling")

def clean_input_color_image(image, start_point):
    img_orig = image.copy()
    image[:, :, 0] = cv2.dilate(image[:, :, 0].astype(np.uint8), np.ones((2, 2), dtype=np.uint8))
    output = cv2.connectedComponentsWithStats(image[:, :, 0].astype(np.uint8), 8, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    # any class with > 100 pixels is valid
    # separate out labels classes into a third axis, where each slice i represents whether labels == i
    labels_3d = np.stack([labels == i for i in range(numLabels)], axis=2)
    valid_classes = np.argwhere(np.sum(labels_3d, axis=(0, 1)) > 100)
    return np.any(labels_3d[:, :, valid_classes], axis=2) * img_orig

def path_now_inside_bbox(path, bboxes):
    pass

def prep_for_cache(pt):
    return (pt[0]//3, pt[1]//3)

def is_valid_successor(pt, next_pt, depth_img, color_img, pts, pts_explored_set, cur_dir, lenient=False):
    next_pt_int = tuple(np.round(next_pt).astype(int))
    if color_img[next_pt_int] == 0:
        return False

    if next_pt_int in pts_explored_set:
        return False

    no_black_on_path = black_on_path(color_img, pt, next_pt, dilate=False) <= (0.4 if not lenient else 0.99)
    if not no_black_on_path:
        return False

    # check if the next point is within the image
    if (next_pt_int[0] < 0 or next_pt_int[1] < 0 or next_pt_int[0] >= color_img.shape[0]
            or next_pt_int[1] >= color_img.shape[1]):
        return False

    # correct_dir = True
    # if cur_dir is not None and not lenient:
    #     correct_dir = cur_dir.dot(normalize(next_pt - pt)) > COS_THRESH_FWD

    # if lenient:
    #     print("lenient cand:", is_centered, no_black_on_path, correct_dir)
    #     plt.scatter(next_pt[1], next_pt[0], c='r')
    return True #correct_dir

# def score_successor(pt, next_pts, depth_img, color_img, pts, pts_explored_set, cur_dir):

def is_similar(pt, next_pt_1, next_pt_2):
    cos_angle = np.dot(normalize(pt - next_pt_1), normalize(pt - next_pt_2))
    return cos_angle > COS_THRESH_SIMILAR and (np.linalg.norm(pt - next_pt_1) - np.linalg.norm(pt - next_pt_2)) < 1 \
        or cos_angle > (1*1 + COS_THRESH_SIMILAR)/2

def dedup_candidates(pt, candidates, depth_img, color_img, pts, pts_explored_set, cur_dir, num_pts_to_consider_before_ret=NUM_POINTS_TO_CONSIDER_BEFORE_RET):
    # TODO: find a way of deduping such that we get exactly the branches we want
    # assumption is that candidates are sorted by distance from the current point
    filtered_candidates = []
    counter = 0

    for lenient in ([False, True] if num_pts_to_consider_before_ret is not None else [False]): # TODO: add True in first list if we want to consider lenient
        # if lenient:
        #     plt.clf()
        #     plt.imshow(color_img)
        for tier in range(len(candidates)):
            if tier > 0 and len(filtered_candidates) > 0:
                return filtered_candidates
            cur_candidates = candidates[tier]
            for i in range(len(cur_candidates)):
                if is_valid_successor(pt, cur_candidates[i], depth_img,
                    color_img, pts, pts_explored_set, cur_dir, lenient=lenient):
                    sim_to_existing = False
                    for j in range(len(filtered_candidates)):
                        if is_similar(pt, cur_candidates[i], filtered_candidates[j]):
                            sim_to_existing = True
                            break
                    if not sim_to_existing:
                        filtered_candidates.append(cur_candidates[i])
                counter += 1
                if num_pts_to_consider_before_ret is not None and len(filtered_candidates) >= num_pts_to_consider_before_ret / counter:
                    return filtered_candidates
        # if lenient:
        #     plt.show()
    return filtered_candidates


def step_path(image, start_point, points_explored, points_explored_set, start_dir=None):
    global step_path_time_count, step_path_time_sum, step_cache
    start_step_time = time.time()
    step_path_time_count += 1
    step_path_time_sum -= time.time()

    depth_img = image[:, :, 0]
    color_img = image[:, :, 0]

    # this will generally be a two-step process, exploring reasonable paths and then
    # choosing the best one based on the scores
    cur_point = start_point

    # points_explored should have at least one point
    cur_dir = normalize(start_point - points_explored[-1]) if len(points_explored) >= NUM_POINTS_BEFORE_DIR else start_dir

    num_points_to_consider_before_ret = NUM_POINTS_TO_CONSIDER_BEFORE_RET
    if cur_dir is not None:
        # generate candidates for next point as every possible angle with step size of STEP_SIZE
        base_angle = np.arctan2(cur_dir[1], cur_dir[0])
        angle_thresh = np.arccos(COS_THRESH_FWD/1.5) * 1.4
        angle_increment = np.pi/90
    else:
        base_angle = 0
        angle_thresh = np.pi
        angle_increment = np.pi/45
        num_points_to_consider_before_ret = None

    arange_len = 2 * int(np.ceil(angle_thresh / angle_increment))
    c = np.zeros(arange_len)
    c[0::2] = base_angle + np.arange(0, angle_thresh, angle_increment)
    c[1::2] = base_angle - np.arange(0, angle_thresh, angle_increment)
    dx = np.cos(c)
    dy = np.sin(c)

    candidates = []
    for ss in STEP_SIZES:
        candidates.append(cur_point + np.array([dx, dy]).T * ss * image.shape[1]/IDEAL_IMG_DIM)

    print("Step, pre-dedup time:", time.time() - start_step_time)
    pre_dedup_time = time.time()
    deduplicated_candidates = dedup_candidates(cur_point, candidates, depth_img,
        color_img, points_explored, points_explored_set, cur_dir, num_points_to_consider_before_ret)
    print("Step, dedup time:", time.time() - pre_dedup_time)

    step_path_time_sum += time.time()
    return deduplicated_candidates

def is_too_similar(new_path, existing_paths):
    if len(existing_paths) > 20:
        return None

    new_path = np.array(new_path)
    def pct_index(lst, pct):
        return lst[min(int(len(lst) * pct), len(lst) - 1)]

    def length_index(lst, lns, lst_cumsum=None):
        # calculate distances between adjacent pairs of points
        distances_cumsum = get_dist_cumsum(lst) if lst_cumsum is None else lst_cumsum
        # lns is an array of values that we want to find the closest indices to
        i = np.argmax(distances_cumsum[:, np.newaxis] > lns[np.newaxis, :], axis=0)
        # interpolate
        pcts = (lns[:] - distances_cumsum[i - 1]) / (distances_cumsum[i] - distances_cumsum[i - 1])
        return lst[i - 1] + (lst[i] - lst[i - 1]) * pcts[:, np.newaxis]

    new_path_len = get_dist_cumsum(new_path)
    if new_path_len[-1] > 5000:
        logger.debug("Path too long, stopping.")
        return True

    for pth in existing_paths:
        pth = pth[0]
        path = np.array(pth)
        # TODO: do this right, check all (or subset) of the points
        path_len = get_dist_cumsum(path)
        min_len = min(path_len[-1], new_path_len[-1])
        lns = np.linspace(0.1*min_len, 1.0*min_len, 6)
        lns_indx = length_index(path, lns, path_len)
        lns_indx_new = length_index(new_path, lns, new_path_len)
        if np.linalg.norm(lns_indx[-1] - lns_indx_new[-1]) < 3: #2.5:
            if np.max(np.linalg.norm(lns_indx - lns_indx_new, axis=-1)) < 5: #4.5:
                # visualize both side by side
                # plt.imshow(np.concatenate((visualize_path(img, path), visualize_path(img, new_path)), axis=1))
                # plt.show()
                # if the paths are exactly identical before the two most recent points, don't consider them to be similar enough
                if abs(len(path) - len(new_path)) < 2:
                    min_len = min(len(path), len(new_path))
                    if np.linalg.norm(np.array(path[:min_len - 1]) - np.array(new_path[:min_len - 1]), axis=-1).sum() == 0:
                        continue
                return True
    return False

def get_pixels_of_path(path):
    # convert the path to a list of pixels, filling in visited pixels and the gaps
    visited_pixels = []
    visited_pixels_set = {}
    for i in range(len(path) - 1):
        segment = path[i + 1] - path[i]
        segment_len = np.linalg.norm(segment)
        for j in range(int(segment_len)):
            pct = j / segment_len
            pixel = path[i] + segment * pct
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    pixel_to_add = pixel + np.array([di, dj])
                    if pixel_to_add not in visited_pixels_set:
                        visited_pixels.append(pixel_to_add)
                        visited_pixels_set[pixel_to_add] = True
    return visited_pixels

def get_updated_traversed_set(image, prev_set, prev_point, new_point, copy=True, sidelen=15):
    travel_vec = new_point - prev_point
    set_cpy = dict(prev_set) if copy else prev_set
    b1 = np.arange(start = -sidelen//2, stop = sidelen//2)
    b2 = np.arange(start = -sidelen//2, stop = sidelen//2)
    buffer = list(itertools.product(b1, b2))
    buffer = np.array(buffer)
    for t in range(0, int(np.linalg.norm(travel_vec)), sidelen):
        curr_point = prev_point + travel_vec*t/np.linalg.norm(travel_vec)
        for p in (curr_point + buffer):
            set_cpy[(int(p[0]), int(p[1]))] = 0

    # spline = np.zeros(image.shape[:2])
    # for point in set_cpy:
    #     spline[point[0], point[1]] = 1.0
    # spline = np.tile(np.expand_dims(spline, axis=-1), 3)
    # cv2.imwrite("spline.png", spline * 255)
    return set_cpy

def is_path_done(final_point, termination_map):
    return termination_map[tuple(final_point.astype(int))].sum() > 0

def trace(image, start_point_1, start_dir=None, timeout=30,
          bboxes=[], viz=False, exact_path_len=None, viz_iter=0, endpoints=[]):
    global step_path_time_count, step_path_time_sum
    step_path_time_sum = 0
    step_path_time_count = 0
    dedup_path_time_sum = 0
    dedup_path_time_count = 0
    similar_check_time = 0
    get_updated_traversed_set_time = 0
    # cv2.imwrite("image_passed_in.png", image*255)
    image = np.where(image > 0.3, 255, 0).astype(np.uint8)

    # plt.imshow(image)
    # plt.title("Invocation")
    # plt.scatter(*start_point_1[::-1])
    # plt.show()

    if len(image.shape) < 3:
        image = np.stack((image, image, image), axis=-1)

    bboxes = np.array(bboxes)

    start_time = time.time()    
    finished_paths, finished_set_paths = [], []
    abandoned_paths, abandoned_set_paths = [], []
    active_paths = [[[np.array(start_point_1)], {tuple(start_point_1): 0}]]
    iter = 0
    while len(active_paths) > 0:
        # print(len(active_paths))
        if viz and viz_iter is not None and iter > viz_iter:
            plt.imsave('results/traces_left/debug.png', visualize_path(image*255, active_paths[0][0]))
            time.sleep(0.2)
            # pass

        if exact_path_len is not None and len(active_paths[0][0]) > exact_path_len:
            finished_path, finished_set_path = active_paths.pop(0)
            finished_paths.append(finished_path)
            finished_set_paths.append(finished_set_path)
            continue

        if len(endpoints) > 0 and np.min(np.linalg.norm(endpoints - active_paths[0][0][-1][None, :], axis=-1)) < 20 and len(active_paths[0][0]) > 15:
            finished_path, finished_set_path = active_paths.pop(0)
            finished_paths.append(finished_path)
            finished_set_paths.append(finished_set_path)
            continue

        iter += 1
        cur_active_path = active_paths.pop(0)
        step_path_res = step_path(image, cur_active_path[0][-1], cur_active_path[0][:-1], cur_active_path[1], start_dir=start_dir)
        # given the new point, add new candidate paths

        if len(step_path_res) == 0:
            abandoned_paths.append(cur_active_path[0])
            abandoned_set_paths.append(cur_active_path[1])

            abandoned_set_paths = abandoned_set_paths[-100:]
        else:
            num_active_paths = len(active_paths)
            dedup_path_time_count += 1
            dedup_path_time_sum -= time.time()
            for new_point_idx, new_point in enumerate(reversed(step_path_res)):
                similar_check_time -= time.time()
                keep_path = not is_too_similar(cur_active_path[0] + [new_point], active_paths[:num_active_paths])
                similar_check_time += time.time()
                if keep_path:
                    get_updated_traversed_set_time -= time.time()
                    new_set = get_updated_traversed_set(image, cur_active_path[1], cur_active_path[0][-1], new_point, new_point_idx < len(step_path_res) - 1)
                    get_updated_traversed_set_time += time.time()
                    active_paths.append([cur_active_path[0] + [new_point], new_set])
                # else:
                #     print("Dropping path, too similar to existing path")
            dedup_path_time_sum += time.time()
        if time.time() - start_time > (1 + 1e5*int(viz)) * timeout:
            break
            
    # done exploring the paths
    tot_time = time.time() - start_time
    print("Done exploring paths, took {} seconds".format(tot_time))
    print("Time to step paths took {} seconds".format(step_path_time_sum))
    print("Time to dedup paths took {} seconds".format(dedup_path_time_sum))
    print("time to check similarity took {} seconds".format(similar_check_time))
    print("time to get updated traversed set took {} seconds".format(get_updated_traversed_set_time))

    ending_points = []
    if viz and len(finished_paths) > 0:
        logger.debug("Showing trace visualizations")
        # create tracing visualization
        side_len_2 = np.ceil(np.sqrt(len(finished_paths))).astype(np.int32)
        side_len = np.ceil(len(finished_paths)/side_len_2).astype(np.int32)
        fig, axs = plt.subplots(side_len, side_len_2, squeeze=False)
        fig.suptitle(f"All {len(finished_paths)} valid paths traced by cable until first knot in {side_len} x {side_len_2} grid.")
        for i in range(side_len):
            for j in range(side_len_2):
                logger.debug(f"On {i}, {j}")
                if i*side_len + j < len(finished_paths):
                    logger.debug(f"Showing {i}, {j}")
                    axs[i, j].imshow(visualize_path(image, finished_paths[i*side_len + j]))
                axs[i, j].set_xticklabels([])
                axs[i, j].set_yticklabels([])
                axs[i, j].set_aspect('equal')
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()


    if len(finished_paths) == 0:
        finished_paths = abandoned_paths
        finished_set_paths = abandoned_set_paths

    for path in finished_paths:
        ending_points.append(path[-1])
    ending_points = np.array(ending_points)
    
    score_start_time = time.time()
    highest_score, highest_scoring_path = None, None
    for finished_path in finished_paths:
        score = score_path(image[..., :3], None, finished_path)
        if highest_score is None or score > highest_score:
            highest_score = score
            highest_scoring_path = finished_path

    score_tot_time = time.time() - score_start_time
    print("time to score took {} seconds".format(score_tot_time))
    print("===============")
    # plt.clf()
    # plt.title("Highest scoring path")
    # plt.imshow(visualize_path(image, highest_scoring_path))
    # plt.show()

    return highest_scoring_path, finished_paths