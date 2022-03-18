import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import queue as Q
import cv2
# from circle_fit import least_squares_circle
from collections import deque
import colorsys
import time
from mpl_toolkits import mplot3d
from utils import *

STEP_SIZES = np.array([10, 20]) #np.arange(3.5, 25, 10)
DEPTH_THRESH = 0.0030
COS_THRESH_SIMILAR = 0.95 #0.94
COS_THRESH_FWD = 0.0 # TODO: why does decreasing this make fewer paths?
WIDTH_THRESH = 0

step_path_time_sum = 0
step_path_time_count = 0
dedup_path_time_sum = 0
dedup_path_time_count = 0

step_cache = {}

def prep_for_cache(pt):
    return (pt[0]//3, pt[1]//3)

def score_path(color_img, depth_img, points, partial_paths=False):
    # get the MLE score for a given path through the image

    # find the farthest distance of a white pixel from all the points
    if not partial_paths:
        white_pixels = color_img.nonzero()
        points = np.array(points)
        distances = np.sqrt((white_pixels[0][None, :] - points[:, 0, None]) ** 2 + (white_pixels[1][None, :] - points[:, 1, None]) ** 2)
        max_distance = np.max(np.min(distances, axis=0), axis=0)
        argmax_distance = np.argmax(np.min(distances, axis=0), axis=0)
        print("Max distance:", max_distance, "at", white_pixels[0][argmax_distance], white_pixels[1][argmax_distance])
        if (max_distance > max(WIDTH_THRESH, max(STEP_SIZES))*1.1):
            print("Invalid path")
            return float('-inf')

    # now assess sequential probability of the path by adding log probabilities
    total_log_prob = 0
    cur_dir = normalize(points[1] - points[0])
    for i in range(1, len(points) - 1):
        new_dir = normalize(points[i+1] - points[i])
        total_log_prob += (np.log((new_dir.dot(cur_dir) + 1)/2)) * (np.linalg.norm(points[i+1] - points[i])) # adjust for distance
        # TODO: add depth differences and other metrics for evaluating
    return total_log_prob

def dist_to_closest_black_point(image, point):
    # INEFFICIENT WAY OF DOING THIS
    # black_points = (image == 0).nonzero()
    # distances = np.sqrt((black_points[0] - point[0]) ** 2 + (black_points[1] - point[1]) ** 2)
    # return np.array([black_points[0][np.argmin(distances)], black_points[1][np.argmin(distances)]]), np.min(distances)
    pass

def has_black_on_path(color_img, pt, next_pt):
    if np.linalg.norm(pt - next_pt) < 5:
        return 0.0
    num_black = 0
    # check if the line between pt and next_pt has a black pixel, using 10 samples spaced evenly along the line
    for i in range(10):
        cur_pt = pt + (next_pt - pt) * (i / 10)
        if color_img[int(cur_pt[0]), int(cur_pt[1])] == 0:
            num_black += 1
    return num_black/10

def is_valid_successor(pt, next_pt, depth_img, color_img, pts, pts_explored_set, cur_dir, lax=False):
    next_pt_int = tuple(np.round(next_pt).astype(int))
    pt_int = tuple(np.round(pt).astype(int))
    if next_pt_int in pts_explored_set:
        return False # TODO: account for anything in the nearby angle also though
    # check if the next point is within the image
    if (next_pt_int[0] < 0 or next_pt_int[1] < 0 or next_pt_int[0] >= color_img.shape[0]
            or next_pt_int[1] >= color_img.shape[1]):
        return False
    is_centered = True #black_pixel_distances[tuple(next_pt_int)] > WIDTH_THRESH
    no_black_on_path = has_black_on_path(color_img, pt, next_pt) <= (0.4 if not lax else 0.9)
    if lax:
        return is_centered and no_black_on_path
    if (not is_centered) or (not no_black_on_path):
        return False
    correct_dir = cur_dir.dot(normalize(next_pt - pt)) > COS_THRESH_FWD
    valid_depth = (not (np.linalg.norm(next_pt - pt) < 10 and \
        abs(depth_img[next_pt_int] - depth_img[pt_int]) > DEPTH_THRESH)) #depth_img[next_pt_int] > 0 and 
    return is_centered and no_black_on_path and correct_dir and valid_depth

def is_similar(pt, next_pt_1, next_pt_2):
    cos_angle = np.dot(normalize(pt - next_pt_1), normalize(pt - next_pt_2))
    return cos_angle > COS_THRESH_SIMILAR and (np.linalg.norm(pt - next_pt_1) - np.linalg.norm(pt - next_pt_2)) < 1 \
        or cos_angle > (1*1 + COS_THRESH_SIMILAR)/2

def dedup_candidates(pt, candidates, depth_img, color_img, pts, pts_explored_set, cur_dir, black_pixel_distances):
    # TODO: find a way of deduping such that we get exactly the branches we want
    # assumption is that candidates are sorted by distance from the current point
    filtered_candidates = []
    filtered_candidates_set = set()
    for lax in [False, True]:
        if lax and len(filtered_candidates) > 0:
            return filtered_candidates
        for tier in range(len(candidates)):
            if tier > 0 and len(filtered_candidates) > 0:
                return filtered_candidates
            cur_candidates = candidates[tier]
            for i in range(len(cur_candidates)):
                if is_valid_successor(pt, cur_candidates[i], depth_img,
                    color_img, pts, pts_explored_set, cur_dir, black_pixel_distances, lax=lax):
                    sim_to_existing = False
                    for j in range(len(filtered_candidates)):
                        if is_similar(pt, cur_candidates[i], filtered_candidates[j]):
                            sim_to_existing = True
                            break
                    if not sim_to_existing:
                        if tuple(cur_candidates[i].astype(int)) not in filtered_candidates_set:
                            filtered_candidates.append(cur_candidates[i])
                            filtered_candidates_set.add(tuple(cur_candidates[i].astype(int)))
                            if tier == 2:
                                print("Using tier 2")
                        if len(filtered_candidates) >= (3 if not lax else 1):
                            return filtered_candidates              
    # print("went through lax", len(filtered_candidates))
    return filtered_candidates

def step_path(image, start_point, points_explored, points_explored_set, black_pixel_distances):
    global step_path_time_count, step_path_time_sum, step_cache
    step_path_time_count += 1
    step_path_time_sum -= time.time()
    save_to_cache = True

    cur_tpl_key = (prep_for_cache(start_point), prep_for_cache(points_explored[-1]))
    if cur_tpl_key in step_cache:
        pass
        # ret_list = [a for a in step_cache[cur_tpl_key] if tuple(np.round(a).astype(int)) not in points_explored_set]
        # # TODO: Fix this
        # if len(ret_list) > 0:
        #     step_path_time_sum += time.time()
        #     print("Cache hit", ret_list)
        #     return ret_list, points_explored + [start_point]
        save_to_cache = False

    depth_img = image[:, :, 3]
    color_img = image[:, :, 0]

    # this will generally be a two-step process, exploring reasonable paths and then
    # choosing the best one based on the scores
    cur_point = start_point

    # points_explored should have at least one point
    cur_dir = normalize(start_point - points_explored[-1])

    # generate candidates for next point as every possible angle with step size of STEP_SIZE
    base_angle = np.arctan2(cur_dir[1], cur_dir[0])
    angle_thresh = np.arccos(COS_THRESH_FWD/1.5)
    # print(base_angle, angle_thresh)

    # TODO: Figure out way to prioritize straight stuff
    arange_len = 2 * int(np.ceil(angle_thresh * 90 / np.pi))
    c = np.zeros(arange_len)
    c[0::2] = base_angle + np.arange(0, angle_thresh, np.pi / 90)
    c[1::2] = base_angle - np.arange(0, angle_thresh, np.pi / 90)
    dx = np.cos(c)
    dy = np.sin(c)

    candidates = []
    for ss in STEP_SIZES:
        candidates.append(cur_point + np.array([dx, dy]).T * ss)

    # candidates_flattened = np.array(candidates).reshape(-1, 2)
    deduplicated_candidates = dedup_candidates(cur_point, candidates, depth_img,
        color_img, points_explored, points_explored_set, cur_dir, black_pixel_distances)
    if save_to_cache:
        step_cache[cur_tpl_key] = deduplicated_candidates
    step_path_time_sum += time.time()
    return deduplicated_candidates, points_explored + [cur_point]

def visualize_path(img, path, black=False):
    def color_for_pct(pct):
        return colorsys.hsv_to_rgb(pct, 1, 1)[0] * 255, colorsys.hsv_to_rgb(pct, 1, 1)[1] * 255, colorsys.hsv_to_rgb(pct, 1, 1)[2] * 255
        # return (255*(1 - pct), 150, 255*pct) if not black else (0, 0, 0)
    img = img.copy()[:, :, :3].astype(np.uint8)
    for i in range(len(path) - 1):
        cv2.line(img, tuple(path[i].astype(int))[::-1], tuple(path[i + 1].astype(int))[::-1], color_for_pct(i/len(path)), 2 if not black else 5)
    return img

def is_too_similar(new_path, existing_paths):
    if len(existing_paths) > 150:
        return None

    new_path = np.array(new_path)
    def pct_index(lst, pct):
        return lst[min(int(len(lst) * pct), len(lst) - 1)]

    def get_dist_cumsum(lst):
        lst_shifted = lst[1:]
        distances = np.linalg.norm(lst_shifted - lst[:-1], axis=1)
        # cumulative sum
        distances_cumsum = np.concatenate(([0], np.cumsum(distances)))
        return distances_cumsum

    def length_index(lst, lns, lst_cumsum=None):
        # calculate distances between adjacent pairs of points
        distances_cumsum = get_dist_cumsum(lst) if lst_cumsum is None else lst_cumsum
        # lns is an array of values that we want to find the closest indices to
        i = np.argmax(distances_cumsum[:, np.newaxis] > lns[np.newaxis, :], axis=0)
        # interpolate
        pcts = (lns[:] - distances_cumsum[i - 1]) / (distances_cumsum[i] - distances_cumsum[i - 1])
        # print(lst[i - 1].shape, pcts.shape)
        return lst[i - 1] + (lst[i] - lst[i - 1]) * pcts[:, np.newaxis]

    new_path_len = get_dist_cumsum(new_path)
    # PRUNING FUNCTION FOR PATHS
    if new_path_len[-1] > 1750:
        print("Path too long, stopping")
        return None

    # TODO: WHY ARE WE DIFFERENT WITH NEWER POINTS
    # TODO: CAN WE MAKE THERE BE FEWER POINT CHOICES
    for pth in existing_paths:
        pth = pth[0]
        path = np.array(pth)
        # TODO: do this right, check all (or subset) of the points
        # back = 1
        path_len = get_dist_cumsum(path)
        min_len = min(path_len[-1], new_path_len[-1])
        lns = np.linspace(0.1*min_len, 1.0*min_len, 6)
        lns_indx = length_index(path, lns, path_len)
        lns_indx_new = length_index(new_path, lns, new_path_len)
        if np.linalg.norm(lns_indx[-1] - lns_indx_new[-1]) < 2.5:
            if np.max(np.linalg.norm(lns_indx - lns_indx_new, axis=-1)) < 4.5:
                # visualize both side by side
                # plt.imshow(np.concatenate((visualize_path(img, path), visualize_path(img, new_path)), axis=1))
                # plt.show()

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

def explore_paths(image, start_point_1, start_point_2, stop_when_crossing=False, resume_from_endpoint=False, timeout=30):
    # TODO: think more about reducing the resolution of the image
    # time this function
    start_time = time.time()
    print("Starting exploring paths")
    black_pixel_distances = None #pixel_to_dist_from_nearest_black_point(image[:, :, 0])
    finished_paths = []
    active_paths = [[[start_point_1, start_point_2], {tuple(start_point_1): 0, tuple(start_point_2): 0}]]

    # TODO: how do we prevent the case where we delete two paths slightly different but only one can proceed? 
    iter = 0
    while len(active_paths) > 0:
        iter += 1
        if iter % 100 == 0:
            print(iter, len(active_paths))
        if iter > 4:
            # print(i, len(active_paths), len(active_paths[i][0]))
            vis = visualize_path(image, active_paths[0][0], black=False)
            plt.imshow(vis)
            plt.show()

        cur_active_path = active_paths.pop(0)
        step_path_res = step_path(image, cur_active_path[0][-1], cur_active_path[0][:-1], cur_active_path[1], black_pixel_distances)

        # given the new point, add new candidate paths
        if len(step_path_res[0]) == 0:
            finished_paths.append(step_path_res[1])
            # vis = visualize_path(image, finished_paths[-1], black=False)
            # plt.imshow(vis)
            # plt.show()
        else:
            num_active_paths = len(active_paths)
            global dedup_path_time_sum, dedup_path_time_count
            dedup_path_time_count += 1
            dedup_path_time_sum -= time.time()
            for new_point in reversed(step_path_res[0]):
                c = False
                discard_path = is_too_similar(step_path_res[1] + [new_point], active_paths[:num_active_paths])
                if (discard_path == False):
                    # block out a region around the line to the new point and continue
                    set_cpy = dict(cur_active_path[1])
                    travel_vec = new_point - cur_active_path[0][-1]
                    for t in range(0, int(np.linalg.norm(travel_vec)), 3):
                        for i in range(-1, 2):
                            for j in range(-1, 2):
                                tp_to_add = tuple((cur_active_path[0][-1] + travel_vec*t/np.linalg.norm(travel_vec) + np.array([i, j])).astype(int))
                                if tp_to_add not in set_cpy:
                                    set_cpy[tp_to_add] = len(step_path_res[1])
                                elif stop_when_crossing and len(step_path_res[1]) - set_cpy[tp_to_add] > 2:
                                    finished_paths.append(step_path_res[1][:set_cpy[tp_to_add]])
                                    c = True
                    if c:   continue
                    active_paths.append([step_path_res[1] + [new_point], set_cpy])
                elif (discard_path == None): # TODO: delete this
                    finished_paths.append(step_path_res[1])
            dedup_path_time_sum += time.time()

        if time.time() - start_time > timeout:
            print("Timeout")
            return None, finished_paths

    # init best score to min possible python value
    best_score, best_path = float('-inf'), None
    for path in finished_paths:
        score = score_path(image[:, :, :3], image[:, :, 3], path, partial_paths=stop_when_crossing)
        if score > best_score:
            best_score = score
            best_path = path
    print("Best score", best_score, "Best path", best_path is not None)

    tot_time = time.time() - start_time
    print("Done exploring paths, took {} seconds".format(tot_time))
    print("Time to step paths took {} seconds".format(step_path_time_sum))
    print("Time to dedup paths took {} seconds".format(dedup_path_time_sum))
    return best_path, finished_paths

if __name__ == "__main__":
    img_path = 'data_bank/large_overhand_drop/1640297206/color_0.npy' #'data_bank/series_simple/1640295900/color_0.npy' #'data_bank/large_figure8_simple/1640297369/color_0.npy'
    color_img = np.load(img_path)

    color_img[600:, :, :] = 0
    color_img[:, :100, :] = 0

    color_img = remove_specks(np.where(color_img < 80, 0, 255))
    depth_img = np.load(img_path.replace('color', 'depth'))

    # plt.imshow(color_img)
    # plt.show()

    # correct for depth image tilt
    # left_val = -0.145 #depth_img[:, 171].mean()
    # right_val = 0.0 #depth_img[:, 1020].mean()
    # table_offset = np.linspace(left_val, right_val, num=depth_img.shape[1])
    # depth_img += table_offset[:, None]

    depth_img *= (color_img[:, :, :1] > 0)
    depth_img = smooth_depth(depth_img)

    # plt.imshow(depth_img)
    # plt.show()
    # visualize_depth_map_in_3d(depth_img)

    # plt.imshow(np.where(depth_img <= 0, -1, depth_img - 1)) #table_offset[:, None] * np.ones(depth_img.shape))
    # plt.clim(-0.08, -0.065)
    # plt.colorbar()
    # plt.show()

    img = np.concatenate((color_img, depth_img), axis=2)
    # FIX RESIZE TO BILINEAR
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_NEAREST)

    start_point_1 = np.array([460, 732]) // 2
    start_point_2 = np.array([448, 745]) // 2

    path, paths = explore_paths(img, start_point_1, start_point_2, stop_when_crossing=False)
    
    # if path is None:
    #     path = paths[0]

    if path is not None:
        # plot 3d spline
        points = []
        for pt in path:
            pt = pt.astype(int)
            pt = closest_nonzero_depth_pixel(pt, img[:, :, 3])
            points.append(np.array([pt[0], pt[1], img[pt[0], pt[1], 3]]))
        
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        lz = list(zip(*points))
        xs = np.array(lz[0]).squeeze()
        ys = np.array(lz[1]).squeeze()
        zs = np.array(lz[2]).squeeze()

        for i in range(len(xs) - 1):
            ax.plot3D([xs[i], xs[i + 1]], [ys[i], ys[i + 1]], [zs[i], zs[i + 1]], c = [i/len(xs), 0, 1 - i/len(xs)])
        plt.show()

        plt.imshow(visualize_path(img, path))
        plt.show()
    else:
        print("No path found, still showing all paths.")

    for path in paths[::-1]:
        plt.imshow(visualize_path(img, path))
        plt.show()
