from logging.config import valid_ident
import numpy as np
import time
from mpl_toolkits import mplot3d
from utils.utils import *

STEP_SIZES = np.array([16, 24]) # 10 and 20 #np.arange(3.5, 25, 10)
DEPTH_THRESH = 0.0030
COS_THRESH_SIMILAR = 0.97 #0.94
COS_THRESH_FWD = 0.0    #TODO: why does decreasing this sometimes make fewer paths?
WIDTH_THRESH = 0
NUM_POINTS_BEFORE_DIR = 1

step_path_time_sum = 0
step_path_time_count = 0
dedup_path_time_sum = 0
dedup_path_time_count = 0

step_cache = {}

def path_now_inside_bbox(path, bboxes):
    pass

def prep_for_cache(pt):
    return (pt[0]//3, pt[1]//3)

def is_valid_successor(pt, next_pt, depth_img, color_img, pts, pts_explored_set, cur_dir):
    next_pt_int = tuple(np.round(next_pt).astype(int))
    if next_pt_int in pts_explored_set:
        return False
    # check if the next point is within the image
    if (next_pt_int[0] < 0 or next_pt_int[1] < 0 or next_pt_int[0] >= color_img.shape[0]
            or next_pt_int[1] >= color_img.shape[1]):
        return False
    is_centered = color_img[next_pt_int] > 0

    no_black_on_path = black_on_path(color_img, pt, next_pt, dilate=False) <= 0.4

    correct_dir = True
    if cur_dir is not None:
        correct_dir = cur_dir.dot(normalize(next_pt - pt)) > COS_THRESH_FWD
    return is_centered and no_black_on_path and correct_dir

def is_similar(pt, next_pt_1, next_pt_2):
    cos_angle = np.dot(normalize(pt - next_pt_1), normalize(pt - next_pt_2))
    return cos_angle > COS_THRESH_SIMILAR and (np.linalg.norm(pt - next_pt_1) - np.linalg.norm(pt - next_pt_2)) < 1 \
        or cos_angle > (1*1 + COS_THRESH_SIMILAR)/2

def dedup_candidates(pt, candidates, depth_img, color_img, pts, pts_explored_set, cur_dir):
    # TODO: find a way of deduping such that we get exactly the branches we want
    # assumption is that candidates are sorted by distance from the current point
    filtered_candidates = []

    for tier in range(len(candidates)):
        if tier > 0 and len(filtered_candidates) > 0:
            return filtered_candidates
        cur_candidates = candidates[tier]
        for i in range(len(cur_candidates)):
            if is_valid_successor(pt, cur_candidates[i], depth_img,
                color_img, pts, pts_explored_set, cur_dir):
                sim_to_existing = False
                for j in range(len(filtered_candidates)):
                    if is_similar(pt, cur_candidates[i], filtered_candidates[j]):
                        sim_to_existing = True
                        break
                if not sim_to_existing:
                    filtered_candidates.append(cur_candidates[i])
                    if len(filtered_candidates) >= 3:
                        return filtered_candidates
    return filtered_candidates

def step_path(image, start_point, points_explored, points_explored_set):
    global step_path_time_count, step_path_time_sum, step_cache
    step_path_time_count += 1
    step_path_time_sum -= time.time()

    depth_img = image[:, :, 3]
    color_img = image[:, :, 0]

    # this will generally be a two-step process, exploring reasonable paths and then
    # choosing the best one based on the scores
    cur_point = start_point

    # points_explored should have at least one point
    cur_dir = normalize(start_point - points_explored[-1]) if len(points_explored) >= NUM_POINTS_BEFORE_DIR else None

    if not cur_dir is None:
        # generate candidates for next point as every possible angle with step size of STEP_SIZE
        base_angle = np.arctan2(cur_dir[1], cur_dir[0])
        angle_thresh = np.arccos(COS_THRESH_FWD/1.5)
    else:
        base_angle = 0
        angle_thresh = np.pi

    arange_len = 2 * int(np.ceil(angle_thresh * 90 / np.pi))
    c = np.zeros(arange_len)
    c[0::2] = base_angle + np.arange(0, angle_thresh, np.pi / 90)
    c[1::2] = base_angle - np.arange(0, angle_thresh, np.pi / 90)
    dx = np.cos(c)
    dy = np.sin(c)

    candidates = []
    for ss in STEP_SIZES:
        candidates.append(cur_point + np.array([dx, dy]).T * ss)

    pre_dedup_time = time.time()
    deduplicated_candidates = dedup_candidates(cur_point, candidates, depth_img,
        color_img, points_explored, points_explored_set, cur_dir)
    # print(f"Deduplication of {len(candidates[0])} candidates time", time.time() - pre_dedup_time)

    step_path_time_sum += time.time()
    return deduplicated_candidates

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
    if new_path_len[-1] > 1750:
        print("Path too long, stopping")
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

def get_updated_traversed_set(prev_set, prev_point, new_point, copy=True, sidelen=3):
    travel_vec = new_point - prev_point
    set_cpy = dict(prev_set) if copy else prev_set
    for t in range(0, int(np.linalg.norm(travel_vec)), sidelen):
        for i in range(-sidelen//2, sidelen//2 + 1):
            for j in range(-sidelen//2, sidelen//2 + 1):
                tp_to_add = tuple((prev_point + travel_vec*t/np.linalg.norm(travel_vec) + np.array([i, j])).astype(int))
                set_cpy[tp_to_add] = 0
    return set_cpy

def is_path_done(final_point, termination_map):
    return termination_map[tuple(final_point.astype(int))].sum() > 0

def trace(image, start_point_1, start_point_2, stop_when_crossing=False, resume_from_endpoint=False, timeout=30,
          bboxes=[], viz=True):
    termination_map = np.zeros(image.shape[:2] + (bboxes.shape[0],))
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        termination_map[bbox[0]:bbox[0]+bbox[2], bbox[1]:bbox[1]+bbox[3], i] = 1

    start_time = time.time()
    print("Starting exploring paths...")
    unfinished_paths_exist = False
    finished_paths = []
    active_paths = [[[start_point_1], {tuple(start_point_1): 0}]]

    iter = 0
    while len(active_paths) > 0:
        if iter % 100 == 0:
            print(f"Iteration {iter}, Active paths {len(active_paths)}")
            print("Visualizing path...")
        if viz and False: #iter > 100:
            plt.imshow(visualize_path(image, active_paths[0][0]))
            plt.show()

        if is_path_done(active_paths[0][0][-1], termination_map):
            finished_paths.append(active_paths[0][0])
            active_paths.pop(0)
            continue

        iter += 1
        cur_active_path = active_paths.pop(0)
        step_path_res = step_path(image, cur_active_path[0][-1], cur_active_path[0][:-1], cur_active_path[1])

        # given the new point, add new candidate paths
        if len(step_path_res) == 0:
            print("Finished current path, doesn't end in bounding box.")
            unfinished_paths_exist = True
        else:
            num_active_paths = len(active_paths)
            global dedup_path_time_sum, dedup_path_time_count
            dedup_path_time_count += 1
            dedup_path_time_sum -= time.time()
            for new_point_idx, new_point in enumerate(step_path_res):
                keep_path = not is_too_similar(cur_active_path[0] + [new_point], active_paths[:num_active_paths])
                if keep_path:
                    new_set = get_updated_traversed_set(cur_active_path[1], cur_active_path[0][-1], new_point, new_point_idx < len(step_path_res) - 1)
                    active_paths.append([cur_active_path[0] + [new_point], new_set])
            dedup_path_time_sum += time.time()
        # print("Full iter time", time.time() - start_iter_time)

        if time.time() - start_time > timeout:
            print("Timeout")
            return None, finished_paths
    
    # done exploring the paths
    tot_time = time.time() - start_time
    print("Done exploring paths, took {} seconds".format(tot_time))
    print("Time to step paths took {} seconds".format(step_path_time_sum))
    print("Time to dedup paths took {} seconds".format(dedup_path_time_sum))

    ending_points = []
    for path in finished_paths:
        ending_points.append(path[-1])
    # find dimensions of bounding box
    min_x = np.min(np.array([p[0] for p in ending_points]))
    max_x = np.max(np.array([p[0] for p in ending_points]))
    min_y = np.min(np.array([p[1] for p in ending_points]))
    max_y = np.max(np.array([p[1] for p in ending_points]))
    if max_y - min_y > 5 or max_x - min_x > 5:
        print(f"Bounding box ({max_y - min_y} x {max_x - min_y}) around ending points is too large, UNCERTAIN.")
        if viz:
            disp_img = color_img.copy()
            cv2.rectangle()
        return None, finished_paths
    else:
        return finished_paths[0], finished_paths