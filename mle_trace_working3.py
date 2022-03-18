import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import queue as Q
import cv2
from circle_fit import least_squares_circle
from collections import deque
import colorsys
import time

STEP_SIZES = np.arange(3, 25, 10)
DEPTH_THRESH = 0.0020
COS_THRESH_SIMILAR = 0.94 #0.97
COS_THRESH_FWD = 0.3 # TODO: why does decreasing this make fewer paths?
WIDTH_THRESH = 0

step_path_time_sum = 0
step_path_time_count = 0
dedup_path_time_sum = 0
dedup_path_time_count = 0

step_cache = {}

def prep_for_cache(pt):
    return (pt[0]//3, pt[1]//3)

def remove_specks(color_img):
    window_size = 5
    # remove tiny regions of stray white pixels 
    for i in range(color_img.shape[0] - window_size):
        for j in range(color_img.shape[1] - window_size):
            if np.sum(color_img[i:i+window_size, j:j+window_size, 0] > 0) < 3:
                color_img[i+window_size//2, j+window_size//2, :] = 0

    # zero out all edges of image
    color_img[0:window_size, :, :] = 0
    color_img[-window_size:, :, :] = 0
    color_img[:, 0:window_size, :] = 0
    color_img[:, -window_size:, :] = 0

    return color_img

def score_path(color_img, depth_img, points):
    # get the MLE score for a given path through the image
    # find the farthest distance of a white pixel from all the points
    white_pixels = color_img.nonzero()
    points = np.array(points)
    distances = np.sqrt((white_pixels[0][None, :] - points[:, 0, None]) ** 2 + (white_pixels[1][None, :] - points[:, 1, None]) ** 2)
    max_distance = np.max(np.min(distances, axis=0), axis=0)
    argmax_distance = np.argmax(np.min(distances, axis=0), axis=0)
    print("Max distance:", max_distance, "at", white_pixels[0][argmax_distance], white_pixels[1][argmax_distance])
    if (max_distance > max(WIDTH_THRESH, max(STEP_SIZES))*0.7):
        return float('-inf')
    
    # find farthest distance by visualizing
    # vis_img = visualize_path(color_img, points, black=True)
    # num_white = np.sum(vis_img > 0) / np.sum(color_img > 0)
    # print("Num white:", num_white)
    # plt.imshow(vis_img)
    # plt.show()
    # if num_white > 0.05:
    #     return float('-inf')
    
    # now assess sequential probability of the path by adding log probabilities
    total_log_prob = 0
    cur_dir = normalize(points[1] - points[0])
    for i in range(1, len(points) - 1):
        new_dir = normalize(points[i+1] - points[i])
        total_log_prob += (np.log((new_dir.dot(cur_dir) + 1)/2)) * (np.linalg.norm(points[i+1] - points[i])) # adjust for distance
        # TODO: add depth differences and other metrics for evaluating
    return total_log_prob

def normalize(vec):
    return vec / np.linalg.norm(vec)

def pixel_to_dist_from_nearest_black_point(image):
    # for each pixel, compute distance to nearest black pixel
    all_black = np.nonzero(image == 0)
    # add all black points to queue
    dq = deque()
    for i in range(len(all_black[0])):
        dq.append(np.array((all_black[0][i], all_black[1][i])))
    
    # initialize distances to infinity
    distances = np.full(image.shape, np.inf)
    distances[all_black] = 0

    # run dijkstra's algorithm
    # while len(q) > 0:
    #     print("Iter")
    #     closest_point, closest_dist = q[0], distances[tuple(q[0])]
    #     for i in range(1, len(q)):
    #         cur_pt_tuple = tuple(q[i])
    #         # print(distances.shape, q[i].shape, distances[tuple(q[i])])
    #         if distances[cur_pt_tuple] < closest_dist:
    #             closest_point, closest_dist = q[i], distances[cur_pt_tuple]
    #     q.remove(closest_point)

    #     # update distances
    #     for i in range(len(q)):
    #         distances[cur_pt_tuple] = min(distances[cur_pt_tuple], closest_dist + np.linalg.norm(q[i] - closest_point))
    
    # run BFS
    iters = 0
    while len(dq) > 0:
        iters += 1
        if iters % 100000 == 0:
            print("Iter", iters)
        next_pt = dq.popleft()
        
        # update distances
        for i in range(-1, 2):
            for j in range(-1, 2):
                cur_pt = next_pt + np.array((i, j))
                if not (cur_pt[0] < 0 or cur_pt[1] < 0 or cur_pt[0] >= image.shape[0]
                        or cur_pt[1] >= image.shape[1]):
                    if (distances[tuple(cur_pt)] == np.inf):
                        distances[tuple(cur_pt)] = distances[tuple(next_pt)] + 1
                        dq.append(cur_pt)
    return distances

def dist_to_closest_black_point(image, point):
    # INEFFICIENT WAY OF DOING THIS
    # black_points = (image == 0).nonzero()
    # distances = np.sqrt((black_points[0] - point[0]) ** 2 + (black_points[1] - point[1]) ** 2)
    # return np.array([black_points[0][np.argmin(distances)], black_points[1][np.argmin(distances)]]), np.min(distances)
    pass

def has_black_on_path(color_img, pt, next_pt):
    if np.linalg.norm(pt - next_pt) < 5:
        return False
    num_black = 0
    # check if the line between pt and next_pt has a black pixel, using 10 samples spaced evenly along the line
    for i in range(10):
        cur_pt = pt + (next_pt - pt) * (i / 10)
        if color_img[int(cur_pt[0]), int(cur_pt[1])] == 0:
            num_black += 1
    return num_black > 4

def is_valid_successor(pt, next_pt, depth_img, color_img, pts, pts_explored_set, cur_dir, black_pixel_distances, lax=False):
    next_pt_int = tuple(np.round(next_pt).astype(int))
    pt_int = tuple(np.round(pt).astype(int))
    if next_pt_int in pts_explored_set:
        return False # TODO: account for anything in the nearby angle also though

    # check if the next point is within the image
    if (next_pt_int[0] < 0 or next_pt_int[1] < 0 or next_pt_int[0] >= color_img.shape[0]
            or next_pt_int[1] >= color_img.shape[1]):
        return False
    is_centered = black_pixel_distances[tuple(next_pt_int)] > WIDTH_THRESH
    if lax:
        return is_centered
    no_black_on_path = not has_black_on_path(color_img, pt, next_pt)
    if (not is_centered) or (not no_black_on_path):
        return False
    correct_dir = cur_dir.dot(normalize(next_pt - pt)) > COS_THRESH_FWD
    valid_depth = not (np.linalg.norm(next_pt - pt) < 10 and \
        abs(depth_img[next_pt_int] - depth_img[pt_int]) > DEPTH_THRESH)
    return is_centered and no_black_on_path and correct_dir and valid_depth

def is_similar(pt, next_pt_1, next_pt_2):
    cos_angle = np.dot(normalize(pt - next_pt_1), normalize(pt - next_pt_2))
    return cos_angle > COS_THRESH_SIMILAR and (np.linalg.norm(pt - next_pt_1) - np.linalg.norm(pt - next_pt_2)) < 1

def dedup_candidates(pt, candidates, depth_img, color_img, pts, pts_explored_set, cur_dir, black_pixel_distances):
    # TODO: find a way of deduping such that we get exactly the branches we want
    # assumption is that candidates are sorted by distance from the current point
    filtered_candidates = []
    filtered_candidates_set = set()
    for lax in [False, True]:
        if lax and len(filtered_candidates) > 0:
            return filtered_candidates
        for tier in range(len(candidates)):
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
                        if len(filtered_candidates) >= (3 if not lax else 1):
                            return filtered_candidates              
    return filtered_candidates

def step_path(image, start_point, points_explored, points_explored_set, black_pixel_distances):
    global step_path_time_count, step_path_time_sum, step_cache
    step_path_time_count += 1
    step_path_time_sum -= time.time()

    cur_tpl_key = (prep_for_cache(start_point), prep_for_cache(points_explored[-1]))
    if cur_tpl_key in step_cache:
        step_path_time_sum += time.time()
        return [a for a in step_cache[cur_tpl_key] if tuple(np.round(a).astype(int)) not in points_explored_set], points_explored + [start_point]

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
        lns = np.linspace(0.1*min_len, 1.0*min_len, 3)
        lns_indx = length_index(path, lns, path_len)
        lns_indx_new = length_index(new_path, lns, new_path_len)
        if np.linalg.norm(lns_indx[-1] - lns_indx_new[-1]) < 2:
            if np.max(np.linalg.norm(lns_indx - lns_indx_new, axis=-1)) < 4:
                # visualize both side by side
                # plt.imshow(np.concatenate((visualize_path(img, path), visualize_path(img, new_path)), axis=1))
                # plt.show()

                if abs(len(path) - len(new_path)) < 2:
                    min_len = min(len(path), len(new_path))
                    if np.linalg.norm(np.array(path[:min_len]) - np.array(new_path[:min_len]), axis=-1).sum() == 0:
                        continue
                return True
    return False

def explore_paths(image, start_point_1, start_point_2):
    # time this function
    start_time = time.time()
    print("Starting exploring paths")
    black_pixel_distances = pixel_to_dist_from_nearest_black_point(image[:, :, 0])
    print("Done doing Dijkstra to find the black point distances")
    finished_paths = []
    active_paths = [[[start_point_1, start_point_2], {tuple(start_point_1), tuple(start_point_2)}]]

    # TODO: how do we prevent the case where we delete two paths slightly different but only one can proceed? 
    iter = 0
    while len(active_paths) > 0:
        iter += 1
        print(iter, len(active_paths))
        # if iter > 380:
        #     # print(i, len(active_paths), len(active_paths[i][0]))
        #     vis = visualize_path(image, active_paths[0][0], black=False)
        #     plt.imshow(vis)
        #     plt.show() #37 38 at 2000

        cur_active_path = active_paths.pop(0)
        step_path_res = step_path(image, cur_active_path[0][-1], cur_active_path[0][:-1], cur_active_path[1], black_pixel_distances)

        print(len(step_path_res[0]))
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
                discard_path = is_too_similar(step_path_res[1] + [new_point], active_paths[:num_active_paths])
                if (discard_path == False):
                    # block out a region around the line to the new point and continue
                    set_cpy = set(cur_active_path[1])
                    travel_vec = new_point - cur_active_path[0][-1]
                    for t in range(0, int(np.linalg.norm(travel_vec)), 3):
                        for i in range(-1, 2):
                            for j in range(-1, 2):
                                set_cpy.add(tuple((cur_active_path[0][-1] + travel_vec*t/np.linalg.norm(travel_vec) + np.array([i, j])).astype(int)))
                    active_paths.append([step_path_res[1] + [new_point], set_cpy])
                elif (discard_path == None): # TODO: delete this
                    finished_paths.append(step_path_res[1])
            dedup_path_time_sum += time.time()

    # init best score to min possible python value
    best_score, best_path = float('-inf'), None
    for path in finished_paths:
        score = score_path(image[:, :, :3], image[:, :, 3], path)
        if score > best_score:
            best_score = score
            best_path = path

    tot_time = time.time() - start_time
    print("Done exploring paths, took {} seconds".format(tot_time))
    print("Time to step paths took {} seconds".format(step_path_time_sum))
    print("Time to dedup paths took {} seconds".format(dedup_path_time_sum))
    return best_path, finished_paths

if __name__ == "__main__":
    img_path = 'data_bank/overhand_drop/1640295327/color_0.npy' #'data_bank/large_figure8_simple/1640297369/color_0.npy'
    color_img = np.load(img_path)

    color_img[600:, :, :] = 0
    color_img[:, :100, :] = 0

    color_img = remove_specks(np.where(color_img < 80, 0, 255))
    depth_img = np.load(img_path.replace('color', 'depth'))
    depth_img *= (color_img[:, :, :1] > 0)

    depth_cpy = depth_img.copy()
    # smooth the depth image with an average blur of non-zero values in a 3x3 window
    for i in range(depth_img.shape[0]):
        for j in range(depth_img.shape[1]):
            if depth_img[i, j] == 0:
                continue
            cnt = 0
            depth_img[i, j] = 0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if (i + di >= 0 and i + di < depth_img.shape[0] and \
                        j + dj >= 0 and j + dj < depth_img.shape[1] and \
                        depth_cpy[i + di, j + dj] > 0):
                        depth_img[i, j] += depth_cpy[i + di, j + dj]
                        cnt += 1
            depth_img[i, j] /= cnt

    img = np.concatenate((color_img, depth_img), axis=2)
    # FIX RESIZE TO BILINEAR
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_NEAREST)

    # plt.imshow(img[:, :, :3])
    # plt.show()

    start_point_1 = np.array([325, 546]) // 2 
    start_point_2 = np.array([319, 543]) // 2 

    path, paths = explore_paths(img, start_point_1, start_point_2)
    
    if path is not None:
        plt.imshow(visualize_path(img, path))
        plt.show()
    else:
        print("No path found, still showing all paths.")

    for path in paths[::-1]:
        plt.imshow(visualize_path(img, path))
        plt.show()
