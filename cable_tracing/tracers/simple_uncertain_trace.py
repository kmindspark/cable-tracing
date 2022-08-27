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

def black_on_path(color_img, pt, next_pt, num_to_check=10, dilate=True):
    # dilation should be precomputed
    img_to_use = cv2.dilate(color_img, np.ones((4, 4), np.uint8)) if dilate else color_img#.copy()
    # if np.linalg.norm(pt - next_pt) < 5:
    #     return 0.0
    num_black = 0
    # check if the line between pt and next_pt has a black pixel, using 10 samples spaced evenly along the line
    for i in range(num_to_check):
        cur_pt = pt + (next_pt - pt) * (i / num_to_check)
        if img_to_use[int(cur_pt[0]), int(cur_pt[1])] == 0:
            num_black += 1
    return num_black/num_to_check

def erode_image(img, kernel=(1, 1)):
    img = img.astype(np.uint8)
    kernel = np.ones(kernel, np.uint8)
    return cv2.erode(img, kernel)

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

def closest_nonzero_pixel(pt, depth_img):
    # find the closest nonzero pixel to pt
    nonzero_pixels = np.nonzero(depth_img)
    # print(nonzero_pixels[0].shape)
    pts_combined = np.array([nonzero_pixels[0], nonzero_pixels[1]]).T
    distances = np.sqrt((pts_combined[:, 0] - pt[0]) ** 2 + (pts_combined[:, 1] - pt[1]) ** 2)
    return pts_combined[np.argmin(distances)]

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

def smooth_depth(depth_img):
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
    return depth_img

def visualize_depth_map_in_3d(depth):
    plt.imshow(depth)
    plt.clim(np.min(depth[np.nonzero(depth)]), np.max(depth[np.nonzero(depth)]))
    plt.show()

    points = []
    counter = 0
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            depth_val = depth[i, j]
            if depth_val <= 0.1:
                continue
            counter += 1
            if counter % 1 != 0:
                continue
            points.append(np.array([i, j, depth_val]))
    print("showing " + str(len(points)))
    points = np.array(points)

    # # fig = plt.figure()
    # # ax = plt.axes(projection='3d')
    lz = list(zip(*points))
    x = np.array(lz[0]).squeeze()
    y = np.array(lz[1]).squeeze()
    z = np.array(lz[2]).squeeze()

    data = [go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=2,
        )
    )]
    # show the plot
    fig = go.Figure(data=data)
    fig.show()
    # exit()

def visualize_spline_in_3d(img, path, plotly=True):
    points = []
    for pt in path:
        pt = pt.astype(int)
        pt = closest_nonzero_pixel(pt, img[:, :, 3])
        points.append(np.array([pt[0], pt[1], img[pt[0], pt[1], 3]]))
    
    lz = list(zip(*points))
    xs = np.array(lz[0]).squeeze()
    ys = np.array(lz[1]).squeeze()
    zs = np.array(lz[2]).squeeze()

    if plotly:
        fig = go.Figure(data=[go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode='markers+lines',
            marker=dict(
                size=2,
                color=[i for i in range(len(xs))],
            ),
            line=dict(
                color=[i for i in range(len(xs))],
            )
        )])
        fig.show()
    else:
        ax = plt.axes(projection='3d')
        for i in range(len(xs) - 1):
            ax.plot3D([xs[i], xs[i + 1]], [ys[i], ys[i + 1]], [zs[i], zs[i + 1]], c = [i/len(xs), 0, 1 - i/len(xs)])
        plt.show()

def dedup_and_center(image, points, dedup_dist):
    # greedy deduplicate points within distance 
    filtered_points = []
    too_close = False
    for pt in points:
        for i in range(len(filtered_points)):
            if np.linalg.norm(pt - filtered_points[i]) < dedup_dist:
                too_close = True
                break
        if not too_close:
            filtered_points.append(pt)
    
    centered_points = []
    for pt in filtered_points:
        centered_points.append(closest_nonzero_pixel(pt, image[:, :, 0]))

    return np.array(centered_points)
    

def grid_cable(image, vis=False, res=20, dedup_dist=14):
    orig_image = image.copy()
    image[:, :, :3] = cv2.dilate(image[:, :, :3], np.ones((3, 3), np.uint8))

    points = []
    U, V = np.nonzero(image[:, :, 0] > 0)
    pts = np.array([U, V]).T
    # choose pts with x or y divisible by 5

    for pt in pts:
        if pt[0] % res == 0 or pt[1] % res == 0:
            points.append(pt)
    points = np.array(points)
    points = dedup_and_center(orig_image, points, dedup_dist)

    if vis:
        plt.imshow(orig_image[:, :, :3])
        plt.scatter(points[:, 1], points[:, 0], s=10)
        plt.show()

    return points

def grid_cable_bfs(image, vis=False, res=40):
    queue = deque()

    points = []
    visited = np.zeros(image.shape[:2])
    counter = 0
    while visited.sum() < (image[:, :, 0] > 0).sum():
        start_point = closest_nonzero_pixel(np.array([0, 0]), (image[:, :, 0] > 0) - visited)
        visited[start_point[0], start_point[1]] = 1
        queue.append((start_point, 0))
        while len(queue) > 0:
            cur_pt = queue.popleft()
            cur_dist = cur_pt[1]
            if cur_dist % res == 0:
                points.append(cur_pt[0])
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == 0 and j == 0:
                        continue
                    next_pt = cur_pt[0] + np.array((i, j))
                    if next_pt[0] < 0 or next_pt[1] < 0 or next_pt[0] >= image.shape[0] or next_pt[1] >= image.shape[1]:
                        continue
                    if visited[next_pt[0], next_pt[1]] == 1:
                        continue
                    if image[next_pt[0], next_pt[1], 0] > 0:
                        queue.append((next_pt, cur_dist + 1))
                        visited[next_pt[0], next_pt[1]] = 1

    points = np.array(points)
    points = dedup_and_center(image, points, 5)

    if vis:
        plt.imshow(image[:, :, :3])
        plt.scatter(points[:, 1], points[:, 0], s=10)
        plt.show()

    return points


def visualize_path(img, path, black=False):
    def color_for_pct(pct):
        return colorsys.hsv_to_rgb(pct, 1, 1)[0] * 255, colorsys.hsv_to_rgb(pct, 1, 1)[1] * 255, colorsys.hsv_to_rgb(pct, 1, 1)[2] * 255
        # return (255*(1 - pct), 150, 255*pct) if not black else (0, 0, 0)
    img = img.copy()[:, :, :3].astype(np.uint8)
    for i in range(len(path) - 1):
        # if path is ordered dict, use below logic
        if not isinstance(path, OrderedDict):
            pt1 = tuple(path[i].astype(int))
            pt2 = tuple(path[i+1].astype(int))
        else:
            path_keys = list(path.keys())
            pt1 = path_keys[i]
            pt2 = path_keys[i + 1]
            # print(pt1, pt2)
        cv2.line(img, pt1[::-1], pt2[::-1], color_for_pct(i/len(path)), 2 if not black else 5)
    return img

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
        if (max_distance > 20): #max(WIDTH_THRESH, max(STEP_SIZES))*1.1):
            print("Invalid path")
            return float('-inf')

    # now assess sequential probability of the path by adding log probabilities
    total_log_prob = 0
    cur_dir = normalize(points[1] - points[0])
    for i in range(1, len(points) - 1):
        new_dir = normalize(points[i+1] - points[i])
        total_log_prob += abs(np.arccos(new_dir.dot(cur_dir))) #(np.log((new_dir.dot(cur_dir) + 1)/2)) * (np.linalg.norm(points[i+1] - points[i])) # adjust for distance
        # TODO: add depth differences and other metrics for evaluating
    return total_log_prob

def get_best_path(image, finished_paths, stop_when_crossing=False):
    # init best score to min possible python value
    best_score, best_path = float('-inf'), None
    for path in finished_paths:
        score = score_path(image[:, :, :3], image[:, :, 3], path, partial_paths=stop_when_crossing)
        if score > best_score:
            best_score = score
            best_path = path
    print("Best score", best_score, "Best path", best_path is not None)
    return best_path

def sort_paths_by_score(image, finished_paths, stop_when_crossing=False):
    # return a list of paths sorted by score
    scores = []
    for path in finished_paths:
        scores.append(score_path(image[:, :, :3], image[:, :, 3], path, partial_paths=stop_when_crossing))
    return np.array(finished_paths)[np.argsort(scores)]

def visualize_edges(image, edges):
    image = image.copy()
    for pt in edges.keys():
        plt.scatter(*pt[::-1], s=10, c='r')
        for second_pt in edges[pt]:
            cv2.line(image, pt[::-1], second_pt[0][::-1], (0, 0, 255), 1)
    return image

def delete_overlap_points(path, threshold=4):
    # delete points that are far apart in the spline
    # but too close to each other
    def within_threshold(pt1, pt2):
        return np.linalg.norm(pt1 - pt2) < threshold

    new_path = []
    for i in range(len(path)):
        add_point = True
        for j in range(len(new_path)):
            if within_threshold(path[i], new_path[j]):
                new_path.pop(j)
                add_point = False
                break
        if add_point:
            new_path.append(path[i])
    return np.array(new_path)


STEP_SIZES = np.array([16, 24]) # 10 and 20 #np.arange(3.5, 25, 10)
DEPTH_THRESH = 0.0030
COS_THRESH_SIMILAR = 0.95 #0.94
COS_THRESH_FWD = 0.0    #TODO: why does decreasing this sometimes make fewer paths?
WIDTH_THRESH = 0
NUM_POINTS_BEFORE_DIR = 1

step_path_time_sum = 0
step_path_time_count = 0
dedup_path_time_sum = 0
dedup_path_time_count = 0

step_cache = {}

def clean_input_color_image(image, start_point):
    img_orig = image.copy()
    image[:, :, 0] = cv2.dilate(image[:, :, 0].astype(np.uint8), np.ones((2, 2), dtype=np.uint8))
    output = cv2.connectedComponentsWithStats(image[:, :, 0].astype(np.uint8), 8, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    cable_class = labels[start_point[0], start_point[1]]
    return (labels == cable_class)[:, :, None] * img_orig

def path_now_inside_bbox(path, bboxes):
    pass

def prep_for_cache(pt):
    return (pt[0]//3, pt[1]//3)

def is_valid_successor(pt, next_pt, depth_img, color_img, pts, pts_explored_set, cur_dir, lenient=False):
    next_pt_int = tuple(np.round(next_pt).astype(int))
    if next_pt_int in pts_explored_set:
        return False
    # check if the next point is within the image
    if (next_pt_int[0] < 0 or next_pt_int[1] < 0 or next_pt_int[0] >= color_img.shape[0]
            or next_pt_int[1] >= color_img.shape[1]):
        return False
    is_centered = color_img[next_pt_int] > 0

    no_black_on_path = black_on_path(color_img, pt, next_pt, dilate=False) <= 0.3 if not lenient else 0.5

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

    for lenient in [False, True]:
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
                        if len(filtered_candidates) >= 3:
                            return filtered_candidates
        if len(filtered_candidates) > 0:
            break
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
    image = clean_input_color_image(image.copy(), start_point_1)

    viz=False
    termination_map = np.zeros(image.shape[:2] + (bboxes.shape[0],))
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        termination_map[bbox[0]:bbox[0]+bbox[2], bbox[1]:bbox[1]+bbox[3], i] = 1

    start_time = time.time()
    print("Starting exploring paths...")
    unfinished_paths_exist = False
    finished_paths = []
    active_paths = [[[np.array(start_point_1)], {tuple(start_point_1): 0}]]

    iter = 0
    while len(active_paths) > 0:
        if iter % 100 == 0:
            print(f"Iteration {iter}, Active paths {len(active_paths)}")
            print("Visualizing path...")
        if viz and True: #iter > 100:
            plt.imshow(visualize_path(image, active_paths[0][0]))
            plt.show()

        if is_path_done(active_paths[0][0][-1], termination_map):
            finished_paths.append(active_paths[0][0])
            active_paths.pop(0)
            continue

        iter += 1
        cur_active_path = active_paths.pop(0)
        step_path_res = step_path(image, cur_active_path[0][-1], cur_active_path[0][:-1], cur_active_path[1])
        print("Result of step path", step_path_res)
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

        if time.time() - start_time > (1 + 1e5*int(viz)) * timeout:
            print("Timeout")
            return None, finished_paths
    
    # done exploring the paths
    tot_time = time.time() - start_time
    print("Done exploring paths, took {} seconds".format(tot_time))
    print("Time to step paths took {} seconds".format(step_path_time_sum))
    print("Time to dedup paths took {} seconds".format(dedup_path_time_sum))

    ending_points = []
    # create tracing visualization
    side_len = np.ceil(np.sqrt(len(finished_paths)))
    side_len_2 = np.ceil(len(finished_paths)/side_len)
    fig, axs = plt.subplots(side_len, side_len_2)
    fig.title("All valid paths traced by cable until first knot.")
    for i in side_len:
        for j in side_len_2:
            axs[i, j].imshow(visualize_path(image, finished_paths[i*side_len + j]))
    fig.show()

    for path in finished_paths:
        plt.imshow(visualize_path(image, path))
        plt.show()
        ending_points.append(path[-1])
    ending_points = np.array(ending_points)
    # find dimensions of bounding box
    if ending_points.shape[0] == 0:
        print("No paths made it to any bounding box.")
        return None, None

    min_x = np.min(np.array([p[0] for p in ending_points]))
    max_x = np.max(np.array([p[0] for p in ending_points]))
    min_y = np.min(np.array([p[1] for p in ending_points]))
    max_y = np.max(np.array([p[1] for p in ending_points]))
    if max_y - min_y > 5 or max_x - min_x > 5:
        print(f"Bounding box ({max_y - min_y} x {max_x - min_x}) around ending points is too large, UNCERTAIN.")
        return None, finished_paths
    else:
        print("Successful trace, returning.")
        return finished_paths[0], finished_paths