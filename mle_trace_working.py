import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import queue as Q
import cv2
from circle_fit import least_squares_circle
from collections import deque

STEP_SIZES = np.arange(3, 20, 4)
DEPTH_THRESH = 0.0011
COS_THRESH_SIMILAR = 0.95
COS_THRESH_FWD = 0.65
WIDTH_THRESH = 0

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
    if (max_distance > max(WIDTH_THRESH, max(STEP_SIZES))*1.1):
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
    num_black = 0
    # check if the line between pt and next_pt has a black pixel, using 10 samples spaced evenly along the line
    for i in range(10):
        cur_pt = pt + (next_pt - pt) * (i / 10)
        if color_img[int(cur_pt[0]), int(cur_pt[1])] == 0:
            num_black += 1
    return num_black > 2

def is_valid_successor(pt, next_pt, depth_img, color_img, pts, cur_dir, black_pixel_distances):
    next_pt_int = tuple(np.round(next_pt).astype(int))
    pt_int = tuple(np.round(pt).astype(int))
    # check if the next point is within the image
    if (next_pt_int[0] < 0 or next_pt_int[1] < 0 or next_pt_int[0] >= color_img.shape[0]
            or next_pt_int[1] >= color_img.shape[1]):
        return False
    is_centered = black_pixel_distances[tuple(next_pt_int)] > WIDTH_THRESH
    no_black_on_path = not has_black_on_path(color_img, pt, next_pt)
    if (not is_centered) or (not no_black_on_path):
        return False
    correct_dir = cur_dir.dot(normalize(next_pt - pt)) > COS_THRESH_FWD
    valid_depth = not (np.linalg.norm(next_pt - pt) < 10 and \
        abs(depth_img[next_pt_int] - depth_img[pt_int]) > DEPTH_THRESH)
    return is_centered and no_black_on_path and correct_dir and valid_depth

def is_similar(pt, next_pt_1, next_pt_2):
    cos_angle = np.dot(normalize(pt - next_pt_1), normalize(pt - next_pt_2))
    return cos_angle > COS_THRESH_SIMILAR

def dedup_candidates(pt, candidates, depth_img, color_img, pts, cur_dir, black_pixel_distances):
    # TODO: find a way of deduping such that we get exactly the branches we want
    # assumption is that candidates are sorted by distance from the current point
    filtered_candidates = []
    for i in range(len(candidates)):
        if is_valid_successor(pt, candidates[i], depth_img,
            color_img, pts, cur_dir, black_pixel_distances):
            sim_to_existing = False
            for j in range(len(filtered_candidates)):
                if is_similar(pt, candidates[i], filtered_candidates[j]):
                    sim_to_existing = True
                    break
            if not sim_to_existing:
                filtered_candidates.append(candidates[i])
    return filtered_candidates

def step_path(image, start_point, points_explored, black_pixel_distances):
    depth_img = image[:, :, 3]
    color_img = image[:, :, 0]

    # this will generally be a two-step process, exploring reasonable paths and then
    # choosing the best one based on the scores
    cur_point = start_point

    # points_explored should have at least one point
    cur_dir = normalize(start_point - points_explored[-1])

    # generate candidates for next point as every possible angle with step size of STEP_SIZE
    base_angle = np.arctan2(cur_dir[1], cur_dir[0])
    angle_thresh = np.arccos(COS_THRESH_FWD/2)
    # print(base_angle, angle_thresh)
    dx = np.cos(base_angle + np.arange(-angle_thresh, angle_thresh, np.pi / 90))
    dy = np.sin(base_angle + np.arange(-angle_thresh, angle_thresh, np.pi / 90))

    candidates = []
    for ss in STEP_SIZES:
        candidates.append(cur_point + np.array([dx, dy]).T * ss)

    candidates_flattened = np.array(candidates).reshape(-1, 2)
    deduplicated_candidates = dedup_candidates(cur_point, candidates_flattened, depth_img,
        color_img, points_explored, cur_dir, black_pixel_distances)
    return deduplicated_candidates, points_explored + [cur_point]

def visualize_path(img, path, black=False):
    def color_for_pct(pct):
        return (255*(1 - pct), 150, 255*pct) if not black else (0, 0, 0)
    img = img.copy()[:, :, :3].astype(np.uint8)
    for i in range(len(path) - 1):
        cv2.line(img, tuple(path[i].astype(int))[::-1], tuple(path[i + 1].astype(int))[::-1], color_for_pct(i/len(path)), 2 if not black else 5)
    return img

def is_too_similar(new_path, existing_paths):
    def pct_index(lst, pct):
        return lst[min(int(len(lst) * pct), len(lst) - 1)]

    # PRUNING FUNCTION FOR PATHS
    if len(new_path) > 150:
        return True

    for path in existing_paths:
        if np.linalg.norm(path[-1] - new_path[-1]) < 3:
            # for i in np.linspace(0, 1, 10):
            #     if np.linalg.norm(pct_index(path, i) - pct_index(new_path, i)) > 10:
            #         return False
            return True
    return False

def explore_paths(image, start_point_1, start_point_2):
    print("Starting exploring paths")
    black_pixel_distances = pixel_to_dist_from_nearest_black_point(image[:, :, 0])
    print("Done doing Dijkstra to find the black point distances")
    finished_paths = []
    active_paths = [[start_point_1, start_point_2]]

    iter = 0
    while len(active_paths) > 0:
        iter += 1
        print(iter, len(active_paths))
        cur_active_path = active_paths.pop(0)
        step_path_res = step_path(image, cur_active_path[-1], cur_active_path[:-1], black_pixel_distances)

        # given the new point, add new candidate paths
        if len(step_path_res[0]) == 0:
            finished_paths.append(step_path_res[1])
        else:
            num_active_paths = len(active_paths)
            for new_point in step_path_res[0]:
                if (not is_too_similar(step_path_res[1] + [new_point], active_paths[:num_active_paths])):
                    active_paths.append(step_path_res[1] + [new_point])
                # else: # TODO: delete this
                #     finished_paths.append(step_path_res[1])

    # init best score to min possible python value
    best_score, best_path = float('-inf'), None
    for path in finished_paths:
        score = score_path(image[:, :, :3], image[:, :, 3], path)
        if score > best_score:
            best_score = score
            best_path = path

    return best_path, finished_paths

if __name__ == "__main__":
    img_path = 'data_bank/nested_simple/1640296271/color_0.npy'
    color_img = np.load(img_path)

    color_img[600:, :, :] = 0
    color_img[:, :100, :] = 0

    color_img = remove_specks(np.where(color_img < 100, 0, 255))
    depth_img = np.load(img_path.replace('color', 'depth'))

    img = np.concatenate((color_img, depth_img), axis=2)
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

    start_point_1 = np.array([36, 84]) #np.array([144, 72]) #np.array([0, 323]) #np.array([300, 255]) / 2
    start_point_2 = np.array([36, 89]) #np.array([144, 78]) #np.array([6, 324]) #np.array([310, 265]) / 2

    plt.scatter(start_point_1[1], start_point_1[0], c='r')
    plt.scatter(start_point_2[1], start_point_2[0], c='b')
    plt.imshow(img[:, :, :3])
    plt.show()

    path, paths = explore_paths(img, start_point_1, start_point_2)
    
    if path is not None:
        plt.imshow(visualize_path(img, path))
        plt.show()
    else:
        print("No path found, still showing all paths.")

    for path in paths:
        plt.imshow(visualize_path(img, path))
        plt.show()
