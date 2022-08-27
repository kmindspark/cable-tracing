import numpy as np
import utils.utils as utils
from collections import OrderedDict
import matplotlib.pyplot as plt
from tracers.mle_dot_trace import MAX_LOOK_RADIUS, MAX_CABLE_SEGMENTS, compute_edges, get_valid_successors

def trace(image, start_point_1, stop_when_crossing=False, vis=True):
    if vis:
        # Show image and starting point
        plt.imshow(image[:, :, :3])
        plt.scatter(start_point_1[1], start_point_1[0], c='r')
        plt.show()

    segmented_points = utils.grid_cable_bfs(image, vis=vis, res=10)
    segmented_points_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    segmented_points_mask[segmented_points[:, 0], segmented_points[:, 1]] = 1

    # project start point to nearest segmented point
    start_point_1 = segmented_points[np.argmin(np.linalg.norm(segmented_points - np.array(start_point_1), axis=1))]

    edges = compute_edges(image, segmented_points, segmented_points_mask)
    edges_image = utils.visualize_edges(image[:, :, :3], edges)
    if vis:
        plt.imshow(edges_image)
        plt.show()

    path_stack = [start_point_1]
    current_path = []
    cache = {}
    finished_paths = []
    last_certain_point = None # earliest point that had no branching
    while len(path_stack) > 0:
        # print("Active", len(path_stack))
        latest_point = path_stack[-1]
        path_stack = path_stack[:-1]

        successor_points = get_valid_successors(current_path, segmented_points_mask, edges)

        for successor_point in successor_points:
            path_stack.append(successor_point)
        if len(successor_points) == 0:
            finished_paths.append(list(current_path))
            current_path = current_path[:-1]
        elif len(successor_points) == 1:
            last_certain_point = last_certain_point or successor_points[0]
        else:
            # put all the consecutive pairs of points into the cache
            last_certain_point = None

    best_path = utils.get_best_path(image, finished_paths)
    return best_path, finished_paths
