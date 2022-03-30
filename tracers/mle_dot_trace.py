import numpy as np
import utils.utils as utils
from collections import OrderedDict
import matplotlib.pyplot as plt

MAX_CABLE_SEGMENTS = 100
MAX_LOOK_RADIUS = 50

def compute_edges(image, points, segmented_points_mask):
    point_to_edges = {}
    for point in points:
        point_to_edges[tuple(point)] = []

        radius = MAX_LOOK_RADIUS
        # print(point, np.nonzero(segmented_points_mask))

        min_y, max_y = max(0, point[0] - radius), min(image.shape[0], point[0] + radius)
        min_x, max_x = max(0, point[1] - radius), min(image.shape[1], point[1] + radius)
        next_points = np.where(segmented_points_mask[min_y:max_y, min_x:max_x])
        next_points = np.array(list(zip(next_points[0] + min_y, next_points[1] + min_x)))
        distances = np.linalg.norm(next_points - point, axis=1)

        for i in range(len(distances)):
            black_amount = utils.black_on_path(image[:, :, 0], point, next_points[i], num_to_check=20)
            if black_amount > 0.3:
                continue
            point_to_edges[tuple(point)].append((next_points[i], distances[i] + black_amount*50))
        
    return point_to_edges

# can you do something with graph partitioning? basically DP really
def get_valid_successors(path: OrderedDict, segmented_points_mask: np.ndarray, edges: dict):
    cur_path_points = np.array(list(path.keys()))
    # get last point in path
    last_point = cur_path_points[-1]
    second_last_point = cur_path_points[-2] if len(cur_path_points) > 1 else None
    
    # get edges
    edges_from_last_point = edges[tuple(last_point)]
        
    # filter out points that are already in path
    filtered_points = []
    distances = []
    for edge in edges_from_last_point:
        point = edge[0]
        if tuple(point) not in path:
            filtered_points.append(point)
            distances.append(edge[1])
    points = np.array(filtered_points)
    distances = np.array(distances)
    if len(filtered_points) == 0:
        return np.empty((0, 2))

    points = points[np.argsort(distances)]

    return points[:max(1, np.sum(distances < 60))]

def trace(image, start_point_1, stop_when_crossing=False, vis=True):
    if vis:
        # Show image and starting point
        plt.imshow(image[:, :, :3])
        plt.scatter(start_point_1[1], start_point_1[0], c='r')
        plt.show()

    segmented_points = utils.grid_cable_bfs(image, vis=vis, res=20)
    segmented_points_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    segmented_points_mask[segmented_points[:, 0], segmented_points[:, 1]] = 1

    # project start point to nearest segmented point
    start_point_1 = segmented_points[np.argmin(np.linalg.norm(segmented_points - np.array(start_point_1), axis=1))]

    # HOW DO WE SAVE MEMORY SO SIMILAR PATHS AREN'T COPIED A TON, SORT OF DP

    edges = compute_edges(image, segmented_points, segmented_points_mask)
    edges_image = utils.visualize_edges(image[:, :, :3], edges)
    if vis:
        plt.imshow(edges_image)
        plt.show()

    active_paths = [OrderedDict([(tuple(start_point_1), 0)])]
    finished_paths = []
    while len(active_paths) > 0:
        print("Active", len(active_paths))
        successor_points = get_valid_successors(active_paths[0], segmented_points_mask, edges)

        # if len(active_paths) > 80000:
        #     plt.imshow(utils.visualize_path(image, active_paths[0]))
        #     plt.show()

        for successor_point in successor_points:
            ordered_dict_copy = active_paths[0].copy()
            ordered_dict_copy[tuple(successor_point)] = 0
            active_paths.append(ordered_dict_copy)
        
        discard_path = active_paths.pop(0)
        if len(successor_points) == 0:
            finished_paths.append(np.array(list(discard_path.keys())))

    best_path = utils.get_best_path(image, finished_paths)
    return best_path, finished_paths
