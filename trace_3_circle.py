import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import queue as Q
import cv2
from circle_fit import least_squares_circle

#from rigid_transformations import RigidTransform
file_id = '81927' #'99423' #'73886'

depth_file = f"all-phoxi-data/depth_{file_id}.npy" #all-phoxi-data/
array = np.load(depth_file)
print(array.shape)

color_file = f"all-phoxi-data/color_{file_id}.npy"
color_array = np.load(color_file)
# intensity threshold
threshold = 0.3 * 255
color_array[color_array < threshold] = 0
color_array[color_array >= threshold] = 1

# show array as image
plt.imshow(np.squeeze(array) * np.squeeze(color_array)[:, :, 0], cmap='gray')
plt.show()

masked = np.squeeze(array) * np.squeeze(color_array)[:, :, 0]
color_array *= 255

depth = {}
for i in range(array.shape[0]):
    for j in range(array.shape[1]):
        if masked[i, j] != 0:
            depth[(i, j)] = masked[i, j]
        else:
            depth[(i, j)] = 0

def cartesian_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def calculate_momentum(past_k_points):
    pass

def fit_line(points):
    # perform pca to minimize perpendicular distance
    pass

def fit_circle(points):
    return np.array(least_squares_circle(points)[:3])

def distance_to_circle(point, circle):
    return abs(np.sqrt(np.sum((point - circle[:2]) ** 2)) - circle[2])

def distance_to_line(point, segment):
    # segment is a tuple of two points
    # point is a tuple of two points
    # returns distance from point to line
    # https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
    x0, y0 = point
    x1, y1 = segment[0]
    x2, y2 = segment[1]

    A = np.array([x2 - x1, y2 - y1])
    B = np.array([x0 - x1, y0 - y1])

    return np.linalg.norm(np.cross(A, B)) / np.linalg.norm(A)

if __name__ == "__main__":
    start_point = (149, 555) #(88, 317) #(396, 336) # ##(370, 114)
    queue = [start_point]
    visited = set()
    past_k_diffs = []
    last_point = None

    # continuously render image with traversed points
    image = color_array.copy()
    cv2.imshow('image', image)
    cv2.waitKey(1)
    stop_cond = False
    depth_thresh = 0.0009

    k = 200
    hysteresis = 5
    past_k_points = []

    counter = 0
    while not stop_cond:
        while len(queue) != 0:
            counter += 1
            point = queue.pop(0)
            if len(past_k_points) == k:
                past_k_points = past_k_points[1:] + [list(point) + [depth[point]]]
                circle = fit_circle(past_k_points)
                recent_point = np.mean(past_k_points[-k//3:], axis=0)
                old_point = np.mean(past_k_points[:-k//3], axis=0)
            else:
                past_k_points.append(list(point) + [depth[point]])
                circle = None
                recent_point = None
                old_point = None

            adaptive_thresh = 0 #recent_z - older_z if counter > k else 0.0
            recent_depth = np.mean(past_k_points[max(0, -2 + len(past_k_points)):], axis=0)[2]

            # fill in pixel of point on image to red
            image[point[0], point[1], 0] = 1
            image[point[0], point[1], 1] = 0.5

            # re-render image
            if counter % 10 == 0:
                cpy = image.copy()
                if circle is not None:
                    cv2.circle(cpy, (int(circle[1]), int(circle[0])), int(circle[2]), (255, 0, 255), 2)
                cv2.imshow('image', cpy)
                cv2.waitKey(10)

            for w in [-1, 0, 1]:
                for h in [-1, 0, 1]:
                    if w == 0 and h == 0 :#or abs(w) + abs(h) != 1:
                        continue
                    new_point = (point[0] + w, point[1] + h)
                    if new_point not in visited and \
                        abs(depth[new_point] - recent_depth - 2*adaptive_thresh/k) < depth_thresh and \
                        (circle is None or distance_to_circle(new_point, circle)) < 3:
                        queue.append(new_point)
                        visited.add(new_point)

        # search across all pixels and find closest one to point that is height-viable
        # then add it to the queue
        closest_point, closest_distance = None, None

        print("searching from:", point)
        # just choose closest point in depth map
        point_to_start_from = np.array(point)
        # perform BFS to find next starting point
        loc_queue = [tuple(point_to_start_from.tolist())]
        loc_visited = visited.copy()
        loc_visited.add(loc_queue[0])
        while len(loc_queue) != 0:
            loc_point = loc_queue.pop(0)

            # image display code
            print("Searching: ", loc_point)
            image[loc_point[0], loc_point[1], 0] = 1
            image[loc_point[0], loc_point[1], 2] = 1
            cv2.imshow('image', image)
            cv2.waitKey(10)
            
            cart_dist = cartesian_distance(loc_point, point_to_start_from)
            if depth[loc_point] > 0 and abs(depth[loc_point] - recent_depth - 2*adaptive_thresh/k) < depth_thresh * (1 + cart_dist/10) and \
                loc_point != tuple(point_to_start_from.tolist()):
                break

            for w in [-1, 0, 1]:
                for h in [-1, 0, 1]:
                    if w == 0 and h == 0 :#or abs(w) + abs(h) != 1:
                        continue
                    new_point = (loc_point[0] + w, loc_point[1] + h)
                    if new_point not in loc_visited and \
                        (circle is None or distance_to_circle(new_point, circle) < 2.5) and \
                        (recent_point is None or np.dot(np.array(recent_point)[:2] - np.array(old_point)[:2], np.array(new_point)[:2] - np.array(recent_point)[:2]) >= 0):
                        loc_queue.append(new_point)
                        loc_visited.add(new_point)

        closest_point = loc_point
        print("closest point found", closest_point)
        if closest_point is None:
            break
        queue.append(closest_point)
        visited.add(closest_point)
        #past_k_points = past_k_points[k//2:]