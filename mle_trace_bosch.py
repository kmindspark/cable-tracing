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
from mpl_toolkits import mplot3d

MAX_RED_PIXEL_DIST = 200 # two adjacent red segments of rope cannot be farther than 200 pixels apart

if __name__ == "__main__":
    img_path = 'test_bosch.png' #'data_bank/large_figure8_simple/1640297369/color_0.npy'
    rgb_image = cv2.imread(img_path)
    # BGR to RGB
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    red_mask = rgb_image[:, :, 0] - rgb_image[:, :, 1] > 100
    # find connected components
    n_components, labels, stats, centroids = cv2.connectedComponentsWithStats(red_mask, connectivity=8)
    # put centers into a list
    centers = []
    for i in range(1, n_components):
        centers.append(centroids[i])

    start_point = np.array([57, 162]) 

    path, paths = explore_paths(img, start_point_1, start_point_2, stop_when_crossing=False)
    
    if path is not None:
        # plot 3d spline
        points = []
        for pt in path:
            pt = pt.astype(int)
            points.append(np.array([pt[0], pt[1], depth_img[pt[0], pt[1]]]))
            print(depth_img[pt[0], pt[1]])
        
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        lz = list(zip(*points))
        xs = np.array(lz[0]).squeeze()
        ys = np.array(lz[1]).squeeze()
        zs = np.array(lz[2]).squeeze()
        ax.plot3D(xs, ys, zs, 'gray')
        plt.show()

        plt.imshow(visualize_path(img, path))
        plt.show()
    else:
        print("No path found, still showing all paths.")

    for path in paths[::-1]:
        plt.imshow(visualize_path(img, path))
        plt.show()
