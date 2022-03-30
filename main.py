from utils.utils import *
# from tracers.mle_cont_trace import trace
from tracers.mle_dot_trace import trace
import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    img_path = 'data_bank/large_overhand_drop/1640297206/color_0.npy'
    #'data_bank/large_overhand_drop/1640297206/color_0.npy' #'data_bank/series_simple/1640295900/color_0.npy' #'data_bank/large_figure8_simple/1640297369/color_0.npy'
    color_img = np.load(img_path)

    color_img[600:, :, :] = 0
    color_img[:, :100, :] = 0

    color_img = np.where(color_img < 80, 0, 255)

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

    img[:, :, :3] = erode_image(img[:, :, :3])

    start_point_1 = np.array([460, 732]) // 2
    start_point_2 = np.array([448, 745]) // 2

    path, paths = trace(img, start_point_1)#, start_point_2, stop_when_crossing=False)
    
    # if path is None:
    #     path = paths[0]

    if path is not None:
        # plot 3d spline
        points = []
        for pt in path:
            pt = pt.astype(int)
            pt = closest_nonzero_pixel(pt, img[:, :, 3])
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
