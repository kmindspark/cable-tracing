from utils.utils import *
# from tracers.mle_cont_trace import trace
# from tracers.mle_dot_dfs_trace import trace
from tracers.simple_uncertain_trace import trace
import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    img_path = 'png_data_bank/color_21.png' #'data_bank/series_simple/1640295900/color_0.npy'
    color_img = (255 * plt.imread(img_path)).astype(np.uint8)  #color_img = np.load(img_path)

    color_img[600:, :, :] = 0
    color_img[:, :100, :] = 0

    color_img = np.where(color_img < 90, 0, 255)
    depth_img = np.load(img_path.replace('color', 'depth').replace('.png', '.npy'))

    # crop the image
    # top_left = (590, 270)
    # bottom_right = (710, 430)
    # color_img = color_img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]
    # depth_img = depth_img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    # visualize_depth_map_in_3d(depth_img * (color_img[:, :, :1] > 0).astype(np.float32))

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
    # img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_NEAREST)

    img[:, :, :3] = erode_image(img[:, :, :3], kernel=(2, 2))
    # plt.imshow(img[:, :, :3])
    # plt.show()

    start_point_1 = np.array([346, 525])  #np.array([151, 69])      #np.array([460, 732]) // 2
    start_point_2 = np.array([357, 527]) #dummy point #np.array([147, 63])      #np.array([448, 745]) // 2

    bboxes = np.array([
        [153, 409, 170, 170]
    ])

    disp_img = color_img.copy()
    for bbox in bboxes:
        disp_img[:, :, :3] = cv2.rectangle(disp_img[:, :, :3].astype(np.uint8), (bbox[1], bbox[0]), (bbox[1]+bbox[3], bbox[0]+bbox[2]), (255, 0, 0), 3)
    plt.imshow(disp_img[:, :, :3])
    plt.show()

    path, paths = trace(img, start_point_1, start_point_2, stop_when_crossing=False, bboxes=bboxes)
    
    # if path is None:
    #     path = paths[0]

    if path is not None:
        # plot 3d spline
        # visualize_spline_in_3d(img, path)
        
        plt.imshow(visualize_path(img, path))
        plt.show()
    else:
        print("No path found, still showing all paths.")

    for path in paths[::-1]:
        print("displaying path with score:", score_path(img[:, :, :3], img[:, :, 3], path))

        # visualize_spline_in_3d(img, path)

        plt.imshow(visualize_path(img, path))
        plt.show()

        plt.clf()
