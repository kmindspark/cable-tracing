from cable_tracing.utils.utils import *
from cable_tracing.tracers.simple_uncertain_trace import trace
import cv2
import numpy as np
import matplotlib.pyplot as plt
from annotate_bbox_start import KeypointsAnnotator
import os

def annotate(img):
    pixel_selector = KeypointsAnnotator()
    orig_img = img.copy()
    annots = pixel_selector.run(orig_img)
    print(annots)
    annots = np.array(annots)
    return annots

if __name__ == "__main__":
    img_path = './bowline'
    save_dir = './bowline_traces'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for f in sorted(os.listdir(img_path)):
        if f[-4:] != '.png':
            continue
        print(f)
        color_img = (255 * plt.imread(os.path.join(img_path, f))).astype(np.uint8) 

        color_img[600:, :, :] = 0
        color_img[:, :100, :] = 0

        non_mask_img = color_img.copy()

        color_img = np.where(color_img < 90, 0, 255)
        depth_img = np.expand_dims(np.zeros(color_img.shape[:2]), axis=-1)
        img = np.concatenate((color_img, depth_img), axis=2)
    
        img[:, :, :3] = erode_image(img[:, :, :3], kernel=(3, 3))

        annots = annotate(img)

        top_left, bottom_right, start_point_1 = annots[0][::-1], annots[1][::-1], annots[2][::-1]
        delta_y = abs(top_left[0] - bottom_right[0])
        delta_x = abs(top_left[1] - bottom_right[1])

        start_point_2 = np.array([0, 0]) #dummy point #np.array([147, 63])      #np.array([448, 745]) // 2

        # y_min, x_min, delta_y, delta_x
        bboxes = np.array([
            [top_left[0], top_left[1], delta_y, delta_x]
        ])

        path, paths, fig = trace(img, non_mask_img, start_point_1, start_point_2, exact_path_len=1000, stop_when_crossing=False, x_min=top_left[1], x_max=bottom_right[1], y_min=top_left[0], y_max=bottom_right[0])
        fig.savefig(os.path.join(save_dir, f))

        if path is not None:
            plt.imshow(visualize_path(non_mask_img, path))
            plt.show()
        else:
            print("No path found, still showing all paths.")

        path_of_max_cov = None
        max_score = float("-inf")
        for path in paths[::-1]:
            score = coverage_score(img[:, :, :3], path)
            print("displaying path with score:", score)
            if score > max_score:
                max_score = score
                path_of_max_cov = path

            plt.imshow(visualize_path(non_mask_img, path))
            plt.show()
            plt.clf()
        if path_of_max_cov == None:
            raise Exception("Wasn't able to trace.")
        save_max = save_dir + "/" + f[:-4] + "_max.png"
        cv2.imwrite(save_max, visualize_path(non_mask_img, path_of_max_cov))