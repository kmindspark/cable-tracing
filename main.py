from cable_tracing.utils.utils import *
from cable_tracing.tracers.simple_uncertain_trace import trace
import cv2
import numpy as np
import matplotlib.pyplot as plt
from annotate_bbox_start import KeypointsAnnotator

def annotate(img):
    pixel_selector = KeypointsAnnotator()
    orig_img = img.copy()
    annots = pixel_selector.run(orig_img)
    print(annots)
    annots = np.array(annots)
    return annots

if __name__ == "__main__":
    img_path = 'test.png' 
    color_img = (255 * plt.imread(img_path)).astype(np.uint8) 

    color_img[600:, :, :] = 0
    color_img[:, :100, :] = 0

    non_mask_img = color_img.copy()

    color_img = np.where(color_img < 90, 0, 255)
    depth_img = np.expand_dims(np.zeros(color_img.shape[:2]), axis=-1)
    img = np.concatenate((color_img, depth_img), axis=2)
   
    img[:, :, :3] = erode_image(img[:, :, :3], kernel=(2, 2))

    annots = annotate(img)

    top_left, bottom_right, start_point_1 = annots[0][::-1], annots[1][::-1], annots[2][::-1]
    delta_y = abs(top_left[0] - bottom_right[0])
    delta_x = abs(top_left[1] - bottom_right[1])

    start_point_2 = np.array([0, 0]) #dummy point #np.array([147, 63])      #np.array([448, 745]) // 2

    # y_min, x_min, delta_y, delta_x
    bboxes = np.array([
        [top_left[0], top_left[1], delta_y, delta_x]
    ])

    # disp_img = color_img.copy()
    # for bbox in bboxes:
    #     disp_img[:, :, :3] = cv2.rectangle(disp_img[:, :, :3].astype(np.uint8), (bbox[1], bbox[0]), (bbox[1]+bbox[3], bbox[0]+bbox[2]), (255, 0, 0), 3)
    # plt.imshow(disp_img[:, :, :3])
    # plt.show()

    # crop = img[top_left[0] : top_left[0] + delta_y, top_left[1] : top_left[1] + delta_x, :]
    # crop[:, :, :3] = erode_image(crop[:, :, :3], kernel=(2, 2))
    # plt.imshow(crop[:, :, :3])
    # plt.show()

    path, paths = trace(img, non_mask_img, start_point_1, start_point_2, exact_path_len=8, stop_when_crossing=False, x_min=top_left[1], x_max=bottom_right[1], y_min=top_left[0], y_max=bottom_right[0])

    if path is not None:
        
        plt.imshow(visualize_path(non_mask_img, path))
        plt.show()
    else:
        print("No path found, still showing all paths.")

    for path in paths[::-1]:
        print("displaying path with score:", score_path(img[:, :, :3], img[:, :, 3], path))

        # visualize_spline_in_3d(img, path)

        plt.imshow(visualize_path(non_mask_img, path))
        plt.show()

        plt.clf()
