import numpy as np
import time
import torch

import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# import plotly.express as px
from scipy import interpolate
from cable_tracing.utils.utils import *
import logging
from torchvision import transforms
from hulk_keypoints.src.model import KeypointsGauss

def draw_spline(self, crop, x, y, label=False):
    """
    Draw a spline through the points (x,y).
    """

    if len(x) < 2:
        raise Exception("if drawing spline, must have 2 points minimum for label")
    k = len(x) - 1 if len(x) < 4 else 3
    tck,u     = interpolate.splprep( [x,y] ,s = 0, k=k)
    xnew,ynew = interpolate.splev( np.linspace( 0, 1, 100 ), tck,der = 0)
    xnew = np.array(xnew, dtype=int)
    ynew = np.array(ynew, dtype=int)

    x_in= np.where(xnew < crop.shape[0])
    xnew = xnew[x_in[0]]
    ynew = ynew[x_in[0]]
    y_in = np.where(ynew < crop.shape[1])
    xnew = xnew[y_in[0]]
    ynew = ynew[y_in[0]]

    spline = np.zeros(crop.shape[:2])
    if label:
        weights = np.ones(len(xnew))
    else:
        weights = np.geomspace(0.5, 1, len(xnew))

    spline[xnew, ynew] = weights
    spline = np.expand_dims(spline, axis=2)
    spline = np.tile(spline, 3)
    spline_dilated = cv2.dilate(spline, np.ones((5,5), np.uint8), iterations=1)
    return spline_dilated[:, :, 0]

def trace(image, start_point_1, start_point_2, stop_when_crossing=False, resume_from_edge=False, timeout=30,
          bboxes=[], termination_map=None, viz=True, exact_path_len=None, viz_iter=None, filter_bad=False, x_min=None, x_max=None, y_min=None, y_max=None, start_points=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KeypointsGauss(1).to(device)
    model.load_state_dict(torch.load('/Users/kaushikshivakumar/Documents/cable/untangling_long_cables_clean/cable_tracing/tracers/checkpoints/model_2_1_49.pth', map_location=device))
    transform = transforms.Compose([transforms.ToTensor()])

    num_condition_points = 4
    crop_size = 100

    if start_points is None or len(start_points) < num_condition_points:
        raise ValueError("Need at least 5 start points")
    path = start_points

    for _ in range(exact_path_len):
        condition_pixels = path[-num_condition_points:]
        crop_center = condition_pixels[-1]
        crop = image[crop_center[0]-crop_size:crop_center[0]+crop_size+1, crop_center[1]-crop_size:crop_center[1]+crop_size+1]
        spline = draw_spline(crop, condition_pixels[:, 0], condition_pixels[:, 1], label=False)
        crop[:, :, 0] = spline
        crop[:, :, 1] = 1 - spline

        if viz:
            plt.imshow(crop.cpu().numpy())
            plt.show()
        
        model_input = transform(crop.copy()).unsqueeze(0).to(device)
        model_output = model(model_input)

        argmax_yx = np.unravel_index(model_output.argmax(), model_output.shape[1:])
        global_yx = np.array([crop_center[0] - crop_size + argmax_yx[0], crop_center[1] - crop_size + argmax_yx[1]])

        path.append(global_yx)