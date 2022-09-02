import numpy as np
import cv2
import os

input_folder = '/Users/kaushikshivakumar/Documents/cable/untangling_long_cables_clean/live_rollout_image_bank'
output_folder = '/Users/kaushikshivakumar/Documents/cable/untangling_long_cables_clean/live_rollout_image_bank_png'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

counter = 0
for file in os.listdir(input_folder):
    # if 'color' not in file:
    #     continue
    img = np.load(os.path.join(input_folder, file),allow_pickle=True)[:, :, :3]
    out_path = os.path.join(output_folder, file.replace('.npy', '.png'))
    cv2.imwrite(out_path, img)