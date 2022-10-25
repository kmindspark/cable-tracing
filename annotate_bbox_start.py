import cv2
import numpy as np
import os
import math

class KeypointsAnnotator:
    def __init__(self):
        pass

    def load_image(self, img):
        self.img = img
        self.click_to_kpt = {0:"PULL1", 1:"PULL2"}

    def mouse_callback(self, event, x, y, flags, param):
        cv2.imshow("pixel_selector", self.img)
        if event == cv2.EVENT_LBUTTONDOWN: #cv2.EVENT_LBUTTONDBLCLK:
            # cv2.putText(img, self.click_to_kpt[len(self.clicks) % 2], (x,y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)
            # if len(self.clicks) % 8 == 0:
            #     self.clicks.append([0, 0])
            #     self.clicks.append(img.shape[:2]) # to tuple

            self.clicks.append([x, y])
            print("Clicked at: ", x, y)
            cv2.circle(self.img, (x, y), 3, (255, 0, 0), -1)

            if len(self.clicks) % 2 == 0: #crop
                clr = (255, 0, 0)
                if len(self.clicks) % 8 == 2: #conditioning
                    clr = (0, 255, 0)
                elif len(self.clicks) % 8 == 4: #cage
                    clr = (0, 0, 255)
                elif len(self.clicks) % 8 == 6: #pinch
                    clr = (255, 255, 0)
                cv2.rectangle(self.img, self.clicks[-2], self.clicks[-1], clr, 3)

    def run(self, img):
        self.load_image(img)
        self.clicks = []
        self.good_or_bad = []
        cv2.namedWindow('pixel_selector')#, cv2.WINDOW_NORMAL)
        cv2.resizeWindow('pixel_selector', 600, 600) 
        cv2.setMouseCallback('pixel_selector', self.mouse_callback)
        cv2.imshow('pixel_selector', img)
        while True:
            waitkey = cv2.waitKey(33)
            if waitkey & 0xFF == 27:
                break
            if waitkey == ord('r'):
                self.clicks = []
                self.load_image(img)
                print('Erased annotations for current image')

        cv2.destroyAllWindows()
        return self.clicks

if __name__ == '__main__':
    pixel_selector = KeypointsAnnotator()

    image_dir = './raw_data/cage_pinch_on_crossings/train/images/bowline'
    output_dir = 'raw_data/cage_pinch_on_crossings/train/annots/bowline'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    all_files = os.listdir(image_dir)

    start_index = int(input("Enter start index: "))
    for i,f in enumerate(sorted(all_files)):
        if f[-4:] != '.png':
            continue
        if (i) < start_index:
            continue

        print("Img %d"%i)
        image_path = os.path.join(image_dir, f)
        print(image_path)
        img = cv2.imread(image_path)
        orig_img = img.copy()

        full_outpath = os.path.join(output_dir, f[:-4] + '.npy')
        if os.path.exists(full_outpath):
            print(f"Skipping annotating image with annot file {full_outpath}")
            continue
        annots = pixel_selector.run(img)
        print("---")
        annots = np.array(annots)
        np.save(full_outpath, {'img': orig_img, 'annots': annots})
        print('image', i, 'annots', annots)