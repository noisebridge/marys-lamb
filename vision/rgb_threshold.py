import cv2
import numpy as np
import sys

# TODO: move this out to YAML

# TODO : make this Enum
BLUE=0
GREEN=1
RED=2
N_COLORS=3

class Thresholder:
    def __init__(self, img):
        self._upper_bound = np.array([255,]*3, dtype=np.uint8)
        self._lower_bound = np.array([0,]*3, dtype=np.uint8)
        self._img = np.copy(img)

    def lane_thresholded_img(self):
        thresholded = cv2.inRange(self._img, self._lower_bound, self._upper_bound)
        return cv2.bitwise_and(self._img, self._img, mask=thresholded) 

    @staticmethod
    def on_upper_trackbar(val, color, thresholder):
        thresholder._upper_bound[color] = val
        print("Upper threshold: ", thresholder._upper_bound)
        cv2.imshow("Thresholded image", thresholder.lane_thresholded_img())

    @staticmethod
    def on_lower_trackbar(val, color, thresholder):
        thresholder._lower_bound[color] = val
        print("Lower threshold: ", thresholder._lower_bound)
        cv2.imshow("Thresholded image", thresholder.lane_thresholded_img())

    @staticmethod
    def on_open_kernel_trackbar(val):
        val_odd = (val-1)*2+1
        print("Open kernel size", val_odd) 
        kernel = np.ones([val_odd, val_odd], np.uint8)
        img = thresholder.lane_thresholded_img()
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        cv2.imshow("Thresholded image", opened)


if __name__ == "__main__":
    img = cv2.imread(sys.argv[1])
    cv2.imshow("Input image: ", img)
    cv2.waitKey(0)
    img = cv2.resize(img, (500, 500))
    thresholder = Thresholder(img)
    trackbar_name = lambda side,i : "{} {} slider".format(side, i)
    title_window = "Thresholded slider"
    cv2.namedWindow(title_window, cv2.WINDOW_NORMAL)
    # Upper
    trackbar_upper_wrapper = list(map(lambda i: lambda val: Thresholder.on_upper_trackbar(val, i, thresholder), range(N_COLORS)))
    cv2.createTrackbar(trackbar_name("upper",0), title_window , 0, 255, trackbar_upper_wrapper[0])
    cv2.createTrackbar(trackbar_name("upper",1), title_window , 0, 255, trackbar_upper_wrapper[1])
    cv2.createTrackbar(trackbar_name("upper",2), title_window , 0, 255, trackbar_upper_wrapper[2])
    # Lower
    trackbar_lower_wrapper = list(map(lambda i: lambda val: Thresholder.on_lower_trackbar(val, i, thresholder), range(N_COLORS)))
    cv2.createTrackbar(trackbar_name("lower",0), title_window , 0, 255, trackbar_lower_wrapper[0])
    cv2.createTrackbar(trackbar_name("lower",1), title_window , 0, 255, trackbar_lower_wrapper[1])
    cv2.createTrackbar(trackbar_name("lower",2), title_window , 0, 255, trackbar_lower_wrapper[2])
    # Open kernel
    cv2.createTrackbar("Open slider", title_window, 0, 30, Thresholder.on_open_kernel_trackbar)
    for i in range(N_COLORS):
        trackbar_upper_wrapper[i](255)
        trackbar_lower_wrapper[i](0)

    cv2.waitKey()
    
    
