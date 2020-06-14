import cv2
import pickle
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from detection_utils import getSatThreshMask, getGRThreshMask
from detection_utils import getHueThreshMask, getSobelGrads, getSobelMag


path = "./driving_datasets/"
videoPath = path + "project_video.mp4"

cap = cv2.VideoCapture(videoPath)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

i = 0
key = None
points = []
while cap.isOpened() and key != ord('q'):
    key = cv2.waitKey(10)
    ret, img = cap.read()    # reads bgr frame
    img = cv2.medianBlur(img, 5)
    hlsImg = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    if ret:
        xGrads = getSobelGrads(img, "x", K=5)
        magMask = getSobelMag([xGrads], 33, 200)
        satMask = getSatThreshMask(hlsImg, 110)
        grMask = getGRThreshMask(img, 200)
        hueMask = getHueThreshMask(hlsImg, 30)
        hsMask = cv2.bitwise_and(hueMask, satMask)
        lanesMask = np.zeros_like(grMask)
        lanesMask[(hsMask > 0) | (grMask > 0)] = 255
        outMask = cv2.bitwise_or(lanesMask, magMask)
        cv2.imshow("3", img)
        cv2.imshow("4", outMask)
