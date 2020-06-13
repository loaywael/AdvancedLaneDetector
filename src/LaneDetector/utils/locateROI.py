import os
import cv2
import numpy as np
from time import time
from detection_utils import save2File, loadFile


path = "./driving_datasets/"
videoPath = path + "project_video.mp4"


cap = cv2.VideoCapture(videoPath)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cv2.namedWindow("control")
cv2.createTrackbar("p1X", "control", 527, width, lambda x: x)
cv2.createTrackbar("p1Y", "control", 500, height, lambda x: x)
cv2.createTrackbar("p4X", "control", 1070, width, lambda x: x)
cv2.createTrackbar("p4Y", "control", 666, height, lambda x: x)
cv2.createTrackbar("min-length", "control", 255, width, lambda x: x)
cv2.createTrackbar("max-length", "control", 775, width, lambda x: x)


key = None
while cap.isOpened() and key != ord('q'):
    key = cv2.waitKey(fps)
    ret, frame = cap.read()
    if ret:
        t1 = time()
        camModel = loadFile("camCalibMatCoeffs")
        camMtx = camModel["camMtx"]
        dstCoeffs = camModel["dstCoeffs"]
        undstFrame = cv2.undistort(frame, camMtx, dstCoeffs, None, camMtx)
        p1X = cv2.getTrackbarPos("p1X", "control")
        p1Y = cv2.getTrackbarPos("p1Y", "control")
        p4X = cv2.getTrackbarPos("p4X", "control")
        p4Y = cv2.getTrackbarPos("p4Y", "control")
        minLength = cv2.getTrackbarPos("min-length", "control")
        maxLength = cv2.getTrackbarPos("max-length", "control")
        pt1 = (p1X, p1Y)
        pt2 = (p1X+minLength, p1Y)
        pt3 = (p4X, p4Y)
        pt4 = (p4X-maxLength, p4Y)
        if p1X and p1Y:
            cv2.circle(undstFrame, pt1, 3, (0, 0, 255), -1)
        if minLength and p1X:
            cv2.circle(undstFrame, pt2, 3, (255, 0, 0), -1)
        if p4X and maxLength:
            cv2.circle(undstFrame, pt3, 3, (0, 255, 255), -1)
        if p4X and p4Y:
            cv2.circle(undstFrame, pt4, 3, (255, 0, 0), -1)
        points = [pt1, pt2, pt3, pt4]
        # cv2.imshow("frame", frame)
        cv2.imshow("undstFrame", undstFrame)
        t2 = time()
        print(f"time to execute: {t2 - t1}")
save2File("roiPoly", points)
cap.release()
cv2.destroyAllWindows()
