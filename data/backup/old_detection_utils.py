import cv2
import numpy as np
import pickle
import os


def updatePoints(event, x, y, flags, params):
    global points
    points, maxPoints = params
    if event == cv2.EVENT_LBUTTONDBLCLK:
        if len(points) == maxPoints:
            points = []
        else:
            points.append([x, y])


def drawRoIPoly(img, points):
    for pt in points:
        cv2.circle(img, pt, 7, (0, 0, 255), -1)
    cv2.polylines(img, [np.array(points)], True, (255, 0, 0), 2)


def getROI(frame, points):
    kernel = np.ones((3, 3), "uint8")
    roiMask = np.zeros(frame.shape[:2], "uint8")
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    grayFrame = cv2.GaussianBlur(grayFrame, (5, 5), 0)
    edgesMask = cv2.Canny(grayFrame, 33, 175)
    cv2.fillPoly(roiMask, [np.array(points)], (255, 255, 255))
    colorMask = getColorThresh(frame, 3, 175)
    outMask = cv2.bitwise_or(edgesMask, colorMask)
    outMask = cv2.bitwise_and(outMask, roiMask)
    outMask = cv2.dilate(outMask, kernel)
    # outMask = cv2.erode(outMask, kernel)
    return outMask


def getColorThresh(bgrFrame, k, th):
    rgbFrame = cv2.cvtColor(bgrFrame, cv2.COLOR_BGR2RGB)
    hsvFrame = cv2.cvtColor(rgbFrame, cv2.COLOR_RGB2HSV)
    h1, s1, v1 = hsvFrame[:, :, 0], hsvFrame[:, :, 1], hsvFrame[:, :, 2]
    hslFrame = cv2.cvtColor(rgbFrame, cv2.COLOR_RGB2HLS)
    h2, l2, s2 = hslFrame[:, :, 0], hslFrame[:, :, 1], hslFrame[:, :, 2]
    mask = cv2.bitwise_or(s1, s2)
    mask = cv2.bitwise_or(mask, rgbFrame[:, :, 0])
    # outMask = cv2.GaussianBlur(mask, (k, k), 0)
    outMask = np.zeros_like(mask)
    outMask[(mask > th) & (mask < 255)] = 255
    return outMask


def warped2BirdPoly(frame, points, shape):
    warpedPoints = np.float32(points)
    birdPoints = np.float32([[0, 0], [shape[0], 0], shape, [0, shape[1]]])
    Matrix = cv2.getPerspectiveTransform(warpedPoints, birdPoints)
    return cv2.warpPerspective(frame, Matrix, shape, flags=cv2.INTER_LINEAR)


def save2File(path, object):
    with open(path, "wb") as wf:
        pickle.dump(object, wf)
        print("roi is saved!")


def loadFile(path):
    with open(path, "rb") as rf:
        return pickle.load(rf)
