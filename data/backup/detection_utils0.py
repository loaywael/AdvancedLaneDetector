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


def getSobelGrads(grayImg, axis="xy", K=9):
    if axis == "x":
        xGrads = cv2.Sobel(grayImg, cv2.CV_64F, 1, 0, ksize=K)
        absXSobel = np.absolute(xGrads)
        return absXSobel
    elif axis == "y":
        yGrads = cv2.Sobel(grayImg, cv2.CV_64F, 0, 1, ksize=K)
        absYSobel = np.absolute(yGrads)
        return absYSobel
    elif axis == "xy":
        xGrads = cv2.Sobel(grayImg, cv2.CV_64F, 1, 0, ksize=K)
        yGrads = cv2.Sobel(grayImg, cv2.CV_64F, 0, 1, ksize=K)
        absXSobel = np.absolute(xGrads)
        absYSobel = np.absolute(yGrads)
        return absXSobel, absYSobel


def getSobelMag(absGrads, minThresh, maxThresh):
    if isinstance(absGrads, (list, tuple, np.ndarray)) and len(absGrads) == 2:
        absXSobel, absYSobel = absGrads
        sobelMag = np.sqrt(np.square(absXSobel) + np.square(absYSobel))
        sobelMag = np.uint8(sobelMag * 255 / np.max(sobelMag))
        edgesMask = np.zeros_like(sobelMag)
        edgesMask[(sobelMag > minThresh) & (sobelMag < maxThresh)] = 1
        cv2.imshow("f", edgesMask)
        return edgesMask
    else:
        sobelMag = np.sqrt(np.square(absGrads))
        sobelMag = np.uint8(sobelMag * 255 / np.max(sobelMag))
        edgesMask = np.zeros_like(sobelMag)
        edgesMask[(sobelMag > minThresh) & (sobelMag < maxThresh)] = 1
        return edgesMask


def getSobelDir(absXSobel, absYSobel, minThresh, maxThresh):
    sobelMag = np.arctan2(absYSobel, absXSobel)
    edgesMask = np.zeros_like(sobelMag)
    edgesMask[(sobelMag >= minThresh) & (sobelMag <= maxThresh)] = 1
    return edgesMask


def getLaneEdgesMask(grayImg):
    xGrads, yGrads = getSobelGrads(grayImg, "xy", K=15)
    magMask = getSobelMag([xGrads, yGrads], 55, 155)
    cv2.imshow("magGrads", xGrads)
    dirMask = getSobelDir(xGrads, yGrads, 0.75, 1.3)
    edgesMask = np.zeros_like(magMask)
    edgesMask[((xGrads == 1) & (yGrads == 1)) | ((magMask == 1) & (dirMask == 1))] = 1
    return edgesMask


def getSatThreshMask(hlsImg, minThresh, maxThresh=255):
    sChannel = hlsImg[:, :, 2]
    satMask = np.zeros_like(sChannel)
    satMask[(sChannel >= minThresh) & (sChannel <= maxThresh)] = 1
    return satMask


def getGRThreshMask(bgrImg, minThresh, maxThresh=255):
    g, r = bgrImg[:, :, 1], bgrImg[:, :, 2]
    grMask = np.zeros_like(r)
    grMask[((r >= minThresh) & (r <= maxThresh)) & ((g >= 1.125*minThresh) & (g <= maxThresh))] = 1
    return grMask


def getLaneColorMask(bgrImg, hlsImg):
    satMask = getSatThreshMask(hlsImg, 125)
    grMask = getGRThreshMask(bgrImg, 175)
    lanesMask = np.zeros_like(grMask)
    lanesMask[(satMask == 1) | (grMask == 1)] = 1
    return lanesMask


def getROI(bgrFrame, points):
    grayFrame = cv2.cvtColor(bgrFrame, cv2.COLOR_BGR2GRAY)
    hlsFrame = cv2.cvtColor(bgrFrame, cv2.COLOR_BGR2HLS)
    roiMask = np.zeros_like(grayFrame)
    lanesMask = np.zeros_like(grayFrame)
    lanesEdgesMask = getLaneEdgesMask(grayFrame)
    lanesColorMask = getLaneColorMask(bgrFrame, hlsFrame)
    # cv2.imshow("thresh", lanesColorMask)
    # cv2.imshow("edges", lanesEdgesMask)
    lanesMask[(lanesEdgesMask == 1) | (lanesColorMask == 1)] = 1
    cv2.fillPoly(roiMask, [np.array(points)], (255, 255, 255))
    outMask = cv2.bitwise_and(lanesMask, roiMask)
    return outMask


def warped2BirdPoly(bgrFrame, points, shape):
    warpedPoints = np.float32(points)
    birdPoints = np.float32([[0, 0], [shape[0], 0], shape, [0, shape[1]]])
    Matrix = cv2.getPerspectiveTransform(warpedPoints, birdPoints)
    return cv2.warpPerspective(bgrFrame, Matrix, shape, flags=cv2.INTER_LINEAR)


def save2File(path, object):
    with open(path, "wb") as wf:
        pickle.dump(object, wf)
        print("roi is saved!")


def loadFile(path):
    with open(path, "rb") as rf:
        return pickle.load(rf)
