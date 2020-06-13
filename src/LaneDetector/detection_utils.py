import cv2
import numpy as np
from time import time
from decimal import Decimal
import pickle
import os


def timeIt(func):
    def wrapper(*args, **kwargs):
        t1 = time()
        ret = func(*args, **kwargs)
        t2 = time()
        print(f"function: {func.__name__}, time to execute: {(t2 - t1)}s")
        return ret
    return wrapper


def updatePoints(event, x, y, flags, params):
    """
    Receives mouse clicking locations on screen to append them to a list to draw a polygon

    @param event: mouse event object from OpenCV
    @param x: x-location of the current point
    @param y: y-location of the current point
    @param flags: any given flag to the function
    @param params: additional parameters to be passed to the function
    """
    global points
    points, maxPoints = params
    if event == cv2.EVENT_LBUTTONDBLCLK:
        if len(points) == maxPoints:
            points = []
        else:
            points.append([x, y])


def drawRoIPoly(img, points):
    """
    Draws the Lane ROI polygon that visualize the points to be warped

    @param img: (np.ndarray) BGR image
    @points: list of points of the Lane ROI
    """
    for pt in points:
        cv2.circle(img, pt, 7, (0, 0, 255), -1)
    cv2.polylines(img, [np.array(points)], True, (255, 0, 0), 2)


# @timeIt
def getSatThreshMask(hlsImg, minThresh, maxThresh=255):
    """
    Creates a binary mask of the lane for the saturation channel (color thresholding)

    @param hlsImg: HLS image formate np.ndarray
    @param minThresh: the minimum theshold anything below is black
    @param maxThresh: the maximum theshold anything above is white
    """
    sChannel = hlsImg[:, :, 2]
    satMask = np.zeros_like(sChannel)
    satMask[(sChannel >= minThresh) & (sChannel <= maxThresh)] = 255
    return satMask


# @timeIt
def getGRThreshMask(bgrImg, minThresh, maxThresh=255):
    """
    Creates a binary mask of the lane for the green and the red channel

    @param bgrImg: (np.ndarray) BGR image
    @param minThresh: the minimum theshold anything below is black
    @param maxThresh: the maximum theshold anything above is white
    """
    g, r = bgrImg[:, :, 1], bgrImg[:, :, 2]
    grMask = np.zeros_like(r)
    grMask[((r >= minThresh) & (r <= maxThresh)) & (
        (g >= 1.125*minThresh) & (g <= maxThresh))] = 255
    return grMask


# @timeIt
def getHueThreshMask(hlsImg, minThresh, maxThresh=255):
    """
    Creates a binary mask of the lane for the hue channel (color thresholding)

    @param hlsImg: HLS image formate np.ndarray
    @param minThresh: the minimum theshold anything below is black
    @param maxThresh: the maximum theshold anything above is white
    """
    hChannel = hlsImg[:, :, 0]
    hueMask = np.zeros_like(hChannel)
    hueMask[(hChannel <= minThresh) & (hChannel >= 0)] = 255
    return hueMask


# @timeIt
def getSobelGrads(grayImg, axis="xy", K=9):
    """
    Creates a binary mask of the absolute edges Sobel gradients from a grayscale image

    @param grayImg: Gray image formate np.ndarray
    @param axis: the axis to extract gradients over
    @param K: Sobel kernel size
    """
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


# @timeIt
def getSobelMag(absGrads, minThresh, maxThresh):
    """"
    Applies gradient magnitude for the absGrads and returns binary edges mask

    @param absGrads: np.ndarray of the absolute gradients binary mask
    @param minThresh: the minimum theshold anything below is black
    @param maxThresh: the maximum theshold anything above is white
    """
    if isinstance(absGrads, (list, tuple, np.ndarray)) and len(absGrads) == 2:
        absXSobel, absYSobel = absGrads
        sobelMag = np.sqrt(np.square(absXSobel) + np.square(absYSobel))
        sobelMag = np.uint8(sobelMag * 255 / np.max(sobelMag))
        edgesMask = np.zeros_like(sobelMag)
        edgesMask[(sobelMag > minThresh) & (sobelMag < maxThresh)] = 255
        return edgesMask
    else:
        sobelMag = np.sqrt(np.square(absGrads[-1]))
        sobelMag = np.uint8(sobelMag * 255 / np.max(sobelMag))
        edgesMask = np.zeros_like(sobelMag)
        edgesMask[(sobelMag > minThresh) & (sobelMag < maxThresh)] = 255
        return edgesMask


# @timeIt
def getSobelDir(absXSobel, absYSobel, minThresh, maxThresh):
    """
    Finds gradients direction of a given absolute gradients of an image

    @param absXGrads: np.ndarray of the absolute gradients binary mask of X-axis
    @param absYGrads: np.ndarray of the absolute gradients binary mask of Y-axis
    @param minThresh: the minimum theshold anything below is black
    @param maxThresh: the maximum theshold anything above is white
    """
    sobelMag = np.arctan2(absYSobel, absXSobel)
    edgesMask = np.zeros_like(sobelMag)
    edgesMask[(sobelMag >= minThresh) & (sobelMag <= maxThresh)] = 255
    return edgesMask


@timeIt
def getLaneMask(bgrFrame, minLineThresh, maxLineThresh,
                satThresh, hueThresh, redThresh, filterSize=5):
    """
    Applies color bluring, thresholding and edge detection on a bgr image
    Returns lanes detected in a binary mask

    @param bgrImg: (np.ndarray) BGR image
    @param minLineThresh: minimum theshold for edge detector
    @param maxLineThresh: maximum theshold for edge detector
    @param satThresh: saturation channel threshold any value above is 255 else is 0
    @param hueThresh: hue channel threshold any value above is 255 else is 0
    @param redThresh: red, and green channel threshold any value above is 255 else is 0
    """
    # bgrFrame = cv2.medianBlur(bgrFrame, filterSize)
    grayImg = cv2.cvtColor(bgrFrame, cv2.COLOR_BGR2GRAY)
    # grayImg = cv2.GaussianBlur(grayImg, (filterSize, filterSize), 0)
    hlsImg = cv2.cvtColor(bgrFrame, cv2.COLOR_BGR2HLS)
    edgesMask = cv2.Canny(grayImg, minLineThresh, maxLineThresh)
    # xGrads = getSobelGrads(grayImg, "x", K=5)
    # edgesMask = getSobelMag([xGrads], minLineThresh, maxLineThresh)
    _, satMask = cv2.threshold(hlsImg[:, :, 2], 175, 255, cv2.THRESH_BINARY)
    _, hueMask = cv2.threshold(hlsImg[:, :, 2], 33, 255, cv2.THRESH_BINARY_INV)
    _, redMask = cv2.threshold(bgrFrame[:, :, 2], 170, 255, cv2.THRESH_BINARY)
    _, greenMask = cv2.threshold(bgrFrame[:, :, 1], 170, 255, cv2.THRESH_BINARY)
    hueSatMask = cv2.bitwise_and(hueMask, satMask)
    redGreenMask = cv2.bitwise_and(redMask, greenMask)
    laneMask = cv2.bitwise_or(redGreenMask, hueSatMask)
    outMask = cv2.bitwise_or(laneMask, edgesMask)
    # cv2.imshow("0", outMask)
    return outMask


def getinitialCenters(warpedFrame):
    """
    Estimates the begining of the lane lines and locate its center points
    by computing peaks in histogram and finding the y-axis from the first
    white pixel and the x-axis from the bottom half and taking its median point

    Returns
    =======
    initialLeftXCenter: (int) X-axis position of the left lane line
    initialRightXCenter: (int) X-axis position of the right lane line
    xPixsHisto: (np.ndarray) histogram of pixels of the given image along X-axis

    @param warpedFrame: bird view binary image of the lane roi
    """
    xPixsHisto = np.sum(warpedFrame[warpedFrame.shape[0]//2:], axis=0)
    midPoint = xPixsHisto.shape[0]//2
    leftXcPoint = np.argmax(xPixsHisto[:midPoint])
    rightXcPoint = np.argmax(xPixsHisto[midPoint:]) + midPoint
    return leftXcPoint, rightXcPoint, xPixsHisto


@timeIt
def fitLaneLines(leftLinePoints, rightLinePoints, lineLength, order=2):
    """
    Fits lines to a given lane lines

    @param leftLinePoints: (tuple) lane lines points X, Y (np.ndarray)
    @param rightLinePoints: (tuple) lane lines points X, Y (np.ndarray)
    @param lineLength: Height of lane line
    @param order: degree of polynomial equation that fits the lane lines
    """
    leftXPoints, leftYPoints = leftLinePoints
    rightXPoints, rightYPoints = rightLinePoints
    leftLineParams = np.polyfit(leftYPoints, leftXPoints, order)
    rightLineParams = np.polyfit(rightYPoints, rightXPoints, order)
    linesParams = (leftLineParams, rightLineParams)
    lineYVals = np.linspace(0, lineLength, lineLength)
    a, b, c = leftLineParams
    leftLineXVals = a*lineYVals**2 + b*lineYVals + c
    a, b, c = rightLineParams
    rightLineXVals = a*lineYVals**2 + b*lineYVals + c
    return linesParams, (leftLineXVals, lineYVals), (rightLineXVals, lineYVals)


@timeIt
def getLanePoints(warpedFrame, nWindows, windowWidth, pixelsThresh):
    """
    Applies blind search for the lane points (white pixels on black background)
    using the sliding window algorithm starting from the centers of the histogram peaks

    Returns
    =======
    outImg: (np.3darray) result of search plotted over
    leftLinePoints: (tuple) lane lines points X, Y (np.ndarray)
    rightLinePoints: (tuple) lane lines points X, Y (np.ndarray)

    @param warpedFrame: (np.ndarray) bird view binary image of the lane roi
    @param nWindows: (int) number of windows allowed to be stacked on top of each other
    @param windowWidth: (int) window width (horizontal distance between diagonals)
    @param pixelsThresh: (int) minimum points to be considered as part of the lane
    """
    leftLanePixelsIds = []
    rightLanePixelsIds = []
    noneZeroIds = warpedFrame.nonzero()
    noneZeroXIds = np.array(noneZeroIds[1])
    noneZeroYIds = np.array(noneZeroIds[0])
    leftXcPoint, rightXcPoint, xPixsHisto = getinitialCenters(warpedFrame)
    windowHeight = np.int(warpedFrame.shape[0] // nWindows)
    rightCenter = (rightXcPoint, warpedFrame.shape[0] - windowHeight//2)
    leftCenter = (leftXcPoint, warpedFrame.shape[0] - windowHeight//2)
    outImg = np.dstack([warpedFrame, warpedFrame, warpedFrame])

    for i in range(1, nWindows+1):
        leftLinePt1 = (leftCenter[0]-windowWidth//2, leftCenter[1]-windowHeight//2)
        leftLinePt2 = (leftCenter[0]+windowWidth//2, leftCenter[1]+windowHeight//2)
        rightLinePt1 = (rightCenter[0]-windowWidth//2, rightCenter[1]-windowHeight//2)
        rightLinePt2 = (rightCenter[0]+windowWidth//2, rightCenter[1]+windowHeight//2)
        cv2.rectangle(outImg, leftLinePt1, leftLinePt2, (0, 255, 0), 3)
        cv2.rectangle(outImg, rightLinePt1, rightLinePt2, (0, 255, 0), 3)

        leftWindowXIds = (noneZeroXIds > leftLinePt1[0]) & (noneZeroXIds < leftLinePt2[0])
        leftWindowYIds = (noneZeroYIds > leftLinePt1[1]) & (noneZeroYIds < leftLinePt2[1])
        leftWindowPoints = leftWindowXIds & leftWindowYIds
        rightWindowXIds = (noneZeroXIds > rightLinePt1[0]) & (noneZeroXIds < rightLinePt2[0])
        rightWindowYIds = (noneZeroYIds > rightLinePt1[1]) & (noneZeroYIds < rightLinePt2[1])
        rightWindowPoints = rightWindowXIds & rightWindowYIds

        leftLanePixelsIds.append(leftWindowPoints)
        rightLanePixelsIds.append(rightWindowPoints)

        leftCenter = leftCenter[0], leftCenter[1] - windowHeight
        if leftWindowPoints.sum() > pixelsThresh:
            #         xC = np.int(np.median(noneZeroXIds[leftWindowPoints]))
            lXc = np.int(noneZeroXIds[leftWindowPoints].mean())
            leftCenter = (lXc, leftCenter[1])

        rightCenter = rightCenter[0], rightCenter[1] - windowHeight
        if rightWindowPoints.sum() > pixelsThresh:
            #         xC = np.int(np.median(noneZeroXIds[leftWindowPoints]))
            rXc = np.int(noneZeroXIds[rightWindowPoints].mean())
            rightCenter = (rXc, rightCenter[1])

    leftLanePixelsIds = np.sum(np.array(leftLanePixelsIds), axis=0).astype("bool")
    rightLanePixelsIds = np.sum(np.array(rightLanePixelsIds), axis=0).astype("bool")
    leftXPoints = noneZeroXIds[leftLanePixelsIds]
    leftYPoints = noneZeroYIds[leftLanePixelsIds]
    rightXPoints = noneZeroXIds[rightLanePixelsIds]
    rightYPoints = noneZeroYIds[rightLanePixelsIds]

    return (leftXPoints, leftYPoints), (rightXPoints, rightYPoints)


def predictLaneLines(warpedFrame, linesParams, margin):
    """
    Predicts lane line in a new frame based on previous detection from blind search

    Returns
    =======
    outImg: (np.3darray) result of search plotted over
    leftLinePoints: (tuple) lane lines points X, Y (np.ndarray)
    rightLinePoints: (tuple) lane lines points X, Y (np.ndarray)

    @param warpedFrame: (np.ndarray) bird view binary image of the lane roi
    @param linesParams: (dict) previous fitted params
    @param margin: (int) lane line detection boundry width
    """
    leftLineParams, rightLineParams = linesParams
    noneZeroIds = warpedFrame.nonzero()
    noneZeroXIds = np.array(noneZeroIds[1])
    noneZeroYIds = np.array(noneZeroIds[0])

    a, b, c = leftLineParams
    leftLineLBoundry = a*noneZeroYIds**2 + b*noneZeroYIds + c - margin
    leftLineRBoundry = a*noneZeroYIds**2 + b*noneZeroYIds + c + margin
    a, b, c = rightLineParams
    rightLineLBoundry = a*noneZeroYIds**2 + b*noneZeroYIds + c - margin
    rightLineRBoundry = a*noneZeroYIds**2 + b*noneZeroYIds + c + margin

    leftLineBoundryIds = (noneZeroXIds > leftLineLBoundry) & (noneZeroXIds < leftLineRBoundry)
    rightLineBoundryIds = (noneZeroXIds > rightLineLBoundry) & (noneZeroXIds < rightLineRBoundry)

    leftLineBoundryX = noneZeroXIds[leftLineBoundryIds]
    leftLineBoundryY = noneZeroYIds[leftLineBoundryIds]
    rightLineBoundryX = noneZeroXIds[rightLineBoundryIds]
    rightLineBoundryY = noneZeroYIds[rightLineBoundryIds]

    linesParams, leftLinePoints, rightLinePoints = fitLaneLines(
        (leftLineBoundryX, leftLineBoundryY),
        (rightLineBoundryX, rightLineBoundryY),
        warpedFrame.shape[0]-1
    )

    return linesParams, leftLinePoints, rightLinePoints


@timeIt
def plotPredictionBoundry(warpedImg, leftLinePoints, rightLinePoints, margin):
    """
    Plot the detected lane lines and lane area over a given image

    @param warpedImg: (np.3darray) result of search plotted over
    @param leftLinePoints: (tuple) lane lines points X, Y (np.ndarray)
    @param rightLinePoints: (tuple) lane lines points X, Y (np.ndarray)
    @param margin: (int) lane line detection boundry width

    Returns
    =======
    boundryMask: (np.3darray) lane lines highlighted ploted image
    laneMask: (np.3darray) lane highlighted area
    """
    warped3DImg = np.dstack([warpedImg, warpedImg, warpedImg])
    boundryMask = np.zeros_like(warped3DImg)
    laneMask = np.zeros_like(warped3DImg)
    leftLine = list(zip(*leftLinePoints))
    righLine = list(zip(*rightLinePoints))
    leftLaneBoundry = np.array(leftLine, "int")
    rightLaneBoundry = np.flipud(np.array(righLine, "int"))
    laneBoundry = list(np.vstack([leftLaneBoundry, rightLaneBoundry]).reshape(1, -1, 2))
    # t1 = time()
    cv2.fillPoly(laneMask, laneBoundry, (255, 255, 0))
    # cv2.polylines(boundryMask, [np.array(leftLine, "int32")], False, (255, 255, 0), 33)
    # cv2.polylines(boundryMask, [np.array(righLine, "int32")], False, (255, 255, 0), 33)
    # t2 = time()
    # print("time >>>>>>>>>>>>>>>>>>>  ", t2 - t1)
    return laneMask


def predictXVal(y, params):
    """
    Predicts the x-axis value of a given y-axis value
    that belongs to a curve given its parameters

    @param y: y-axis value of the points to be estimated
    @param params: (tuple) of the lane line fitted parameters
    """
    a, b, c = params
    return a*y**2 + b*y + c


# @timeIt
def measureCurveRadius(y, params):
    """
    Computes the radius of curvature for a given (x, y) point
    that belogns to a lane curve.

    @param y: y-axis value of the points to find radius of curvature at
    @param params: (tuple) of a lane line fitted parameters
    """
    a, b, c = params
    x = predictXVal(y, params)
    dydx = 2*a*y + b
    d2ydx2 = 2*a
    r = ((1 + (dydx**2))**1.5)/d2ydx2
    return r


# @timeIt
def warped2BirdPoly(bgrFrame, points, width, height):
    """
    Warps a given image to a prespective bird view (top-plane)

    @param bgrFrame: (np.ndarray) BGR image
    @param points: ROI lane points to be warped from polygon to rectangle
    @param width: width of the target warped image
    @param height: height of the target warped image
    """
    warpedPoints = np.float32(points)
    birdPoints = np.float32([
        [points[-1][0], points[0][1]],
        [points[2][0], points[0][1]],
        points[2], points[-1]
    ])
    Matrix = cv2.getPerspectiveTransform(warpedPoints, birdPoints)
    warpedFrame = cv2.warpPerspective(bgrFrame, Matrix, (width, height))
    return warpedFrame, birdPoints


# @timeIt
def save2File(path, obj):
    """
    Saves a data object to a local binary file

    @param path: local path to store the object in
    @param obj: data object to be saved.
    """
    with open(path, "wb") as wf:
        pickle.dump(obj, wf)


# @timeIt
def loadFile(path):
    """
    Loads and Returns a data object from a local binary file

    @param path: local path to store the object in
    """
    with open(path, "rb") as rf:
        return pickle.load(rf)


@timeIt
def applyLaneMasks(srcFrame, birdPoint, roiPoints, *masks):
    M = cv2.getPerspectiveTransform(birdPoint, np.float32(roiPoints))
    laneMask = masks[-1]
    # boundryMask = cv2.warpPerspective(boundryMask, M, (1280, 720), cv2.INTER_LINEAR)
    laneMask = cv2.warpPerspective(laneMask, M, (1280, 720))
    displayedFrame = cv2.addWeighted(srcFrame, 1, laneMask, 0.25, 0)
    # displayedFrame = cv2.add(displayedFrame, boundryMask)
    return displayedFrame


def getLaneWidthVariance(leftYPoints, rightYPoints, leftParams, rightParams):
    # approximate lane width
    leftUpperX = predictXVal(0, leftParams)
    rightUpperX = predictXVal(0, rightParams)
    leftBottomX = predictXVal(720, leftParams)
    rightBottomX = predictXVal(720, rightParams)
    dist1 = np.abs(rightUpperX - leftUpperX)
    dist2 = np.abs(rightBottomX - leftBottomX)
    distRange = np.abs(dist1 - dist2)
    return distRange


def getSlopeVariance(leftYPoints, rightYPoints, leftParams, rightParams):
    a, b, c = leftParams
    left_dxdy = a*leftYPoints + b
    a, b, c = rightParams
    right_dxdy = a*leftYPoints + b
    slopeDiff = np.abs(right_dxdy - left_dxdy).mean()
    return slopeDifference


def linesInParallel(leftYPoints, rightYPoints, leftParams, rightParams):
    distRange = getLaneWidthVariance(leftYPoints, rightYPoints, leftParams, rightParams)
    slopeDifference = getSlopeVariance(leftYPoints, rightYPoints, leftParams, rightParams)

    distInRange = False if distRange > 150 else True
    slopeInRange = False if slopeDifference > 1 else True

    if distInRange and slopeInRange:
        return True
    return False


def paramsInRange(currLeftParams, currRightParams, prevLeftParams, prevRightParams):
    diff1 = currLeftParams - prevLeftParams
    diff2 = currRightParams - prevRightParams
    return False if (diff1 - diff2) > 1.0 else True
