import numpy as np
import pickle
import cv2

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



def drawRoIPoly(img, points, size=9):
    """
    Draws the Lane ROI polygon that visualize the points to be warped

    @param img: (np.ndarray) BGR image
    @points: list of points of the Lane ROI
    """
    # for pt in points:
    points = np.float32([points[0], points[2], points[3], points[1]])
    #     cv2.circle(img, tuple(pt), size, (0, 0, 255), -1)
    cv2.polylines(img, [points.astype(np.int32)], True, (0, 255, 0), 2)
    return img




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
    return slopeDiff


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
