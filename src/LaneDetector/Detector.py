from decimal import Decimal
import numpy as np
import pickle
import cv2
import os


class Detector:
    """ 
    C

    Attributes
    ----------

    Methods
    -------

    """
    def __init__(self):
        self.allRightParams = []
        self.allLeftParams = []
        self.path = "configs/"
        self.roiPoints = Detector.loadFile(self.path+"roiPoly.sav")
        self.camModel = Detector.loadFile(self.path+"camCalibMatCoeffs.sav")
        self.dstCoeffs = self.camModel["dstCoeffs"]
        self.camMtx = self.camModel["camMtx"]
    
    @staticmethod
    def save2File(path, obj):
        """
        Saves a data object to a local binary file

        @param path: local path to store the object in
        @param obj: data object to be saved.
        """
        with open(path, "wb") as wf:
            pickle.dump(obj, wf)

    @staticmethod
    def loadFile(path):
        """
        Loads and Returns a data object from a local binary file

        @param path: local path to store the object in
        """
        with open(path, "rb") as rf:
            return pickle.load(rf)

    @staticmethod
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

    def warp2BirdPoly(self, bgrFrame, points, width, height):
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
        # boundryMask = np.zeros_like(warped3DImg)
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

    @staticmethod
    def applyLaneMasks(srcFrame, birdPoint, roiPoints, *masks):
        M = cv2.getPerspectiveTransform(birdPoint, np.float32(roiPoints))
        laneMask = masks[-1]
        # boundryMask = cv2.warpPerspective(boundryMask, M, (1280, 720), cv2.INTER_LINEAR)
        laneMask = cv2.warpPerspective(laneMask, M, (1280, 720))
        displayedFrame = cv2.addWeighted(srcFrame, 1, laneMask, 0.25, 0)
        # displayedFrame = cv2.add(displayedFrame, boundryMask)
        return displayedFrame

    def __call__(self, img):
        undist_frame = cv2.undistort(img, self.camMtx, self.dstCoeffs, None, self.camMtx)
        displayedFrame = undist_frame.copy()
        binaryLanes = Detector.getLaneMask(displayedFrame, 33, 175, 110, 30, 170)
        birdFrame, birdPoint = self.warp2BirdPoly(binaryLanes, points, 1280, 720)
        return binaryLanes