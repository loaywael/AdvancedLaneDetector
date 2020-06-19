from decimal import Decimal
from LaneDetector.utils import profiling
import numpy as np
import imutils
import pickle
import cv2
import os

 
class Detector:
    """ 
    Sliding Window Curved Lane Detector

    Attributes
    ----------
    path : (int) default save dir for detector configurations
    allRightParams : (list) keeps fitted right lane line params history for each frame
    allLeftParams : (list) keeps fitted right lane line params history for each frame
    threshold: (int) minimum points to be considered as part of the lane
    windowHeight : (int) sliding window height dependent on number of windows and img height
    numWindows : (int) number of windows to scan the image
    windowWidth : (int) sliding window width
    roiPoints : (np.2darray) the ROI trapezoid points representing the lane
    birdPoints : (np.2darray) rectangular transformation of the ROI points 2 bird view
    camModel : (dict) of the undistorted and calibrated camera model
    dstCoeffs : (np.ndarray) distortion coefficients of the camera
    camMtx : (np.ndarray) Camera Matrix 

    Methods
    -------
    save2File(path, obj)
    loadFile(path)
    trans2BirdPoint(roiPoints)
    getLaneMask( minEdge, maxEdge, satThresh, hueThresh, redThresh)
    warp2BirdPoly(self, bgrFrame, width, height)
    getInitialCenters(self, binaryImg)
    getLanePoints(self, binaryImg, pixelsThresh, visualize=False)
    fitLaneLines(leftLinePoints, rightLinePoints, lineLength, order=2)
    predictLaneLines(self, binaryImg, margin, smoothThresh=5)
    plotPredictionBoundry(warpedImg, leftLinePoints, rightLinePoints, margin)
    applyLaneMasks(self, srcFrame, *masks)
    __call__(self, img)
    """
    def __init__(self, frameShape, windowWidth=175, numWindows=11, threshold=55):
        """
        @param windowWidth: (int) window width (horizontal distance between diagonals)
        @param windowWidth: (int) number of windows allowed to be stacked on top of each other
        @param threshold: (int) minimum points to be considered as part of the lane
        """
        self.frameShape = (*frameShape[-2::-1], frameShape[-1])
        self.windowWidth = windowWidth
        self.numWindows = numWindows
        self.windowHeight = np.int(self.frameShape[0] // numWindows)
        self.threshold = threshold
        self.path = "configs/"
        self.allRightParams = []
        self.allLeftParams = []
        self.laneMidPoint = None
        self.carHeadMidPoint = self.frameShape[1] // 2
        self.roiPoints = Detector.loadFile(self.path+"roiPoly.sav")
        self.roiPoints = np.float32(self.roiPoints)
        self.birdPoints = Detector.trans2BirdPoint(self.roiPoints)
        self.roi2birdTransMtx = cv2.getPerspectiveTransform(self.roiPoints, self.birdPoints)
        self.bird2roiTransMtx = cv2.getPerspectiveTransform(self.birdPoints, self.roiPoints)
        self.steerWheel = Detector.initSteerWheelFG()
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
    def trans2BirdPoint(roiPoints):
        """
        Map ROI points to bird view points

        @param roiPoints: ROI lane roiPoints to be warped from polygon to rectangle
        """
        birdPoints = np.array([
            [roiPoints[-1][0], roiPoints[0][1]],
            [roiPoints[2][0], roiPoints[0][1]],
            roiPoints[2], roiPoints[-1]
        ])
        return birdPoints

    @staticmethod
    def initSteerWheelFG():
        """
        """
        steerWheel = cv2.imread("../assets/steering_wheel.png")
        steerWheel = cv2.resize(steerWheel, None, fx=0.25, fy=0.25)
        steerWheelMask = cv2.cvtColor(steerWheel, cv2.COLOR_BGR2GRAY)
        steerWheelMask = cv2.medianBlur(steerWheelMask, 3)
        steerWheelMask = cv2.morphologyEx(steerWheelMask, cv2.MORPH_DILATE, kernel=np.ones((3, 3)))
        _, steerWheelMask = cv2.threshold(steerWheelMask, 245, 255, cv2.THRESH_BINARY_INV)
        fgSteerWheel = cv2.bitwise_and(steerWheel, steerWheel, mask=steerWheelMask)
        return fgSteerWheel

    @staticmethod
    # @profiling.timer
    def getLaneMask(bgrFrame, minEdge, maxEdge, satThresh, hueThresh, redThresh):
        """
        Applies color bluring, thresholding and edge detection on a bgr image
        Returns lanes detected in a binary mask

        @param bgrImg: (np.ndarray) BGR image
        @param minEdge: minimum theshold for edge detector
        @param maxEdge: maximum theshold for edge detector
        @param satThresh: saturation channel threshold any value above is 255 else is 0
        @param hueThresh: hue channel threshold any value above is 255 else is 0
        @param redThresh: red, and green channel threshold any value above is 255 else is 0

        Returns
        =======
        outMask : (np.2darray) binary mask of the thresholded lanes 
        """
        grayImg = cv2.cvtColor(bgrFrame, cv2.COLOR_BGR2GRAY)
        hlsImg = cv2.cvtColor(bgrFrame, cv2.COLOR_BGR2HLS)
        edgesMask = cv2.Canny(grayImg, minEdge, maxEdge)
        _, satMask = cv2.threshold(hlsImg[:, :, 2], 175, 255, cv2.THRESH_BINARY)
        _, hueMask = cv2.threshold(hlsImg[:, :, 2], 33, 255, cv2.THRESH_BINARY_INV)
        _, redMask = cv2.threshold(bgrFrame[:, :, 2], 170, 255, cv2.THRESH_BINARY)
        _, greenMask = cv2.threshold(bgrFrame[:, :, 1], 170, 255, cv2.THRESH_BINARY)
        hueSatMask = cv2.bitwise_and(hueMask, satMask)
        redGreenMask = cv2.bitwise_and(redMask, greenMask)
        colorMask = cv2.bitwise_or(redGreenMask, hueSatMask)
        outMask = cv2.bitwise_or(colorMask, edgesMask)
        return outMask

    def getInitialCenters(self, binaryImg):
        """
        Estimates the begining of the lane lines and locate its center points
        by computing peaks in histogram and finding the y-axis from the first
        white pixel and the x-axis from the bottom half and taking its median point
        
        @param binaryImg: bird view binary image of the lane roi

        Returns
        =======
        leftCenter: (int) X-axis position of the left lane line
        rightCenter: (int) X-axis position of the right lane line
        xAxisHisto: (np.ndarray) histogram of pixels of the given image along X-axis

        """
        xAxisHisto = np.sum(binaryImg[self.frameShape[0]//2:], axis=0)
        xMidPoint = xAxisHisto.shape[0]//2    # histogram x-axis midpoint
        leftXcPoint = np.argmax(xAxisHisto[:xMidPoint])
        # abs id is found by adding the midpoint to start from 0 point
        rightXcPoint = np.argmax(xAxisHisto[xMidPoint:]) + xMidPoint
        rightCenter = (rightXcPoint, self.frameShape[0] - self.windowHeight//2)
        leftCenter = (leftXcPoint, self.frameShape[0] - self.windowHeight//2)
        return leftCenter, rightCenter, xAxisHisto
    
    # @profiling.timer
    def getLanePoints(self, binaryImg):
        """
        Applies blind search for the lane points (white pixels on black background)
        using the sliding window algorithm starting from the centers of the histogram peaks

        @param binaryImg: (np.2darray) bird view binary image of the lane roi
        
        Returns
        =======
        leftLinePoints: (tuple) left lane line points  (X, Y np.ndarray)
        rightLinePoints: (tuple) right lane line points (X, Y np.ndarray)

        """
        leftLanePixelsIds = []
        rightLanePixelsIds = []
        noneZeroIds = binaryImg.nonzero()
        noneZeroXIds = np.array(noneZeroIds[1])
        noneZeroYIds = np.array(noneZeroIds[0])
        leftCenter, rightCenter, _ = self.getInitialCenters(binaryImg)

        for i in range(1, self.numWindows+1):
            leftLinePt1 = (leftCenter[0]-self.windowWidth//2, leftCenter[1]-self.windowHeight//2)
            leftLinePt2 = (leftCenter[0]+self.windowWidth//2, leftCenter[1]+self.windowHeight//2)
            rightLinePt1 = (rightCenter[0]-self.windowWidth//2, rightCenter[1]-self.windowHeight//2)
            rightLinePt2 = (rightCenter[0]+self.windowWidth//2, rightCenter[1]+self.windowHeight//2)

            leftWindowXIds = (noneZeroXIds > leftLinePt1[0]) & (noneZeroXIds < leftLinePt2[0])
            leftWindowYIds = (noneZeroYIds > leftLinePt1[1]) & (noneZeroYIds < leftLinePt2[1])
            leftWindowPoints = leftWindowXIds & leftWindowYIds
            rightWindowXIds = (noneZeroXIds > rightLinePt1[0]) & (noneZeroXIds < rightLinePt2[0])
            rightWindowYIds = (noneZeroYIds > rightLinePt1[1]) & (noneZeroYIds < rightLinePt2[1])
            rightWindowPoints = rightWindowXIds & rightWindowYIds

            leftLanePixelsIds.append(leftWindowPoints)
            rightLanePixelsIds.append(rightWindowPoints)

            leftCenter = leftCenter[0], leftCenter[1] - self.windowHeight
            if leftWindowPoints.sum() > self.threshold:
                lXc = np.int(noneZeroXIds[leftWindowPoints].mean())
                leftCenter = (lXc, leftCenter[1])

            rightCenter = rightCenter[0], rightCenter[1] - self.windowHeight
            if rightWindowPoints.sum() > self.threshold:
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
    # @profiling.timer
    def fitLaneLines(leftLinePoints, rightLinePoints, order=2):
        """
        Fits lines to a given lane lines

        @param leftLinePoints: (tuple) lane lines points X, Y (np.ndarray)
        @param rightLinePoints: (tuple) lane lines points X, Y (np.ndarray)
        @param order: degree of polynomial equation that fits the lane lines

        Returns
        =======
        leftLineParams: (np.ndarray) left line fitted params
        rightLineParams: (np.ndarray) right line fitted params
        """
        leftXPoints, leftYPoints = leftLinePoints
        rightXPoints, rightYPoints = rightLinePoints
        leftLineParams = np.polyfit(leftYPoints, leftXPoints, order)
        rightLineParams = np.polyfit(rightYPoints, rightXPoints, order)
        return (leftLineParams, rightLineParams)

    @staticmethod
    def genLinePoints(lineParams, lineLength):
        """  
        @param lineLength: Height of lane line
        @param lineParams: fitted parameters of the line to map to

        Returns
        =======
        lineXVals: (np.ndarray) X values of the fitted line
        lineYVals: (np.ndarray) Y values of the fitted line
        """
        a, b, c = lineParams
        lineYVals = np.linspace(0, lineLength, lineLength)
        lineXVals = a*lineYVals**2 + b*lineYVals + c
        return  lineXVals, lineYVals

    # @profiling.timer
    def predictLaneLines(self, binaryImg, margin, smoothThresh=5):
        """
        Predicts lane line in a new frame based on previous detection from blind search
        
        @param binaryImg: (np.2darray) bird view binary image of the lane roi
        @param margin: (int) width of the lane line
        @param smoothThresh: (int) number of last frames to average its params

        Returns
        =======
        leftLinePoints: (tuple) left lane line points  (X, Y np.ndarray)
        rightLinePoints: (tuple) right lane line points (X, Y np.ndarray)
        """
        rightLineParams = np.mean(self.allRightParams[-smoothThresh::], axis=0)
        leftLineParams = np.mean(self.allLeftParams[-smoothThresh::], axis=0)

        noneZeroIds = binaryImg.nonzero()
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

        leftXPoints = noneZeroXIds[leftLineBoundryIds]
        leftYPoints = noneZeroYIds[leftLineBoundryIds]
        rightXPoints = noneZeroXIds[rightLineBoundryIds]
        rightYPoints = noneZeroYIds[rightLineBoundryIds]
        
        return (leftXPoints, leftYPoints), (rightXPoints, rightYPoints)
    
    def setLaneXcPoint(self, leftLineParams, rightLineParams, Yc=670):
        """

        """
        a, b, c = leftLineParams
        leftLineXc = a*Yc**2 + b*Yc + c
        a, b, c = rightLineParams
        rightLineXc = a*Yc**2 + b*Yc + c
        self.laneMidPoint = int(((rightLineXc - leftLineXc) // 2) + leftLineXc)

    # @profiling.timer
    def plotSteeringWheel(self, srcImg):
        angle = (self.laneMidPoint - self.carHeadMidPoint) // -3
        steerWheelHeight, steerWheelWidth = self.steerWheel.shape[:2]
        center = steerWheelWidth//2, steerWheelHeight//2
        rotMtx = cv2.getRotationMatrix2D(center, angle, 1.0)
        fgSteerWheel = cv2.warpAffine(self.steerWheel, rotMtx, (steerWheelWidth, steerWheelHeight))
        roiPatch = srcImg[50:50+steerWheelHeight, 550:550+steerWheelWidth]
        roiPatch[fgSteerWheel > 1] = 0
        steerWheel = cv2.add(fgSteerWheel, roiPatch)
        srcImg[50:50+steerWheelHeight, 550:550+steerWheelWidth] = steerWheel
        return srcImg

    @staticmethod
    # @profiling.timer
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
        laneMask = np.zeros_like(warped3DImg)
        leftLine = list(zip(*leftLinePoints))
        righLine = list(zip(*rightLinePoints))
        leftLaneBoundry = np.array(leftLine, "int")
        rightLaneBoundry = np.flipud(np.array(righLine, "int"))
        laneBoundry = list(np.vstack([leftLaneBoundry, rightLaneBoundry]).reshape(1, -1, 2))
        cv2.fillPoly(laneMask, laneBoundry, (0, 255, 0))
        
        return laneMask

    # @profiling.timer
    def plotLaneMarker(self, frame):
        carCenterPt1 = self.carHeadMidPoint, 670
        carCenterPt2 = self.carHeadMidPoint, 670 - 100
        laneCenterPt1 = self.laneMidPoint, 670
        laneCenterPt2 = self.laneMidPoint, 670 - 50
        leftCenterMarkerPt1 = (laneCenterPt1[0]-180, laneCenterPt1[1])
        leftCenterMarkerPt2 = (laneCenterPt2[0]-180, laneCenterPt2[1])
        rightCenterMarkerPt1 = (laneCenterPt1[0]+180, laneCenterPt1[1])
        rightCenterMarkerPt2 = (laneCenterPt2[0]+180, laneCenterPt2[1])

        cv2.line(frame, carCenterPt1, carCenterPt2, (255, 0, 0), 5)
        cv2.line(frame, laneCenterPt1, laneCenterPt2, (0, 255, 0), 5)
        cv2.line(frame, leftCenterMarkerPt1, leftCenterMarkerPt2, (0, 0, 255), 3)
        cv2.line(frame, rightCenterMarkerPt1, rightCenterMarkerPt2, (0, 0, 255), 3)

    # @profiling.timer
    def applyLaneMasks(self, srcFrame, *masks):
        """
        Applies detected lane mask over source image to be displayed

        @param srcFrame: (np.3darray) source image to be displayed
        @masks: other masks to apply over the source image 

        Returns
        =======
        displayedFrame: (np.3darray) detected frame to be displayed
        """
        # laneMask = masks[-1]
        # boundryMask = cv2.warpPerspective(boundryMask, M, (1280, 720), cv2.INTER_LINEAR)
        for mask in masks:
            laneMask = cv2.warpPerspective(mask, self.bird2roiTransMtx, (1280, 720))
            displayedFrame = cv2.addWeighted(srcFrame, 1, laneMask, 0.25, 0)
        self.plotLaneMarker(displayedFrame)
        self.plotSteeringWheel(displayedFrame)
        return displayedFrame

    # @profiling.timer
    def __call__(self, bgrImg):
        """
        Applies the detector functionality over a given frame

        @param bgrImg: (np.3darray) image source to apply detection over

        Returns
        =======
        displayedFrame: (np.3darray) detected frame to be displayed
        """
        undist_frame = cv2.undistort(bgrImg, self.camMtx, self.dstCoeffs, None, self.camMtx)
        displayedFrame = undist_frame.copy()
        binaryLanes = Detector.getLaneMask(displayedFrame, 33, 175, 110, 30, 170)
        birdFrame = cv2.warpPerspective(binaryLanes, self.roi2birdTransMtx, (1280, 720))

        if len(self.allRightParams) > 2:
            leftLinePoints, rightLinePoints = self.predictLaneLines(birdFrame, margin=100)

        else:
            leftLinePoints, rightLinePoints = self.getLanePoints(birdFrame)

        lane_max_height = birdFrame.shape[0]
        leftLineParams, rightLineParams = Detector.fitLaneLines(leftLinePoints, rightLinePoints, order=2)    
        leftLinePoints = Detector.genLinePoints(leftLineParams, lane_max_height)
        rightLinePoints = Detector.genLinePoints(rightLineParams, lane_max_height)
        self.setLaneXcPoint(leftLineParams, rightLineParams, 670)
        laneMask = Detector.plotPredictionBoundry(birdFrame, leftLinePoints, rightLinePoints, margin=100)
        displayedFrame = self.applyLaneMasks(displayedFrame, laneMask)
        self.allLeftParams.append(leftLineParams)
        self.allRightParams.append(rightLineParams)
        return displayedFrame