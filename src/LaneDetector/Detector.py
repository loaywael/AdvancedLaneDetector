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
    roiPoints : (np.2darray) the ROI trapezoid points representing the lane
    birdPoints : (np.2darray) rectangular transformation of the ROI points 2 bird view
    allRightParams : (list) keeps fitted right lane line params history for each frame
    allLeftParams : (list) keeps fitted right lane line params history for each frame
    path : (int) default save dir for detector configurations
    dstInMeter : (int) the x-distance between car head center and the predicted lane center
    laneMidPoint : (int) the predicted lane center x point
    carHeadMidPoint : (int) x-axis center point of the car head
    curveRadiusInMeter : (int) radius of the lane curvature
    roi2birdTransMtx : (np.ndarray) ROI points to bird points transformation matrix
    bird2roiTransMtx : (np.ndarray) bird points to ROI points transformation matrix
    frameShape : the input image/frame shape
    windowWidth : (int) sliding window width
    numWindows : (int) number of windows to scan the image
    windowHeight : (int) sliding window height dependent on number of windows and img height
    threshold : (int) minimum points to be considered as part of the lane
    steerWheel : (np.3darray) png image of the steering wheel to be overlayed
    camModel : (dict) of the undistorted and calibrated camera model
    dstCoeffs : (np.ndarray) distortion coefficients of the camera
    camMtx : (np.ndarray) Camera Matrix 

    Methods
    -------
    save2File(path, obj)
    loadFile(path)
    setWarpMatrices(self, roiPoints)
    initSteerWheelFG()
    getLaneMask(bgrFrame, minEdge, maxEdge, satThresh=85, blueThresh=190)
    warp2BirdPoly(self, bgrFrame, width, height)
    getInitialCenters(self, binaryImg)
    getLanePoints(self, binaryImg, visualize=False)
    predictLaneLines(self, binaryImg, margin, smoothThresh=5, visualize=False)
    fitLaneLines(leftLinePoints, rightLinePoints, order=2)
    genLinePoints(lineParams, lineLength)
    setLaneXcPoint(self, leftLineParams, rightLineParams, Yc=620, xPx2Mt=3.7/850)
    setCurveRadius(self, leftLinePoints, rightLinePoints, y, yPx2Mt=30/720, xPx2Mt=3.7/850)
    plotSteeringWheel(self, srcFrame)
    plotLaneMarker(self, srcFrame)
    plotPredictionBoundry(binaryImg, leftLinePoints, rightLinePoints)
    applyLaneMasks(self, srcFrame, mask)
    __call__(self, bgrFrame)
    """
    def __init__(self, roiPoints, frameShape, windowWidth=200, numWindows=13, threshold=33):
        """
        @param roiPoints: (dict) ROI lane roiPoints to be warped from polygon to rectangle
        @param frameShape: (tuple) shape of the expcted input frames
        @param windowWidth: (int) window width (horizontal distance between diagonals)
        @param numWindows: (int) number of windows allowed to be stacked on top of each other
        @param threshold: (int) minimum points to be considered as part of the lane
        """
        self.roiPoints = []
        self.birdPoints = []
        self.allLeftParams = []
        self.allRightParams = []
        self.path = "configs/"
        self.dstInMeter = 0
        self.laneMidPoint = None
        self.frameShape = (*frameShape[-2::-1], frameShape[-1])
        self.carHeadMidPoint = self.frameShape[1] // 2
        self.curveRadiusInMeter = 0
        self.roi2birdTransMtx = None
        self.bird2roiTransMtx = None
        self.windowWidth = windowWidth
        self.numWindows = numWindows
        self.windowHeight = np.int(self.frameShape[0] // numWindows)
        self.threshold = threshold
        self.steerWheel = Detector.initSteerWheelFG("../assets/")
        self.camModel = Detector.loadFile(self.path+"camCalibMatCoeffs.sav")
        self.dstCoeffs = self.camModel["dstCoeffs"]
        self.camMtx = self.camModel["camMtx"]
        self.setWarpMatrices(roiPoints)
        
    
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

    def setWarpMatrices(self, roiPoints):
        """
        Initializing ROI points and Bird points transformation matrices

        @param roiPoints: (dict) ROI lane roiPoints to be warped from polygon to rectangle
        """
        self.roiPoints = np.float32([
            roiPoints["topLeft"],
            roiPoints["bottomLeft"],
            roiPoints["topRight"],
            roiPoints["bottomRight"]
        ])
        self.birdPoints = np.float32([[200, 0], [200, 720], [1000, 0], [1000, 720]])
        # self.birdPoints = np.float32([[200, 0], [100, 680], [1180, 0], [1180, 680]])
        self.roi2birdTransMtx = cv2.getPerspectiveTransform(self.roiPoints, self.birdPoints)
        self.bird2roiTransMtx = cv2.getPerspectiveTransform(self.birdPoints, self.roiPoints)

    @staticmethod
    def initSteerWheelFG(assetPath=""):
        """
        Initializing steering wheel overlay mask

        @param assetPath: (str) steering wheel path folder

        Returns
        =======
        fgSteerWheel: (np.3darray) steering wheel foreground mask
        """
        steerWheel = cv2.imread(assetPath+"steering_wheel.png")
        steerWheel = cv2.resize(steerWheel, None, fx=0.25, fy=0.25)
        steerWheelMask = cv2.cvtColor(steerWheel, cv2.COLOR_BGR2GRAY)
        steerWheelMask = cv2.medianBlur(steerWheelMask, 3)
        steerWheelMask = cv2.morphologyEx(steerWheelMask, cv2.MORPH_OPEN, kernel=np.ones((3, 3)))
        steerWheelMask = cv2.morphologyEx(steerWheelMask, cv2.MORPH_DILATE, kernel=np.ones((3, 3)))
        steerWheelMask = cv2.morphologyEx(steerWheelMask, cv2.MORPH_CLOSE, kernel=np.ones((3, 3)))

        _, steerWheelMask = cv2.threshold(steerWheelMask, 245, 255, cv2.THRESH_BINARY_INV)
        fgSteerWheel = cv2.bitwise_and(steerWheel, steerWheel, mask=steerWheelMask)
        return fgSteerWheel

    @staticmethod
    # @profiling.timer
    def getLaneMask(bgrFrame, minEdge, maxEdge, satThresh=85, blueThresh=190):
        """
        Applies color bluring, thresholding and edge detection on a bgr image
        Returns lanes detected in a binary mask

        @param bgrFrame: (np.3darray) BGR image
        @param minEdge: minimum theshold for edge detector
        @param maxEdge: maximum theshold for edge detector
        @param satThresh: saturation channel threshold any value above is 255 else is 0
        @param blueThresh: blue channel threshold any value above is 255 else is 0

        Returns
        =======
        outMask : (np.2darray) binary mask of the thresholded lanes 
        """
        grayImg = cv2.cvtColor(bgrFrame, cv2.COLOR_BGR2GRAY)
        hlsImg = cv2.cvtColor(bgrFrame, cv2.COLOR_BGR2HLS)
        edgesMask = cv2.Canny(grayImg, minEdge, maxEdge)
        _, blueMask = cv2.threshold(bgrFrame[:, :, 0], blueThresh, 255, cv2.THRESH_BINARY)
        _, satMask = cv2.threshold(hlsImg[:, :, 2], satThresh, 255, cv2.THRESH_BINARY)
        colorMask = cv2.bitwise_or(satMask, blueMask)
        outMask = cv2.bitwise_or(colorMask, edgesMask)
        outMask = cv2.morphologyEx(outMask, cv2.MORPH_CLOSE, kernel=np.ones((3, 3)))
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
    def getLanePoints(self, binaryImg, visualize=False):
        """
        Applies blind search for the lane points (white pixels on black background)
        using the sliding window algorithm starting from the centers of the histogram peaks

        @param binaryImg: (np.2darray) bird view binary image of the lane roi
        @params visualize: (bool) returns the scanning result overlayed on the input image

        Returns
        =======
        scanedImg: (np.3darray) sliding window overlay over the input image
        leftLinePoints: (tuple) left lane line points  (X, Y np.ndarray)
        rightLinePoints: (tuple) right lane line points (X, Y np.ndarray)

        """
        leftLanePixelsIds = []
        rightLanePixelsIds = []
        noneZeroIds = binaryImg.nonzero()
        noneZeroXIds = np.array(noneZeroIds[1])
        noneZeroYIds = np.array(noneZeroIds[0])
        leftCenter, rightCenter, _ = self.getInitialCenters(binaryImg)
        if visualize:
            scanedImg = np.dstack([binaryImg]*3)

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
            
            if visualize:
                cv2.rectangle(scanedImg, leftLinePt1, leftLinePt2, (0, 255, 0), 3)
                cv2.rectangle(scanedImg, rightLinePt1, rightLinePt2, (0, 255, 0), 3)
                

        leftLanePixelsIds = np.sum(np.array(leftLanePixelsIds), axis=0).astype("bool")
        rightLanePixelsIds = np.sum(np.array(rightLanePixelsIds), axis=0).astype("bool")
        leftXPoints = noneZeroXIds[leftLanePixelsIds]
        leftYPoints = noneZeroYIds[leftLanePixelsIds]
        rightXPoints = noneZeroXIds[rightLanePixelsIds]
        rightYPoints = noneZeroYIds[rightLanePixelsIds]

        if visualize:
            scanedImg[leftYPoints, leftXPoints] = [255, 0, 0]
            scanedImg[rightYPoints, rightXPoints] = [0, 0, 255]
            return scanedImg, (leftXPoints, leftYPoints), (rightXPoints, rightYPoints)
        return (leftXPoints, leftYPoints), (rightXPoints, rightYPoints)

    def predictLaneLines(self, binaryImg, margin, smoothThresh=5, visualize=False):
        """
        Predicts lane line in a new frame based on previous detection from blind search
        
        @param binaryImg: (np.2darray) bird view binary image of the lane roi
        @param margin: (int) width of the lane line
        @param smoothThresh: (int) number of last frames to average its params
        @params visualize: (bool) returns the scanning result overlayed on the input image


        Returns
        =======
        scanedImg: (np.3darray) lane line highlights overlay over the input image
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

        if visualize:
            scanedImg = np.dstack([binaryImg]*3)
            scanedImg[leftYPoints, leftXPoints] = [255, 0, 0]
            scanedImg[rightYPoints, rightXPoints] = [0, 0, 255]
            return scanedImg, (leftXPoints, leftYPoints), (rightXPoints, rightYPoints)

        return (leftXPoints, leftYPoints), (rightXPoints, rightYPoints)

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
        @param lineParams: fitted parameters of the line to map to
        @param lineLength: Height of lane line

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
    def setLaneXcPoint(self, leftLineParams, rightLineParams, Yc=620, xPx2Mt=3.7/850):
        """
        Estimates the lane X-axis center point

        @param leftLineParams: (tuple) left lane line fitted params corresponding with the current frame
        @param rightLineParams: (tuple) right lane line fitted params corresponding with the current frame
        @param Yc: (int) the marker height to evaluate center at
        @param xPx2Mt: (float) X-axis scaling to convert from pixels to meter dimensions
        """
        a, b, c = leftLineParams
        leftLineXc = a*Yc**2 + b*Yc + c
        a, b, c = rightLineParams
        rightLineXc = a*Yc**2 + b*Yc + c
        self.laneMidPoint = int(((rightLineXc - leftLineXc) // 2) + leftLineXc)
        self.dstInMeter = (self.laneMidPoint - self.carHeadMidPoint) * xPx2Mt

    # @profiling.timer
    def setCurveRadius(self, leftLinePoints, rightLinePoints, y, yPx2Mt=30/720, xPx2Mt=3.7/850):
        """
        Estimates the lane radius of curvature

        @param leftLinePoints: (tuple) left lane line fitted params corresponding with the current frame
        @param rightLinePoints: (tuple) right lane line fitted params corresponding with the current frame
        @param y: (int) the marker height to evaluate center at
        @param yPx2Mt: (float) Y-axis scaling to convert from pixels to meter dimensions
        @param xPx2Mt: (float) X-axis scaling to convert from pixels to meter dimensions
        """
        realLeftParams = np.polyfit(leftLinePoints[1]*yPx2Mt, leftLinePoints[0]*xPx2Mt, 2)
        realRightParams = np.polyfit(rightLinePoints[1]*yPx2Mt, rightLinePoints[0]*xPx2Mt, 2)
        a, b, c = realLeftParams
        leftCurveRadius = ((1 + (2*a*y*yPx2Mt+b)**2)**1.5) / np.abs(2*a)
        a, b, c = realRightParams
        rightCurveRadius = ((1 + (2*a*y*yPx2Mt+b)**2)**1.5) / np.abs(2*a)
        self.curveRadiusInMeter = np.mean([rightCurveRadius, leftCurveRadius], axis=0)

    def plotSteeringWheel(self, srcFrame):
        """ 
        Overlays steering wheel over a given frame 
        
        @param: (np.3darray) source frame

        Returns
        =======
        srcFrame: (np.3darray) the same source image with the overlay
        """
        distance = (self.laneMidPoint - self.carHeadMidPoint)
        angle = distance // -2
        steerWheelHeight, steerWheelWidth = self.steerWheel.shape[:2]
        center = steerWheelWidth//2, steerWheelHeight//2
        rotMtx = cv2.getRotationMatrix2D(center, angle, 1.0)
        fgSteerWheel = cv2.warpAffine(self.steerWheel, rotMtx, (steerWheelWidth, steerWheelHeight))
        roiPatch = srcFrame[150:150+steerWheelHeight, 100:100+steerWheelWidth]
        roiPatch[fgSteerWheel > 1] = 0
        steerWheel = cv2.add(fgSteerWheel, roiPatch)
        srcFrame[150:150+steerWheelHeight, 100:100+steerWheelWidth] = steerWheel
        return srcFrame

    # @profiling.timer
    def plotLaneMarker(self, srcFrame):
        """
        Overlays the estimated lane center markers over the source frame

        @param srcFrame: (np.3darray) source frame
        """
        carCenterPt1 = self.carHeadMidPoint, 670
        carCenterPt2 = self.carHeadMidPoint, 670 - 75
        laneCenterPt1 = self.laneMidPoint, 670
        laneCenterPt2 = self.laneMidPoint, 670 - 50
        leftCenterMarkerPt1 = (laneCenterPt1[0]-180, laneCenterPt1[1])
        leftCenterMarkerPt2 = (laneCenterPt2[0]-180, laneCenterPt2[1])
        rightCenterMarkerPt1 = (laneCenterPt1[0]+180, laneCenterPt1[1])
        rightCenterMarkerPt2 = (laneCenterPt2[0]+180, laneCenterPt2[1])

        cv2.arrowedLine(srcFrame, (self.carHeadMidPoint, 645), (self.laneMidPoint-5, 645), (0, 0, 255), 3)
        cv2.line(srcFrame, carCenterPt1, carCenterPt2, (255, 0, 0), 3)
        cv2.line(srcFrame, laneCenterPt1, laneCenterPt2, (0, 255, 0), 3)
        cv2.line(srcFrame, leftCenterMarkerPt1, leftCenterMarkerPt2, (0, 0, 255), 3)
        cv2.line(srcFrame, rightCenterMarkerPt1, rightCenterMarkerPt2, (0, 0, 255), 3)

    @staticmethod
    # @profiling.timer
    def plotPredictionBoundry(binaryImg, leftLinePoints, rightLinePoints):
        """
        Plot the detected lane lines and lane area over a given image

        @param binaryImg: (np.2darray) result of search plotted over
        @param leftLinePoints: (tuple) lane lines points X, Y (np.ndarray)
        @param rightLinePoints: (tuple) lane lines points X, Y (np.ndarray)

        Returns
        =======
        boundryMask: (np.3darray) lane lines highlighted ploted image
        laneMask: (np.3darray) lane highlighted area
        """
        warped3DImg = np.dstack([binaryImg, binaryImg, binaryImg])
        laneMask = np.zeros_like(warped3DImg)
        leftLine = list(zip(*leftLinePoints))
        righLine = list(zip(*rightLinePoints))
        leftLaneBoundry = np.array(leftLine, "int")
        rightLaneBoundry = np.flipud(np.array(righLine, "int"))
        laneBoundry = list(np.vstack([leftLaneBoundry, rightLaneBoundry]).reshape(1, -1, 2))
        cv2.fillPoly(laneMask, laneBoundry, (0, 255, 0))
        # cv2.rectangle(laneMask, (0, 0), (250, 100), (125, 125, 125), -1)
        return laneMask

    # @profiling.timer
    def applyLaneMasks(self, srcFrame, mask):
        """
        Applies detected lane mask over source image to be displayed

        @param srcFrame: (np.3darray) source image to be displayed
        @masks: other masks to apply over the source image 

        Returns
        =======
        displayedFrame: (np.3darray) detected frame to be displayed
        """
        cv2.putText(srcFrame, f"Radius of curvature: {abs(int(self.curveRadiusInMeter))} (m)", (25, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 15), 1, cv2.LINE_AA)
        cv2.putText(srcFrame, f"Distance from camera center: {abs(int(self.dstInMeter*100))} (cm)", (25, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 15), 1, cv2.LINE_AA)
        laneMask = cv2.warpPerspective(mask, self.bird2roiTransMtx, (1280, 720))
        displayedFrame = cv2.addWeighted(srcFrame, 1, laneMask, 0.25, 0)
        self.plotLaneMarker(displayedFrame)
        self.plotSteeringWheel(displayedFrame)
        return displayedFrame

    # @profiling.timer
    def __call__(self, bgrFrame):
        """
        Applies the detector functionality over a given frame

        @param bgrFrame: (np.3darray) image source to apply detection over

        Returns
        =======
        displayedFrame: (np.3darray) detected frame to be displayed
        """
        undist_frame = cv2.undistort(bgrFrame, self.camMtx, self.dstCoeffs, None, self.camMtx)
        birdFrame = cv2.warpPerspective(undist_frame, self.roi2birdTransMtx, (1280, 720))
        binaryLanes = Detector.getLaneMask(birdFrame, 33, 175)

        if len(self.allRightParams) < 15:
            leftLinePoints, rightLinePoints = self.getLanePoints(binaryLanes)
        else:
            leftLinePoints, rightLinePoints = self.predictLaneLines(binaryLanes, 100, smoothThresh=15)
     
        lane_max_height = binaryLanes.shape[0]
        leftLineParams, rightLineParams = Detector.fitLaneLines(leftLinePoints, rightLinePoints, order=2)    
        leftLinePoints = Detector.genLinePoints(leftLineParams, lane_max_height)
        rightLinePoints = Detector.genLinePoints(rightLineParams, lane_max_height)
        self.setLaneXcPoint(leftLineParams, rightLineParams, 620)
        self.setCurveRadius(leftLinePoints, rightLinePoints, 620)
        laneMask = Detector.plotPredictionBoundry(binaryLanes, leftLinePoints, rightLinePoints)
        displayedFrame = self.applyLaneMasks(bgrFrame, laneMask)
     
        self.allLeftParams.append(leftLineParams)
        self.allRightParams.append(rightLineParams)
        return displayedFrame