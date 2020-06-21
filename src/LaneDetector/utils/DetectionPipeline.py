from LaneDetector import Detector
from LaneDetector.utils.toolbox import drawRoIPoly
import numpy as np
import cv2


class Pipeline(Detector):
    def __init__(self, *args, **kwargs):
        super(Pipeline, self).__init__(*args, **kwargs)
        self.rightPoints = None
        self.rightParams = None
        self.leftPoints = None
        self.leftParams = None

    def colorfyLaneBoundry(self, coloredWarpedLanes):
        srcImg = coloredWarpedLanes.copy()
        self.leftPoints = self.genLinePoints(self.leftParams, 720)
        self.rightPoints = self.genLinePoints(self.rightParams, 720)
        dstImg3 = self.plotFittedLines(srcImg, self.leftPoints, self.rightPoints)
        srcImg = coloredWarpedLanes.copy()
        dstImg4 = self.plotPredictionBoundry(srcImg[:, :, 0], self.leftPoints, self.rightPoints, 50)
        dstImg4 = cv2.addWeighted(srcImg, 1, dstImg4, 0.3, 0)
        # dstImg4 = srcImg.copy()
        return dstImg3, dstImg4

    @staticmethod
    def getPipeLineBoard(srcImg1, srcImg2, srcImg3, srcImg4, sizeFactor=0.5):
        upperRow = np.hstack([srcImg1, srcImg2])
        upperRow = cv2.resize(upperRow, None, fx=sizeFactor, fy=sizeFactor)
        lowerRow = np.hstack([srcImg3, srcImg4])
        lowerRow = cv2.resize(lowerRow, None, fx=sizeFactor, fy=sizeFactor)
        fullBoard = np.vstack([upperRow, lowerRow])
        return fullBoard

    @staticmethod
    def plotFittedLines(warpedImg, leftLinePoints, rightLinePoints, margin=50):
        """  """
        # warpedImg = np.dstack([warpedImg, warpedImg, warpedImg])
        boundryMask = np.zeros_like(warpedImg).astype(np.uint8)
        leftLineLeftMargin = leftLinePoints[0] - margin, leftLinePoints[1]
        leftLineRightMargin = leftLinePoints[0] + margin, leftLinePoints[1]
        rightLineLeftMargin = rightLinePoints[0] - margin, rightLinePoints[1]
        rightLineRightMargin = rightLinePoints[0] + margin, rightLinePoints[1]

        leftLineLeftMargin = np.array(list(zip(*leftLineLeftMargin)), "int")
        leftLineRightMargin = np.flipud(np.array(list(zip(*leftLineRightMargin)), "int"))
        leftBoundry = list(np.vstack([leftLineLeftMargin, leftLineRightMargin]).reshape(1, -1, 2))

        cv2.fillPoly(boundryMask, leftBoundry, (0, 255, 0))
        rightLineLeftMargin = np.array(list(zip(*rightLineLeftMargin)), "int")
        rightLineRightMargin = np.flipud(np.array(list(zip(*rightLineRightMargin)), "int"))
        rightBoundry = list(np.vstack([rightLineLeftMargin, rightLineRightMargin]).reshape(1, -1, 2))
        cv2.fillPoly(boundryMask, rightBoundry, (0, 255, 0))
        outImg = cv2.addWeighted(warpedImg, 1, boundryMask, 0.3, 0)
        return outImg


    def smoothParams(self):
        self.allLeftParams.append(self.leftParams)
        self.allRightParams.append(self.rightParams)
        if len(self.allRightParams) > 5:
            self.leftParams = np.mean(self.allLeftParams[-5::], axis=0)
            self.rightParams = np.mean(self.allRightParams[-5::], axis=0)


    def __call__(self, X):
        undist_frame = cv2.undistort(X, self.camMtx, self.dstCoeffs, None, self.camMtx)
        displayedFrame = undist_frame.copy()
        dstImg1 = cv2.warpPerspective(displayedFrame, self.roi2birdTransMtx, (1280, 720))
        drawRoIPoly(displayedFrame, self.roiPoints)

        binaryLanes = Detector.getLaneMask(dstImg1, 33, 175, 110, 30, 170)
        dstImg2, self.leftPoints, self.rightPoints = self.getLanePoints(binaryLanes, visualize=True)
        self.leftParams, self.rightParams = self.fitLaneLines(self.leftPoints, self.rightPoints)

        dstImg3, dstImg4 = self.colorfyLaneBoundry(dstImg2)
        fullBoard = Pipeline.getPipeLineBoard(dstImg1, dstImg2, dstImg3, dstImg4, 0.5)
  
        return fullBoard
