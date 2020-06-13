import os
import cv2
import numpy as np
from time import time
from detection_utils import updatePoints, drawRoIPoly, timeIt
from detection_utils import save2File, loadFile, getLaneMask
from detection_utils import warped2BirdPoly, fitLaneLines, applyLaneMasks
from detection_utils import getLanePoints, predictLaneLines
from detection_utils import linesInParallel, paramsInRange
from detection_utils import measureCurveRadius, plotPredictionBoundry


path = "./driving_datasets/"
videoPath = path + "project_video.mp4"

cap = cv2.VideoCapture(videoPath)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
camModel = loadFile("camCalibMatCoeffs")
camMtx = camModel["camMtx"]
dstCoeffs = camModel["dstCoeffs"]

print(width, height)
i = 0
key = None
points = []
allRightParams = []
allLeftParams = []
while cap.isOpened() and key != ord('q'):
    key = cv2.waitKey(1)
    ret, frame = cap.read()    # reads bgr frame
    if ret:
        t1 = time()
        frame = cv2.undistort(frame, camMtx, dstCoeffs, None, camMtx)
        displayedFrame = frame.copy()
        if os.path.exists("./roiPoly"):
            points = loadFile("./roiPoly")
        if points is not None:
            binaryLanes = getLaneMask(frame, 33, 200, 110, 30, 165)
            # binaryLanes = getLaneMask(frame, 33, 175, 110, 30, 170)
            birdFrame, birdPoint = warped2BirdPoly(binaryLanes, points, 1280, 720)
            if allLeftParams and allRightParams and i > 5:
                if is
                # print("up")
                linesParams = np.mean(allLeftParams[-5::],
                                      axis=0), np.mean(allRightParams[-5::], axis=0)
                linesParams, leftLine, rightLine = predictLaneLines(
                    birdFrame, linesParams, margin=100)

            else:
                # print("down")
                leftLanePoints, rightLanePoints = getLanePoints(birdFrame, 11, 175, 55, )
                linesParams, leftLinePoints, rightLinePoints = fitLaneLines(
                    leftLanePoints, rightLanePoints, birdFrame.shape[0], order=2
                )
            allLeftParams.append(linesParams[0])
            allRightParams.append(linesParams[1])
            laneMask = plotPredictionBoundry(
                birdFrame, leftLinePoints, rightLinePoints, margin=100)
            displayedFrame = applyLaneMasks(
                displayedFrame, birdPoint, points, laneMask)

        cv2.imshow("result", displayedFrame)
        t2 = time()
        print(f"time to execute: {t2 - t1}")
        i += 1

cap.release()
cv2.destroyAllWindows()
