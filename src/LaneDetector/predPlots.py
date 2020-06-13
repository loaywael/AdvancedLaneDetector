def plotPredictionTesting(warpedImg, leftLinePoints, rightLinePoints, margin):
    """
    Plot the detected lane boundries over a given image

    @param warpedImg: (np.3darray) result of search plotted over
    @param leftLinePoints: (tuple) lane lines points X, Y (np.ndarray)
    @param rightLinePoints: (tuple) lane lines points X, Y (np.ndarray)
    @param margin: (int) lane line detection boundry width
    """
    boundryMask = np.zeros_like(warpedImg)
    leftLineLeftMargin = leftLinePoints[0] - margin, leftLinePoints[1]
    leftLineRightMargin = leftLinePoints[0] + margin, leftLinePoints[1]
    rightLineLeftMargin = rightLinePoints[0] - margin, rightLinePoints[1]
    rightLineRightMargin = rightLinePoints[0] + margin, rightLinePoints[1]
    leftLineLeftMargin = np.array(list(zip(*leftLineLeftMargin)), "int")
    leftLineRightMargin = np.flipud(np.array(list(zip(*leftLineRightMargin)), "int"))
    leftBoundry = list(np.vstack([leftLineLeftMargin, leftLineRightMargin]).reshape(1, -1, 2))
    cv2.fillPoly(boundryMask, leftBoundry, (255, 255, 0))
    rightLineLeftMargin = np.array(list(zip(*rightLineLeftMargin)), "int")
    rightLineRightMargin = np.flipud(np.array(list(zip(*rightLineRightMargin)), "int"))
    rightBoundry = list(np.vstack([rightLineLeftMargin, rightLineRightMargin]).reshape(1, -1, 2))
    cv2.fillPoly(boundryMask, rightBoundry, (255, 255, 0))
    outImg = cv2.addWeighted(warpedImg, 1, boundryMask, 0.2, 0)
    return outImg

    # @timeIt

    def predictLaneLinesTest(warpedFrame, linesParams, margin):
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
        rightLineBoundryIds = (noneZeroXIds > rightLineLBoundry) & (
            noneZeroXIds < rightLineRBoundry)

        leftLineBoundryX = noneZeroXIds[leftLineBoundryIds]
        leftLineBoundryY = noneZeroYIds[leftLineBoundryIds]
        rightLineBoundryX = noneZeroXIds[rightLineBoundryIds]
        rightLineBoundryY = noneZeroYIds[rightLineBoundryIds]
        outImg = np.dstack([warpedFrame, warpedFrame, warpedFrame])
        outImg[leftLineBoundryY, leftLineBoundryX] = [255, 0, 0]
        outImg[rightLineBoundryY, rightLineBoundryX] = [0, 0, 255]
        linesParams, leftLinePoints, rightLinePoints = fitLaneLines(
            (leftLineBoundryX, leftLineBoundryY),
            (rightLineBoundryX, rightLineBoundryY),
            warpedFrame.shape[0]-1
        )
        return outImg, linesParams, leftLinePoints, rightLinePoints


def getLanePointsTests(warpedFrame, nWindows, windowWidth, pixelsThresh):
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
    outImg[leftYPoints, leftXPoints] = [255, 0, 0]
    outImg[rightYPoints, rightXPoints] = [0, 0, 255]
    return outImg, (leftXPoints, leftYPoints), (rightXPoints, rightYPoints)
