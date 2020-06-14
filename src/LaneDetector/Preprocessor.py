class Preprocessor:
    def __init__()
        pass

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def getSobelMag(absGrads, minThresh, maxThresh):
        """
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

    @staticmethod
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

    def __call__(self, img):
        binaryLanes = getLaneMask(frame, 33, 175, 110, 30, 170)
