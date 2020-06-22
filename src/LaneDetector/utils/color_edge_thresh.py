import cv2
import pickle
import numpy as np




def getLaneMask(bgrFrame):
    """
    Applies color bluring, thresholding and edge detection on a bgr image
    Returns lanes detected in a binary mask
    """
    # grayImg = cv2.cvtColor(bgrFrame, cv2.COLOR_BGR2GRAY)
    hlsImg = cv2.cvtColor(bgrFrame, cv2.COLOR_BGR2HLS)
    # hlsImg = cv2.cvtColor(bgrFrame, cv2.COLOR_BGR2HLS)
    # edgesMask = cv2.Canny(grayImg, minEdge, maxEdge)
    # --------------------------------------------------------------------------
    _, satMask = cv2.threshold(hlsImg[:, :, 2], 155, 255, cv2.THRESH_BINARY)
    # _, hueMask = cv2.threshold(hlsImg[:, :, 2], 75, 255, cv2.THRESH_BINARY_INV)
    # _, redMask = cv2.threshold(bgrFrame[:, :, 0], 155, 255, cv2.THRESH_BINARY)
    # _, greenMask = cv2.threshold(bgrFrame[:, :, 1], 175, 255, cv2.THRESH_BINARY)
    # --------------------------------------------------------------------------
    # hueSatMask = cv2.bitwise_and(hueMask, satMask)
    # redGreenMask = cv2.bitwise_and(redMask, greenMask)
    # laneMask = cv2.bitwise_or(redGreenMask, hueSatMask)
    # --------------------------------------------------------------------------
    # outMask = cv2.bitwise_or(colorMask, edgesMask)
    # outMask = cv2.morphologyEx(outMask, cv2.MORPH_DILATE, kernel=np.ones((3, 3)))
    # outMask = cv2.morphologyEx(outMask, cv2.MORPH_CLOSE, kernel=np.ones((3, 3)))

    return satMask


path = "../data/driving_datasets/"
videoPath = path + "project_video.mp4"

cap = cv2.VideoCapture(videoPath)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

with open("configs/camCalibMatCoeffs.sav", "rb") as rf:
    camModel = pickle.load(rf)
dstCoeffs = camModel["dstCoeffs"]
camMtx = camModel["camMtx"]
        

key = None
while True:
    key = cv2.waitKey(10)
    ret, frame = cap.read() 
    frame = cv2.warpPerspective(frame, self.roi2birdTransMtx, (1280, 720))
    if ret and cap.isOpened() and key != ord('q'):
        mask = getLaneMask(frame)
        cv2.imshow("mask", mask)

    elif key == ord('p'):
        cv2.waitKey(0)

    else:
        break
