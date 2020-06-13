import cv2
import numpy as np
import pickle
import os


if __name__ == "__main__":
    colorBag = []
    files = os.listdir()
    for file in files:
        if file.endswith(".jpg"):
            color = None

            def pickBGRColor(event, x, y, flags, params):
                global image, color, colorBag
                image = params
                if event == cv2.EVENT_LBUTTONDBLCLK:
                    color = image[y, x]
                    # cv2.circle(image, (x, y), 2, (255, 255, 255), 2)
                    colorBag.append(color)

            bgrImg = cv2.imread(file, cv2.IMREAD_COLOR)
            hsvImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2HSV)
            cv2.namedWindow(file)
            cv2.setMouseCallback(file, pickBGRColor, bgrImg)
            while cv2.waitKey(10) != ord('q'):
                if color is not None:
                    print()
                    hsvColor = cv2.cvtColor(np.array(color, "uint8", ndmin=3), cv2.COLOR_BGR2HSV)
                    h, s, v = hsvColor.ravel()
                    lowerColor = np.array([h - 30, s - 30, v])
                    upperColor = np.array([h + 30, 255, 255])
                    print(colorBag)
                    maskedColor = cv2.inRange(hsvImg, lowerColor, upperColor)
                    colorPicked = cv2.bitwise_and(bgrImg, bgrImg, mask=maskedColor)
                    cv2.imshow(file+"-mask", colorPicked)
                cv2.imshow(file, bgrImg)
            cv2.destroyAllWindows()
    meanRGBColor = np.mean(np.array(colorBag, "uint8").reshape(-1, 3), axis=0)
    with open("yellow", "wb") as wf:
        pickle.dump(meanRGBColor, wf)

with open("yellow", "rb") as rf:
    white = pickle.load(rf)
    print(white)
