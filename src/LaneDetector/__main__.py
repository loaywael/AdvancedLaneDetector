from LaneDetector import Detector
import matplotlib.pyplot as plt
import time
import cv2
import sys
import os


def main(arg_vars):
    if len(arg_vars) == 1:
        media_path = arg_vars[0] 
        
    else:
        print("ERROR: invalid argument list")
        print("|-----> please enter image, video path with supported arguments")
        return

    print("media", media_path)
    media_extension = os.path.splitext(media_path)[-1][1:]
    print("extension: ", media_extension)
    supported_imgs = ["jpg", "png", "jpeg"]
    supported_videos = ["mp4"]
    detector = Detector()               

    if media_extension in supported_imgs:
        img = plt.imread(media_path)
        # img = cv2.resize(img, None, fx=0.50, fy=0.50)
        ###############################################
        detectedImg = detector(img)
        ###############################################
        cv2.imshow("detection", detectedImg)
        cv2.waitKey(0)

    elif media_extension in supported_videos:
        cap = cv2.VideoCapture(media_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        while True:
            ret, frame = cap.read()
            if ret:
                t1 = time.time()
                ###############################################
                detectedImg = detector(frame)
                ###############################################
                t2 = time.time()
                cv2.putText(detectedImg, f"FPS: {int(1.0/(t2-t1))}", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow("detection", detectedImg)
                k = cv2.waitKey(1)
                if k & 0xFF == ord('q'):
                    break

    else:
        print("ERROR: this file is not supported!")


if __name__ == "__main__":
    main(sys.argv[1:])
    cv2.destroyAllWindows()
