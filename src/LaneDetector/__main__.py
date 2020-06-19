from LaneDetector import Detector, Pipeline
import time
import cv2
import sys
import os


def main(arg_vars):
    media_name = None
    if len(arg_vars) == 1:
        media_path = arg_vars[0] 
    elif len(arg_vars) == 2:
        media_path, media_name = arg_vars
    else:
        print("ERROR: invalid argument list")
        print("|-----> please enter image, video path with supported arguments")
        return

    print("media", media_path)
    media_extension = os.path.splitext(media_path)[-1][1:]
    print("extension: ", media_extension)
    supported_imgs = ["jpg", "png", "jpeg"]
    supported_videos = ["mp4"]
    detector = Pipeline((1280, 720, 3))               

    if media_extension in supported_imgs:
        img = cv2.imread(media_path)
        # img = cv2.resize(img, None, fx=0.50, fy=0.50)
        ###############################################
        detectedImg = detector(img)
        if media_name:
            print(media_name)
            cv2.imwrite("../assets/" + media_name, detectedImg)
        ###############################################
        cv2.imshow("detection", detectedImg)
        cv2.waitKey(0)

    elif media_extension in supported_videos:
        cap = cv2.VideoCapture(media_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if media_name:
            fourcc = 0x7634706d#cv2.VideoWriter_fourcc(*"MP4V")
            vidWriter = cv2.VideoWriter("../assets/" + media_name, fourcc, fps, (width, height))
        while True:
            ret, frame = cap.read()
            if ret:
                t1 = time.time()
                ###############################################
                detectedImg = detector(frame)
                ###############################################
                t2 = time.time()
                cv2.putText(detectedImg, f"FPS: {int(1.0/(t2-t1))}", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 15), 1, cv2.LINE_AA)
                cv2.imshow("detection", detectedImg)
                if media_name:
                    vidWriter.write(detectedImg)
                k = cv2.waitKey(1)
                if k & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        if media_name:
            vidWriter.release()

    else:
        print("ERROR: this file is not supported!")

if __name__ == "__main__":
    main(sys.argv[1:])
    cv2.destroyAllWindows()
    
