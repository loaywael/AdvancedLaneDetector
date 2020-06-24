from LaneDetector import Detector, Pipeline
import threading
import argparse
import time
import cv2
import sys
import os


def main(arg_vars):
    roiPoints = {
        # "topLeft"     : [ 568, 460],
        # "topRight"    : [ 717, 460],
        # "bottomRight" : [ 1043, 680],
        # "bottomLeft"  : [260, 680]
        #-----------------------------
        "topLeft"     : [ 560, 460],
        "topRight"    : [ 720, 460],
        "bottomRight" : [ 1080, 680],
        "bottomLeft"  : [200, 680]
    }

    media_path = arg_vars.path
    print(">>> media", media_path)
    save_path = "../assets/"
    if arg_vars.save:
        output_name = arg_vars.name if arg_vars.name else "output_video.mp4"
        print(">>> output folder: ", save_path)

    media_extension = os.path.splitext(media_path)[-1][1:]
    print("extension: ", media_extension)
    supported_imgs = ["jpg", "png", "jpeg"]
    supported_videos = ["mp4"]
    # detector = Pipeline(roiPoints, (1280, 720, 3))               
    detector = Detector(roiPoints, (1280, 720, 3))               

    if media_extension in supported_imgs:
        img = cv2.imread(media_path)
        #############################
        detectedImg = detector(img)
        #############################
        if arg_vars.save:
            cv2.imwrite(save_path+output_name, detectedImg)
        cv2.imshow("detection", detectedImg)
        cv2.waitKey(0)

    elif media_extension in supported_videos:
        cap = cv2.VideoCapture(media_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        recording = False

        if arg_vars.save:
            fourcc = 0x7634706d #cv2.VideoWriter_fourcc(*"MP4V")
            vidWriter = cv2.VideoWriter(save_path+output_name, fourcc, fps, (width, height))
        while True:
            ret, frame = cap.read()
            if ret:
                t1 = time.perf_counter()
                #############################
                detectedImg = detector(frame)
                #############################
                t2 = time.perf_counter()
                cv2.putText(detectedImg, f"FPS: {int(1.0/(t2-t1))}", (1100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 15), 1, cv2.LINE_AA)
                cv2.imshow("detection", detectedImg)
                k = cv2.waitKey(1)
                if k & 0xFF == ord('q'):
                    break
                elif k & 0xFF == ord('p'):
                    recording = False
                    k = cv2.waitKey(0)
                    if k & 0xFF == ord("s"):
                        cv2.imwrite(save_path+"screenshot_1.jpg", detectedImg)
                elif k & 0xFF == ord('r'): 
                    if arg_vars.save:
                        recording = True 
                    else:
                        print("ERROR: -s keyword argument should be given")
                if recording:
                    vidWriter.write(detectedImg)
            else:
                break
        cap.release()
        if arg_vars.save:
            vidWriter.release()
    else:
        print("ERROR: this file is not supported!")

if __name__ == "__main__":
    video_parser = argparse.ArgumentParser()
    # --------------------------------------------------------------
    video_parser.add_argument(
        "-p", "--path", required=True, type=str, metavar="",
        help="media path of the driving scene mp4/jpg"
    )
    video_parser.add_argument(
        "-s", "--save", required=False, action="store_true",
        help="save the output rendered video"
    )
    video_parser.add_argument(
        "-n", "--name", required=False, type=str, metavar="",
        help="name of the rendered video to be saved"
    )
    # --------------------------------------------------------------
    args = video_parser.parse_args()
    main(args)
    cv2.destroyAllWindows()
    
