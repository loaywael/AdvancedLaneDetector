import cv2
from threading import Thread, Event
from time import sleep


class GetFrame:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        self.grapped, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        th = Thread(target=self.get, args=())
        th.daemon = True
        th.start()
        return self
    
    def get(self):
        while not self.stopped:
            if not self.grapped:
                self.stop()
                self.stream.release()
                return
            else:
                (self.grapped, self.frame) = self.stream.read()
                sleep(0.001)
    
    def stop(self):
        self.stopped = True

    
class ShowFrame:
    def __init__(self, frame, windowName="Frame"):
        self.frame = frame
        self.stopped = False
        self.title = windowName

    def start(self):
        th = Thread(target=self.show, args=())
        th.daemon = True
        th.start()
        return self

    def show(self):
        while not self.stopped:
            cv2.imshow(self.title, self.frame)
            key = cv2.waitKey(1)
            # sleep(0.1)
            if key == ord('q'):
                self.stop()
                return
            elif key == ord('p'):
                cv2.waitKey(0)

    def stop(self):
        self.stopped = True



def delay(interval):
    def decorator(func):
        def wrapper(*args, **kwargs):
            stopped = threading.Event()
            
            def loop():
                while not stopped.wait(interval):
                    func(*a, **kwargs)
            th = Thread(target=loop)
            th.daemon = True
            th.start()
            return stopped
        return wrapper
    return decorator