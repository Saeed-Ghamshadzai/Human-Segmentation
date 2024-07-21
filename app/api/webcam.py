import cv2
from threading import Thread
import time

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.grabbed, self.frame = self.video.read()
        self.running = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
    
    def stop(self):
        self.running = False
        self.thread.join()
        self.video.release()

    def update(self):
        while self.running:
            self.grabbed, self.frame = self.video.read()
            time.sleep(0.03)

    def get_frame(self):
        _, jpeg = cv2.imencode('.jpg', self.frame)
        return jpeg.tobytes()

    def get_segmented_frame(self):
        # Placeholder for segmentation logic
        segmented_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)  # Dummy segmentation
        _, jpeg = cv2.imencode('.jpg', segmented_frame)
        return jpeg.tobytes()