import cv2
from threading import Thread
from .models import Cameras

class VideoCamera(object):
    def __init__(self, camera):
        cameraIP = camera.ip
        if cameraIP == '8.8.8.8':
            self.video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            reqStr = 'rtsp://username:password@'+cameraIP+':554/user='+camera.username+'_password='+camera.password+'_channel=channel_number_stream=1.sdp'
            self.video = cv2.VideoCapture(reqStr)
        (self.grabbed, self.frame) = self.video.read()
        Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()