# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import redis
import time
import cv2
import io

# TODO: Add params
class PiCamSensor:
    def __init__(self, sensor_params):
        # initialize the camera and grab a reference to the raw camera capture
        self._camera = PiCamera()
        self._camera.resolution = (320, 240)
        #self._camera.start_preview()
        # allow the camera to warmup
        time.sleep(2.0)

    def sense(self):
        self._rawCapture = PiRGBArray(self._camera)
        # grab an image from the camera
        self._camera.capture(self._rawCapture, format="bgr", use_video_port=True)
        return self._rawCapture.array

def default_camera_sensor():
    sensor_params = None
    return PiCamSensor(sensor_params)

