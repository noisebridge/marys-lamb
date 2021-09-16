import time
from controller.redis_controller import * 
from actuator.PCA9685 import *
from sensor.picam import default_camera_sensor, PiCamSensor
from planning.fishbrain import create_action
from redis_io import get_redis_instance, store_np_image
from vision.unet import UNetWrapper
import asyncio
import cv2
import numpy as np

def bgr_to_rgb(img):
    return img[:, :, ::-1]

def overlay_image(img, mask):
    img_overlay = cv2.resize(img, (128, 128))
    img_overlay[:,:,1] = img_overlay[:,:,1]/2 + np.maximum(img_overlay[:,:,1], mask)/2
    return img_overlay

def main():
    redis_inst = get_redis_instance()
    controller = default_redis_controller(redis_inst)
    unet = UNetWrapper()
    unet.generate_model("/home/pi/tf14-best.h5")
    # unet.load_tf_lite_model()
    # unet.save_keras_model()
    # Only run actuator if present
    try:
        actuator = PCA9685()
    except OSError as e:
        actuator = None
    picam = default_camera_sensor()

    # TODO: Make this asynchronous
    i=0
    img=None
    mask=None
    directon = DiscreteControls.STOP
    while True:
        start_time = time.time()
        if i%2==0 or img is None:
            img = picam.sense()
            print("time taken for picam.sense", time.time() - start_time)
        else:
            mask = unet.predict(img)
            print("time taken for mask", time.time() - start_time)
            img_overlay = overlay_image(img, mask)
            store_np_image(img_overlay, "img")
            print("time taken to store image", time.time() - start_time)
        if mask is not None:
            direction = create_action(mask)
        try:
            controller_direction = controller.run()
            # For always control:
            #if True:
            #    direction = controller_direction
            if controller_direction == DiscreteControls.STOP:
                direction = controller_direction
            
            if actuator:
                actuator.run(direction)

        except KeyError as e:
            print(e)
        print(direction)
        i+=1

main()
