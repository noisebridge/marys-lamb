import time
from controller.redis_controller import * 
from actuator.PCA9685 import *
from actuator.hbridge_gpio import HBridgeGpio
from sensor.picam import default_camera_sensor, PiCamSensor
from planning.fishbrain import create_action
from redis_io import get_redis_instance, store_np_image
from vision.unet import UNetWrapper, convert_keras_to_tf_lite
import asyncio
import cv2
import numpy as np

def bgr_to_rgb(img):
    return img[:, :, ::-1]

def overlay_image(img, mask):
    img_overlay = img
    img_overlay[:,:,1] = img[:,:,1]/2 + np.maximum(img[:,:,1], 255*mask)/2
    return img_overlay

def main():
    redis_inst = get_redis_instance()
    controller = default_redis_controller(redis_inst)
    unet = UNetWrapper(init_model=False)
    #convert_keras_to_tf_lite('/home/pi/keras-model', '/home/pi/trained_model_pi_converted.tflite')
    #unet.generate_model("/home/pi/tf14-best.h5")
    #unet.load_tf_lite_model("/home/pi/mobile_net.tflite")
    #unet.load_tf_lite_model("/home/pi/trained_model_10_09.tflite")
    unet.load_tf_lite_model("/home/pi/trained_model_10_11.tflite")
    # unet.save_keras_model()
    # Only run actuator if present
    try:
        actuator = HBridgeGpio()
    except OSError as e:
        actuator = None
    picam = default_camera_sensor()

    # TODO: Make this asynchronous
    i=0
    img=None
    img_small=None
    mask=None
    directon = DiscreteControls.STOP
    while True:
        start_time = time.time()
        img = picam.sense()
        img = picam.sense()
        img_small = cv2.resize(img, (128, 128))
        print("time taken for picam.sense", time.time() - start_time)
        mask = unet.predict(img_small)
        print("time taken for mask", time.time() - start_time)
        img_overlay = overlay_image(img_small, mask)
        store_np_image(img_overlay, "img")
        print("time taken to store image", time.time() - start_time)
        if mask is not None:
            direction = create_action(mask)
        try:
            controller_direction = controller.run()
            # For always control:
            if True:
                direction = controller_direction
            #if controller_direction == DiscreteControls.STOP:
            #    direction = controller_direction
            
            if actuator:
                actuator.run(direction)

        except KeyError as e:
            print(e)
        print(direction)
        print("Total time:", time.time() - start_time)
        i+=1

main()
