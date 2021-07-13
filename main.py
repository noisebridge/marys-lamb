import time
from controller.redis_controller import * 
from actuator.PCA9685 import *
from sensor.picam import default_camera_sensor, PiCamSensor
from redis_io import get_redis_instance, store_np_image

redis_inst = get_redis_instance()
controller = default_redis_controller(redis_inst)
# Only run actuator if present
try:
    actuator = PCA9685()
except OSError as e:
    actuator = None
picam = default_camera_sensor()

while True:
    img = picam.sense()
    store_np_image(img, "img")
    try:
        direction = controller.run()
        if actuator:
            actuator.run(direction)
    except KeyError as e:
        print(e)
    time.sleep(0.01)
