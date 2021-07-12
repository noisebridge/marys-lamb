import time
from controller.redis_controller import * 
from actuator.PCA9685 import *

controller = default_redis_controller()
actuator = PCA9685()

while True:
    try:
        direction = controller.run()
        actuator.run(direction)
    except KeyError as e:
        print(e)
    time.sleep(0.01)
