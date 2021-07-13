import redis
from controller.controller_enum import DiscreteControls
from redis_io import REDIS_CONTROL_KEY, get_redis_instance


# TODO: Combine redis controller and redis I/O
class RedisController:
    def __init__(self, redis_inst, key):
        self.r = redis_inst
        self._key = key

    def run(self):
        cmd = self.r.get(self._key).decode('ascii')
        try:
            direction = DiscreteControls[cmd]
            return direction
        except KeyError as e:
            raise e

def default_redis_controller(redis_inst):
    return RedisController(redis_inst, REDIS_CONTROL_KEY)

if __name__ == "__main__":
    controller = default_redis_controller(get_redis_instance())
    controller.r.set(REDIS_CONTROL_KEY, "FWD")
    direction = controller.run()
    print("Expect FWD:")
    print(direction)

    controller.r.set(REDIS_CONTROL_KEY, "STOP")
    direction = controller.run()
    print("Expect STOP:")
    print(direction)

    print("Expect keyerror:")
    controller.r.set(REDIS_CONTROL_KEY, "fwd")
    direction = controller.run()
