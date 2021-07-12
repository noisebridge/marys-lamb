import redis
from controller.controller_enum import DiscreteControls

REDIS_PORT=6379
REDIS_KEY='redis_control'


class RedisController:
    def __init__(self, port, key):
        self.r = redis.Redis(host='localhost', port=port, db=0)
        self._key = key

    def run(self):
        cmd = self.r.get(self._key).decode('ascii')
        try:
            direction = DiscreteControls[cmd]
            return direction
        except KeyError as e:
            raise e

def default_redis_controller():
    return RedisController(REDIS_PORT, REDIS_KEY)

if __name__ == "__main__":
    controller = default_redis_controller()
    controller.r.set(REDIS_KEY, "FWD")
    direction = controller.run()
    print("Expect FWD:")
    print(direction)

    controller.r.set(REDIS_KEY, "STOP")
    direction = controller.run()
    print("Expect STOP:")
    print(direction)

    print("Expect keyerror:")
    controller.r.set(REDIS_KEY, "fwd")
    direction = controller.run()
