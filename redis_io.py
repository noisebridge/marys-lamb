import struct
import redis
import numpy as np


REDIS_PORT=6379
REDIS_CONTROL_KEY='redis_control'

# TODO: wrap all this in a class
redis_inst =  redis.Redis(host='localhost', port=REDIS_PORT, db=0)

# Assumes the image is np.uint8

def store_np_image(img_nparray, key):
   """Store given Numpy array 'img_nparray' in Redis under key 'key'"""
   if len(img_nparray.shape) == 2:
       h, w = img_nparray.shape
       shape = struct.pack('>II',h,w)
   else:
       h, w, d = img_nparray.shape
       shape = struct.pack('>III',h,w,d)
   encoded = shape + img_nparray.tobytes()

   # Store encoded data in Redis
   redis_inst.set(key,encoded)

def get_np_image_3d(redis_inst,n):
   """Retrieve Numpy array from Redis key 'n'"""
   encoded = r.get(n)
   h, w, d = struct.unpack('>III',encoded[:12])
   # Add slicing here, or else the array would differ from the original
   img = np.frombuffer(encoded[12:], dtype=np.uint8).reshape(h,w,d)
   return img

def get_redis_instance():
    return redis_inst

