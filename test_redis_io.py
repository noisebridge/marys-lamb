
import cv2
from redis_io import store_np_image, get_np_image_3d

# Replace this with an available image
a0 = cv2.imread("/home/pi/Camera/output.jpg")

# Redis connection
r = redis.Redis(host='localhost', port=6379, db=0)

# Store array a0 in Redis under name 'a0array'
store_np_image(r,a0,'a0array')

# Retrieve from Redis
a1 = get_np_image_3d(r,'a0array')

np.testing.assert_array_equal(a0,a1)

