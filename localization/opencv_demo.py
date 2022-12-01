import cv2
import yaml
import numpy as np

def convert_K_from_yaml(yaml_K):
    K_list = [[float(kii.strip()) for kii in ki.split(',')] for ki in yaml_K]
    return np.array(K_list)

def get_distortion_coefficients(yaml_dist):
    return np.array([float(d) for d in yaml_dist[0].split(',')])

# TODO : Camera calibration steps
# Source : https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

# object_points : Points in 3D space
# image_points : Points in 2D (image) space
# K : intrinsic camera matrix
# dist : Distortion coefficients
# flags=SOLVEPNP_IPPE says we will use the IPPE solver
# cv2.solvePnP(object_points, image_points, K, dist, flags=cv2.SOLVEPNP_IPPE
if __name__ == '__main__':
    yaml_str  = "K: \n- 1, 0, 3\n- 0, 5, 6\n- 0, 0, 1\ndist: \n- 1,2,3,4,5"
    K_yaml = yaml.safe_load(yaml_str)['K']
    dist_yaml = yaml.safe_load(yaml_str)['K']
    K = convert_K_from_yaml(K_yaml)
    dist = get_distortion_coefficients(dist_yaml)
    object_points = # TODO
    imge_points = # TODO 
    cv2.solvePnP(object_points, image_points, K, dist, flags=cv2.SOLVEPNP_IPPE)

    print(K)
    print(dist)
