import enum
import numpy as np
from controller.controller_enum import DiscreteControls

# Clear threshold
CLEAR = 0.38
# Outputs the action
def create_action(mask):
    third_length = mask.shape[1] // 3
    left_available = np.sum(mask[:, :third_length])
    center_available = np.sum(mask[:, third_length:(2*third_length)])
    right_available = np.sum(mask[:, (2*third_length):])
    # mask is 0 to 255
    total = 255*np.product(mask.shape) // 3
    print(total)
    print(center_available)
    if center_available / total > CLEAR:
        return DiscreteControls.FWD
    elif np.maximum(left_available, right_available) / total > 0.75*CLEAR:
        return DiscreteControls.LEFT if left_available > right_available else DiscreteControls.RIGHT 
    else:
        return DiscreteControls.STOP
