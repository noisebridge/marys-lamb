import enum

class FishAction(enum.Enum):
   LEFT=0 
   STRAIGHT=1
   RIGHT=2

# Outputs the action
def create_action(mask):
    third_length = mask.shape[1] // 3
    left_vote = np.sum(mask[:, :third_length])
    center_vote = np.sum(mask[:, third_length:(2*third_length)])
    right_vote = np.sum(mask[:, (2*third_length):])
    direction = FishAction(np.argmax([left_vote, center_vote, right_vote]))
    return direction
