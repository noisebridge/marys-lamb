from enum import Enum

'''
   Enum for high-level discrete vehicle controls. Currently a stand-in until more complex maneuvers / paths are implemented
'''
class DiscreteControls(Enum):
    STOP=0
    FWD=1
    LEFT=2
    RIGHT=3
    REV=4
