import time
from controller.controller_enum import DiscreteControls

PCA_THROTTLE=1
PCA_STEERING=1
FULL_THROTTLE=390

class PCA9685:
    ''' 
    PWM motor controler using PCA9685 boards. 
    This is used for most RC Cars
    '''
    def __init__(self, address=0x40, frequency=60, busnum=None, init_delay=1.0):

        self.default_freq = 60
        self.pwm_scale = frequency / self.default_freq

        import Adafruit_PCA9685
        # Initialise the PCA9685 using the default address (0x40).
        if busnum is not None:
            from Adafruit_GPIO import I2C
            # replace the get_bus function with our own
            def get_bus():
                return busnum
            I2C.get_default_bus = get_bus
        try: 
            self.pwm = Adafruit_PCA9685.PCA9685(address=address)
            self.pwm.set_pwm_freq(frequency)
        except OSError as e:
            raise e
        self.set_pulse(360, 0)
        self.set_pulse(360, 1)
        time.sleep(init_delay) # "Tamiya TBLE-02" makes a little leap otherwise

    def set_pulse(self, pulse, channel=PCA_THROTTLE):
        try:
            self.pwm.set_pwm(channel, 0, int(pulse * self.pwm_scale))
        except:
            self.pwm.set_pwm(channel, 0, int(pulse * self.pwm_scale))

    def run(self, pca_input):
        # Only supported case: DiscreteControls
        if type(pca_input) == DiscreteControls:
            if pca_input == DiscreteControls.STOP:
                self.set_pulse(0, 0)
                self.set_pulse(375, 1)
            elif pca_input == DiscreteControls.LEFT:
                self.set_pulse(FULL_THROTTLE, 0)
                self.set_pulse(445, 1)
            elif pca_input == DiscreteControls.RIGHT:
                self.set_pulse(FULL_THROTTLE, 0)
                self.set_pulse(305, 1)
            elif pca_input == DiscreteControls.FWD:
                self.set_pulse(FULL_THROTTLE, 0)
                self.set_pulse(375, 1)
            else:
                raise ValueError("Unrecognized DiscreteControls enum value")
        else:
            raise TypeError("Unrecognized controller input")


if __name__ == "__main__":
    pca = PCA9685()
    pca.set_pulse(370, channel=0)
    time.sleep(1)
    pca.set_pulse(410, channel=0)
    time.sleep(1)
    # Neutral steering 
    pca.set_pulse(350, channel=1)
    time.sleep(1)
    # Left
    pca.set_pulse(400, channel=1)
    time.sleep(1)
    # Right
    pca.set_pulse(300, channel=1)
    time.sleep(1)
    # Can't figure out backwards PWM ?
    #pca.set_pulse()
    #time.sleep(1)
    pca.set_pulse(370, channel=0)
    pca.set_pulse(350, channel=1)
    time.sleep(1)
