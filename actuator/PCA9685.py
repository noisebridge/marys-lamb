import time
from controller.controller_enum import DiscreteControls

PCA_THROTTLE=0
PCA_STEERING=1
FULL_THROTTLE=395
COAST=380

# TODO : USE PCA_THROTTLE and PCA_STEERING in code.
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
        self._accel = False

    def set_pulse(self, pulse, channel=PCA_THROTTLE):
        try:
            self.pwm.set_pwm(channel, 0, int(pulse * self.pwm_scale))
        except:
            self.pwm.set_pwm(channel, 0, int(pulse * self.pwm_scale))

    '''
      Because keeping the vehicle at speed required fine tuning of the PWM output,
      the car simply alternates between acceleration and coasting, so as not to build up speed
    '''
    def intermittent_accel(self, channel=PCA_THROTTLE):
        if self._accel:
            self.set_pulse(COAST, channel)
            self._accel = False
        else:
            self.set_pulse(FULL_THROTTLE, channel)
            self._accel = True
    

    def run(self, pca_input):
        # Only supported case: DiscreteControls
        if type(pca_input) == DiscreteControls:
            if pca_input == DiscreteControls.STOP:
                self.set_pulse(0, PCA_THROTTLE)
                self.set_pulse(375, PCA_STEERING)
            elif pca_input == DiscreteControls.LEFT:
                self.intermittent_accel(PCA_THROTTLE)
                self.set_pulse(445, PCA_STEERING)
            elif pca_input == DiscreteControls.RIGHT:
                self.intermittent_accel(PCA_THROTTLE)
                self.set_pulse(305, PCA_STEERING)
            elif pca_input == DiscreteControls.FWD:
                self.intermittent_accel(PCA_THROTTLE)
                self.set_pulse(375, PCA_STEERING)
            else:
                raise ValueError("Unrecognized DiscreteControls enum value")
        else:
            raise TypeError("Unrecognized controller input")


if __name__ == "__main__":
    pca = PCA9685()
    # Start moving
    for i in range(4):
        pca.set_pulse(FULL_THROTTLE, channel=PCA_THROTTLE)
        print("Full throttle")
        time.sleep(0.5)
        pca.set_pulse(COAST, channel=PCA_THROTTLE)
        print("Coasting")
        time.sleep(0.5)
    pca.set_pulse(360, channel=PCA_THROTTLE)
    print("Stopped")
