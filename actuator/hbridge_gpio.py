import time
import RPi.GPIO as GPIO
from controller.controller_enum import DiscreteControls


# 4 motor H bridge
# Pin order LC, RC, LH, RH (actually not sure which is C or which is H)
FWD = (GPIO.LOW, GPIO.LOW, GPIO.HIGH, GPIO.HIGH)
REV = (GPIO.HIGH, GPIO.HIGH, GPIO.LOW, GPIO.LOW)
LEFT = (GPIO.HIGH, GPIO.LOW, GPIO.LOW, GPIO.HIGH)
RIGHT = (GPIO.LOW, GPIO.HIGH, GPIO.HIGH, GPIO.LOW)
STOP = (GPIO.LOW, GPIO.LOW, GPIO.LOW, GPIO.LOW)

class HBridgeGpio:
    def __init__(self):
        # use P1 header pin numbering convention
        GPIO.setmode(GPIO.BOARD)
        # TODO: Make this a config
        self._pins = (3, 8, 5, 10)
        self._pwm_pins = [32, 12]
        self._pwms = []
        for pin in self._pins:
            GPIO.setup(pin, GPIO.OUT)
        for pin in self._pwm_pins:
            GPIO.setup(pin, GPIO.OUT)
            # Setting PWM to 100, higher values made whiny noises and recommended values are [inconclusive](https://electronics.stackexchange.com/questions/309056/l298n-pwm-frequency)
            pi_pwm = GPIO.PWM(pin,100)

            # PWM ranges from 0 to 100
            pi_pwm.start(50.)

            self._pwms.append(pi_pwm)

    def set_gpio(self, gpio_settings):
        for gpio_out, pin in zip(gpio_settings, self._pins):
            GPIO.output(pin, gpio_out)

    def run(self, pca_input):
        if type(pca_input) == DiscreteControls:
            if pca_input == DiscreteControls.STOP:
                self.set_gpio(STOP)
            elif pca_input == DiscreteControls.LEFT:
                self.set_gpio(LEFT)
            elif pca_input == DiscreteControls.RIGHT:
                self.set_gpio(RIGHT)
            elif pca_input == DiscreteControls.FWD:
                self.set_gpio(FWD)
            elif pca_input == DiscreteControls.REV:
                self.set_gpio(REV)
            else:
                raise ValueError("Unrecognized DiscreteControls enum value")

