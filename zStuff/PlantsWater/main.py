from machine import Pin
from time import sleep

inter_water_interval = 10   # in hours

pin_pump = Pin(22, Pin.OUT)
pin_led = Pin('LED', Pin.OUT)

def status_start():

    values = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]

    for i in values:
        pin_led.value(1)
        sleep(i)
        pin_led.value(0)
        sleep(i)

    for i in range(20):
        pin_led.value(1)
        sleep(0.05)
        pin_led.value(0)
        sleep(0.05)


def water_cycle(reps=5, dur=1, pause=1):

    for i in range(reps):

        pin_pump.value(1)
        sleep(dur)

        pin_pump.value(0)
        sleep(pause)


status_start()
pin_led.value(1)

while True:
    sleep(inter_water_interval)# * 60 * 60)   # in seconds
    water_cycle()
