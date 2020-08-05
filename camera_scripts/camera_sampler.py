from picamera import PiCamera
from time import sleep
import time
import picamera
import numpy as np

INPUT_SIZE = (480,480)
FRAME_RATE = 10

def take_image():
    with picamera.PiCamera() as camera:
        camera.resolution = INPUT_SIZE
        camera.framerate = FRAME_RATE
        time.sleep(2)
        output = np.empty(INPUT_SIZE + (3,), dtype=np.uint8)
        print(output.shape)
        camera.capture(output,'rgb')
    return output


if __name__ == '__main__':
    print(take_image())
