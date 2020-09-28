from PIL import Image
import os
import argparse
import tflite_runtime.interpreter as tflite
from picamera import PiCamera
from time import sleep
import time
import picamera
import numpy as np
from camera_scripts.camera_sampler import read_license_plate
from communications.communicator import Communicator

INPUT_SIZE_1 = (480, 480)
INPUT_SIZE_2 = (280, 560)
FRAME_RATE = 10
FILE_NAME = 'tmp.jpeg'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run the inference and send sms messages")
    parser.add_argument('bbox_model',
                        metavar='path',
                        type=str,
                        help="Path to stage 1 tflite model")
    parser.add_argument('recognition_model',
                        metavar='path',
                        type=str,
                        help="Path to stage 2 tflite model")
    parser.add_argument('-sms', type=str, help="SMS destination number")
    args = parser.parse_args()
    if args.sms:
        communicator = Communicator.getComunicator()
    while (True):
        license_plate = read_license_plate(args)
        print(license_plate)
        if (args.sms):
            communicator.send_mesage(args.sms, license_plate)
