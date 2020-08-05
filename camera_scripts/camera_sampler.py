import os
import argparse
import tflite_runtime.interpreter as tflite
from picamera import PiCamera
from time import sleep
import time
import picamera
import numpy as np

INPUT_SIZE = (480,480)
FRAME_RATE = 10

def take_image(input_size):
    with picamera.PiCamera() as camera:
        camera.resolution = input_size
        camera.framerate = FRAME_RATE
        time.sleep(2)
        output = np.empty(INPUT_SIZE + (3,), dtype=np.uint8)
        camera.capture(output,'rgb')
    return output

def run_inference(args):
    model_path = args.model
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = list(input_details[0]['shape'])
    image_shape = (input_shape[1],input_shape[2])
    image = take_image(image_shape) / 255.0
    image = image.astype(np.float32).reshape(input_shape)
    interpreter.set_tensor(input_details[0]['index'], image)
    start = time.time()
    interpreter.invoke()
    print(f'execution time: {time.time() - start}')
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="model and input file paths")
    parser.add_argument('model', metavar='path', type=str, help="Path to tflite model")
    args = parser.parse_args()
    run_inference(args)
