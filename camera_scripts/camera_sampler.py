from PIL import Image
import os
import argparse
import tflite_runtime.interpreter as tflite
from picamera import PiCamera
from time import sleep
import time
import picamera
import numpy as np

INPUT_SIZE_1 = (480,480)
FRAME_RATE = 10
FILE_NAME = 'tmp.jpeg'

def take_image(input_size):
    with picamera.PiCamera() as camera:
        camera.resolution = input_size
        camera.framerate = FRAME_RATE
        time.sleep(2)
        output = np.empty(input_size + (3,), dtype=np.uint8)
        camera.capture(output,'rgb')
    return output

def run_inference_bbox(args,image):
    model_path = args.bbox_model
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = list(input_details[0]['shape'])
    image_shape = (input_shape[1],input_shape[2])
    image = image / 255.0
    image = image.astype(np.float32).reshape(input_shape)
    interpreter.set_tensor(input_details[0]['index'], image)
    start = time.time()
    interpreter.invoke()
    print(f'execution time: {time.time() - start}')
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def save_image(image):
    im = Image.fromarray(image)
    im.save(FILE_NAME, 'JPEG')
    
def load_image(file_name,size):
    image = Image.open(file_name).resize(size)
    return np.array(image)

def to_box_coords(bbox, image_size):
    x,y,w,h = bbox
    left = int(((x-(w/2)) * image_size[0]))
    upper = int(((y+(h/2)) * image_size[1]))
    right = int(((x+(w/2)) * image_size[0]))
    down = int(((y-(h/2)) * image_size[1]))
    return (left,upper, right, down)

def crop_image(image,box_coords):
    cropped_image = image[box_coords[3]:box_coords[1],box_coords[0]:box_coords[2],:]
    return cropped_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="model and input file paths")
    parser.add_argument('bbox_model', metavar='path', type=str, help="Path to stage 1 tflite model")
    parser.add_argument('recognition_model', metavar='path', type=str, help="Path to stage 2 tflite model")
    parser.add_argument('-image', metavar='path', type=str, help="Path to an image to be used insted of camera image")
    args = parser.parse_args()
    if args.image:
        image = load_image(args.image,INPUT_SIZE_1)
    else:
        image = take_image(INPUT_SIZE_1)
    bbox = run_inference_bbox(args,image)[0]
    box_coords = to_box_coords(bbox, INPUT_SIZE_1)
    license_plate = crop_image(image,box_coords)
    print(license_plate)
