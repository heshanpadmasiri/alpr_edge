import os
import argparse
import numpy as np
import tensorflow as tf
import time
from PIL import Image

def run_inference(args):
    model_path = args.model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = list(input_details[0]['shape'])
    image_shape = (input_shape[1],input_shape[2])
    image_path = args.image
    if image_path:
        image = np.array(Image.open(image_path).resize(image_shape), dtype=np.float32) / 255.0
        image = image.reshape(input_shape)
    else:
        image = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], image)
    start = time.time()
    interpreter.invoke()
    print(f'execution time: {time.time() - start}')
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="model and input file paths")
    parser.add_argument('model', metavar='path', type=str, help="Path to tflite model")
    parser.add_argument('-image', metavar='path', type=str, help="Path to image")

    args = parser.parse_args()
    run_inference(args)
