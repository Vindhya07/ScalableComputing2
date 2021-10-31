#!/usr/bin/env python3

import warnings
import tflite_runtime.interpreter as tflite
from tflite_runtime.interpreter import Interpreter

import os
import cv2
import numpy
import string
import random
import argparse
import csv

def removeNoise(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = ~gray
    img = cv2.erode(img, numpy.ones((2, 2), numpy.uint8), iterations=1)
    img = ~img
    img = scipy.ndimage.median_filter(img, (5, 1))
    img = scipy.ndimage.median_filter(img, (1, 1))
    thresh = ~cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opening = thresh
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # opening = cv2.erode(thresh, kernel)
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 20:
            cv2.drawContours(opening, [c], -1, 0, -1)

    result = 255 - opening
    return cv2.GaussianBlur(result, (3, 3), 0)

def decode(characters, y):
    y = numpy.argmax(numpy.array(y), axis=2)[:,0]
    y = np.delete(y, np.where(y == 53))
    return ''.join([characters[x] for x in y])

def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='Model name to use for classification', type=str)
    parser.add_argument('--captcha-dir', help='Where to read the captchas to break', type=str)
    parser.add_argument('--output', help='File where the classifications should be saved', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    args = parser.parse_args()

    if args.model_name is None:
        print("Please specify the CNN model to use")
        exit(1)

    if args.captcha_dir is None:
        print("Please specify the directory with captchas to break")
        exit(1)

    if args.output is None:
        print("Please specify the path to the output file")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    symbols_list = list(captcha_symbols)
    arr = []

    print("Classifying captchas with symbol set {" + captcha_symbols + "}")

    with open(args.output, 'w', newline='') as op_csv:
        output_csv_writer = csv.writer(op_csv, delimiter=' ')
        output_csv_writer.writerow(['vnagaraj'])

        interpreter = Interpreter(model_path=args.model_name)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        for x in os.listdir(args.captcha_dir):
            # load image and preprocess it
            raw_data = cv2.imread(os.path.join(args.captcha_dir, x))
            noiseless_Image = removeNoise(raw_data)
            image = noiseless_Image.reshape([-1, 64, 128, 1])
                

            input_shape = input_details[0]['shape']
            input_data = numpy.array(image, dtype=numpy.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)

            interpreter.invoke()

            captcha_op = ''
            # The function `get_tensor()` returns a copy of the tensor data.
            # Use `tensor()` in order to get a pointer to the tensor.
            for i in range(5):
                output_data = interpreter.get_tensor(output_details[i]['index'])
                od = numpy.squeeze(output_data)
                # labels = load_labels('symbols.txt')
                od_char=numpy.argmax(od)
                captcha_op = captcha_op + symbols_list[od_char]

            print(x+ "," + captcha_op)
            arr.append(x+','+captcha_op)

        arr = sorted(arr, key=str)
        for a in arr:
            output_csv_writer.writerow([a])


if __name__ == '__main__':
    main()
