"""
Script to save predictions on a given image series.
Predictions can be saved as images or as vid (avi).
"""

import sys

import os
from os.path import join, basename
import argparse

import pathlib
from PIL import Image
import io
from tqdm import tqdm
from glob import glob
import pickle
import numpy as np
import cv2

import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .prediction_utils import run_inference_for_single_image

from data.bbox_utils import boxes_to_center_points
from visualization import draw_circles_on_image

def main(image_dir, model_dir, out_dir, out_type):

    if not out_dir:
        out_dir = join(image_dir, "predictions")

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    images = glob(join(image_dir, "*.png"))

    detection_model = _load_model(model_dir)

    if out_type == "img":
        predictor = image_predictor(images, detection_model)
        save_image_predictions(images, predictor, out_dir)
    elif out_type == "vid":
        predictor = image_predictor(images, detection_model)
        save_video_prediction(images, predictor, out_dir)
    elif out_type == "pickle":
        predictor = model_predictor(images, detection_model)
        save_pickle_predictions(predictor, out_dir)
    else:
        print("Unsupported output type: ", out_type)


def save_image_predictions(images, predictor, out_dir):
    for img_path in tqdm(images):
        prediction = next(predictor)
        cv2.imwrite(join(out_dir, os.path.basename(img_path)), prediction)


def save_pickle_predictions(predictor, out_dir):
    for i, prediction in enumerate(tqdm(predictor)):
        pickle.dump(prediction, open(join(out_dir, f"prediction_{i}.p"), "wb"))


def save_video_prediction(images, predictor, out_dir):
    height, width, *_ = cv2.imread(images[0]).shape
    video = cv2.VideoWriter(join(out_dir, 'predictions.avi'), cv2.VideoWriter_fourcc(*'DIVX'), 5, (width, height))
    for _ in tqdm(images):
        prediction = next(predictor)
        video.write(prediction)
    video.release()


def model_predictor(image_dir, model):
    index = 0
    while index < len(image_dir):
        image = cv2.imread(image_dir[index], 1)
        prediction = _run_inference_for_single_image(model, image)

        detected_boxes = prediction["detection_boxes"]

        # Transform bboxes to image coordinates. Prediction is in yx1yx2 format.
        img_height, img_width, *_ = image.shape
        shape_matrix = np.array([img_height, img_width, img_height, img_width])
        prediction["detection_boxes"] = detected_boxes * shape_matrix
        prediction["image_dir"] = image_dir[index]

        index += 1
        yield prediction


def image_predictor(image_dir, model):
    index = 0
    while index < len(image_dir):

        image = cv2.imread(image_dir[index], 1)

        # prediction = _run_inference_for_single_image(model, image)

        # # Take only predictions with detection_scores >= 0.5
        # detected_boxes = prediction["detection_boxes"][prediction['detection_scores'] >= .5, :]

        # # Transform bboxes to image coordinates
        # img_height, img_width, *_ = image.shape
        # shape_matrix = np.array([img_height, img_width, img_height, img_width])
        # normalized_bboxes = detected_boxes * shape_matrix
        prediction = run_inference_for_single_image(model, image)
        bboxes = prediction.get("detection_boxes")[prediction.get("detection_scores") >= 0.5]
        points = boxes_to_center_points(bboxes)
        image = draw_circles_on_image(image, points, default_color=(0, 0, 255))
        index += 1
        yield image
        # # Draw bboxes on image
        # for bbox in normalized_bboxes:
        #     y1, x1, y2, x2 = bbox
        #     color = (255, 0, 0)
        #     thickness = 2
        #     image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

        # index += 1
        # yield image


def predictor(model, image_dir):
    index = 0
    while index < len(image_dir):

        image = cv2.imread(image_dir[index], 1)
        prediction = run_inference_for_single_image(model, image)
        index += 1
        yield prediction, image


def _load_model(model_name):
    _model_dir = pathlib.Path(model_name)/"saved_model"
    model = tf.saved_model.load(str(_model_dir))
    model = model.signatures['serving_default']
    return model


def _dir_path(path):
    if os.path.isdir(path):
        return path
    raise Exception("Invalid directory")


def _run_inference_for_single_image(model, image):

    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy()
                 for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    return output_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Predict on image series.')

    parser.add_argument('--image_dir', '-i', type=_dir_path,
                        help="Path to directory with folders models.")
    parser.add_argument('--model_dir', '-m', type=_dir_path,
                        help="Path to directory with saved model.")
    parser.add_argument('--out_dir', '-o', type=str,
                        help="Path to save predictions.")
    parser.add_argument('--out_type', '-t', type=str,
                        help="vid=Save predictions as video \nimg=Save predictions as images \npickle=Save detections as pickles")
    args = parser.parse_args()

    main(args.image_dir, args.model_dir, args.out_dir, args.out_type)
