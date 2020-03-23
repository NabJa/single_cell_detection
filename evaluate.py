"""
Script to evaluate a trained model.
"""

import os
from os.path import join, isdir, basename, dirname
from glob import glob
from time import time
from datetime import datetime
import argparse
import cv2
from tqdm import tqdm
import prediction.prediction_utils as predict
import prediction.predict_on_series as series_pred
from data import bbox_utils as box
from training import export_all_models
import visualization
import statistics


def main(graph_dir, image_path, out_dir):

    assert isdir(graph_dir), "Model dir not a directory."

    eval_images = glob(join(image_path, "*.png"))

    _, _, out_dir_images = init_subdirs(out_dir, basename(graph_dir))

    for graph in glob(join(graph_dir, "*")):
        print(f"Evaluating {basename(graph)}")

        model = predict.load_model(graph)
        predictor = series_pred.predictor(model, eval_images)

        for i, (prediction, image) in enumerate(tqdm(predictor, total=len(eval_images))):
            bboxes = prediction.get("detection_boxes")[prediction.get("detection_scores") >= 0.5]
            write_image_prediction(join(out_dir_images, f"prediction_{i}.png"), image, bboxes)


def init_subdirs(out_dir, graph_name):
    """
    Init folder structure:
        evaluation
            -> model
                -> images
    """
    evaluation_path = join(out_dir, "evaluation")
    make_new_dir(evaluation_path)

    # Directory for every model
    model_path = join(evaluation_path, graph_name)
    make_new_dir(model_path)

    # Image directory in model
    out_image_path = join(model_path, "images")
    make_new_dir(out_image_path)

    return evaluation_path, model_path, out_image_path


def write_image_prediction(path, image, prediction, points=True):
    """
    Predict on images and save predictions as points.
    """
    if points:
        points = box.boxes_to_center_points(prediction)
        image_prediction = visualization.draw_circles_on_image(image, points)
    else:
        image_prediction = visualization.draw_bboxes_on_image(image, prediction)

    cv2.imwrite(path, image_prediction)


def make_new_dir(path):
    """
    Make a new directory. If it does already exist, add timestamp.
    """
    try:
        os.mkdir(path)
    except FileExistsError:
        dir_name = dirname(path)
        new_dir = basename(path)
        time_stamp = datetime.fromtimestamp(time()).strftime("%Y-%m-%d_%H-%M-%S")
        new_dir = f"{new_dir}-{time_stamp}"
        os.mkdir(join(dir_name, new_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval_images", "-i", required=True)
    parser.add_argument("--out_dir", "-o")
    parser.add_argument("--graph_dirs", "-gd")

    args = parser.parse_args()

    main(args.graph_dirs, args.eval_images, args.out_dir)
