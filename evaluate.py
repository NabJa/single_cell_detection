"""
Script to evaluate a trained model.
"""

import os
from os.path import join, isdir, basename, dirname
from glob import glob
from time import time
from datetime import datetime
import argparse
import prediction.prediction_utils as predict
import prediction.predict_on_series as series_pred
from training import export_all_models
import visualization
import statistics


def main(graph_dir, image_path, out_dir):

    assert isdir(graph_dir), "Model dir not a directory."

    eval_images = glob(join(image_path, "*.png"))

    evaluation_output_path = join(out_dir, "evaluation")

    make_new_dir(evaluation_output_path)

    for graph in glob(join(graph_dir, "*")):
        print(f"Evaluating {basename(graph)}")
        model = predict.load_model(graph)

        evaluation_model_path = join(evaluation_output_path, basename(graph))
        make_new_dir(evaluation_model_path)

        write_image_predictions(model, eval_images, evaluation_model_path)


def write_image_predictions(model, images, out_dir):
    out_dir_images = join(out_dir, "images")
    make_new_dir(out_dir_images)

    predictor = series_pred.image_predictor(images, model)
    series_pred.save_image_predictions(images, predictor, out_dir_images)


def make_new_dir(path):
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
