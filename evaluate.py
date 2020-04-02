"""
Script to evaluate a trained model.

Evaluation metrics:
    - Prediction images (Single images)
    - Precision recall values (One per image)
    - AUC boxplot (One box for every model)
    - TODO Tracks based on TrackMate (One XML file for every model)
"""

from pathlib import Path
from time import time
from sklearn.metrics import auc
import pickle

import argparse
import cv2
from tqdm import tqdm

from prediction.prediction_utils import load_model, validation_predictor
from data import bbox_utils as box
from data import tf_record_loading as loader
import visualization
import statistics

METRICS = {
    "aucs": [],
    "maps": [],
    "precisions": [],
    "recalls": [],
    "images": []
}


def main(graph_dir, pipeline):

    evaluation_path = make_eval_dir(graph_dir)
    model_dirs = [x for x in graph_dir.iterdir() if x.joinpath("saved_model").is_dir()]

    if "predict" in pipeline:
        get_metrics(model_dirs, evaluation_path, METRICS.copy())
    if "compare" in pipeline:
        compare_models(evaluation_path)


def compare_models(eval_path):
    metrics_dir = eval_path.rglob("metrics.p")
    aucs, maps, names = [], [], []

    for metric_path in metrics_dir:
        metric = pickle.load(metric_path)

        aucs.append(metric.get("aucs"))
        maps.append(metric.get("maps"))
        names.append(metric_path.parent.name)

    visualization.plot_simple_lines(aucs, labels=names, save=eval_path.joinpath("aucs_line.png"))
    visualization.plot_simple_lines(maps, labels=names, save=eval_path.joinpath("maps_line.png"))
    visualization.plot_simple_boxplot(aucs, labels=names, save=eval_path.joinpath("aucs_box.png"))
    visualization.plot_simple_boxplot(maps, labels=names, save=eval_path.joinpath("maps_box.png"))


def get_metrics(model_dir, eval_path, metrics):
    """
    Fills metrics dictionary for a given model_dir.
    """
    for model_dir in model_dir:
        model_eval_dir = eval_path.joinpath(model_dir.name)
        model_eval_dir.mkdir()
        model = load_model(model_dir)

        validation_path = find_validation_image_path(model_dir)
        validation_length = find_validation_length(validation_path)
        predictor = validation_predictor(model, validation_path)

        print("\nEvaluating model: ", model_dir.name)
        metrics = evaluate(predictor, metrics, validation_length)
        save_metrics(metrics, model_eval_dir)


def save_metrics(metrics, out_dir):
    """
    Save images in metrics as images and other values in pickle file.
    """

    out_dir = Path(out_dir)

    image_dir = out_dir.joinpath("images")
    image_dir.mkdir()
    for i, img in enumerate(metrics.pop("images", [])):
        img_path = str(image_dir.joinpath(f"image_{i}.png"))
        cv2.imwrite(img_path, img)

    pickle.dump(metrics, out_dir.joinpath("metrics.p").open("wb"))


def evaluate(predictor, metric, total=None, detection_score=0.5):
    """
    Evaluates prediction of predictor.

    :param predictor: Generator of predictions.
    :param metric: Dictionary of metrics to be saved.
    :param total: Number of validation images.
    :param detection_score: Classification cutoff.
    """

    for prediction in tqdm(predictor, total=total):

        image, gt_bboxes, pred = prediction.get("image"), prediction.get("gt_boxes"), prediction.get("pred")
        pred_bboxes = pred.get("detection_boxes")[pred.get("detection_scores") >= detection_score]

        mAP, precision, recall, _ = statistics.compute_ap(pred_bboxes, gt_bboxes)

        metric.get("aucs").append(auc(recall, precision))
        metric.get("maps").append(mAP)
        metric.get("precision", precision)
        metric.get("recall", recall)

        gt_points = box.boxes_to_center_points(gt_bboxes)
        pred_points = box.boxes_to_center_points(pred_bboxes)
        image = visualization.draw_circles_on_image(image, gt_points, pred_points)
        metric.get("images").append(image)

    return metric


def find_validation_image_path(path):
    """
    Finds the last input_path field in the pipeline.config file.
    Last input_path must correspond to evaluation path.
    """
    path = Path(path)
    pipeline_config = next(path.glob("pipeline.config"))
    res = None
    with pipeline_config.open() as pipeline:
        for line in pipeline.readlines():
            if line.find("input_path") != -1:
                res = line.split(":", 1)[1].strip()
    return Path(res[1:-1])


def find_validation_length(path):
    g = loader.tf_dataset_generator(str(path))
    return len([_ for _ in g])


def make_eval_dir(path):
    try:
        evaluation_path = path.joinpath("Evaluation")
        evaluation_path.mkdir()
    except FileExistsError:
        evaluation_path = path.joinpath(f"Evaluation{time()}")
        evaluation_path.mkdir()
    return evaluation_path


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_dir", "-g", required=True)
    parser.add_argument("--pipeline", "-p", nargs="*", default=["predict", "compare"])
    args = parser.parse_args()

    args_pipeline = [x.lower().strip() for x in args.pipeline]

    arg_metrics = {k: [] for k in args.metric}
    main(Path(args.graph_dir), args_pipeline)
