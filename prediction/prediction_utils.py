"""
Utility functions to predict on images given a tenforlow object detection API model.
"""

import sys

import argparse
from pathlib import Path
import pickle

import numpy as np
import tensorflow as tf
import cv2

sys.path.append("../")
from data import tf_record_loading as tf_loader
from data import bbox_utils as box
from visualization import draw_circles_from_boxes


def load_model(model_name):
    model_dir = Path(model_name) / "saved_model"

    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']

    return model


def run_inference_for_single_image(model, image):
    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=-1)

    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(
        np.int64)
    detections_normalized = box.normalize_bboxes_to_image(output_dict["detection_boxes"], image)
    detections_normalized = np.apply_along_axis(box.bbox_yx1yx2_to_xy1xy2, 1, detections_normalized)

    output_dict["detection_boxes"] = detections_normalized

    return output_dict


def run_inference_on_batch(model, batch):
    """
    model: tf model
    batch: batch of images with shape (B, W, H, C)
    """

    output_dict = model(batch)

    # TODO function box.normalize_bboxes_to_image should take shape, not image
    detections_normalized = box.normalize_bboxes_to_image(output_dict["detection_boxes"].numpy(), batch[0, ...].numpy())
    detections_normalized = np.apply_along_axis(box.bbox_yx1yx2_to_xy1xy2, 2, detections_normalized)
    output_dict["detection_boxes"] = detections_normalized
    return output_dict


def predict_on_tiled_image(model, image, tiles=4):
    """
    """

    if tiles == 0:
        prediction = run_inference_for_single_image(model, image)

        bboxes = prediction.get("detection_boxes")
        confidence = prediction.get("detection_scores")

        bboxes = box.normalize_bboxes_to_image(bboxes, image)
        bboxes = np.apply_along_axis(box.bbox_yx1yx2_to_xy1xy2, 1, bboxes)

        return bboxes, confidence

    split = tiles / 2

    height, width, *_ = image.shape

    new_height = height / split
    new_width = width / split

    tile_height = new_height

    all_bboxes = []
    confidences = []

    for row in range(int(split)):
        tile_width = new_width
        for col in range(int(split)):

            ymin = int(tile_height - new_height)
            ymax = int(tile_height)
            xmin = int(tile_width - new_width)
            xmax = int(tile_width)

            cropped_image = image[ymin:ymax, xmin:xmax]

            prediction = run_inference_for_single_image(model, cropped_image)

            # Normalize prediction bboxes
            bboxes = prediction.get("detection_boxes")
            confidence = prediction.get("detection_scores")

            new_bboxes = _normalize_bbox_coordinates(
                bboxes, col, row, new_width, new_height)

            if len(all_bboxes) == 0:
                all_bboxes = new_bboxes
                confidences = confidence

            all_bboxes = np.append(all_bboxes, new_bboxes, axis=0)
            confidences = np.append(confidences, confidence, axis=0)

            tile_width = tile_width + new_width
        tile_height = tile_height + new_height

    return all_bboxes, confidences


def validation_predictor(model, tf_record):
    """
    Predicts on every image in tf_record
    """
    data = tf_loader.load_tf_dataset(tf_record)
    images = data.get("images")
    gt_bboxes = data.get("bboxes")
    for image, gt_bbox in zip(images, gt_bboxes):
        prediction = run_inference_for_single_image(model, image)
        yield {"gt_boxes": gt_bbox, "pred": prediction, "image": image}


def patch_image(img, sizes, strides, padding="SAME"):
    """
    This op collects patches from the input image, as if applying a convolution
    and reshapes the result into the format (BATCH, ROWS, COLUMNS, WIDTH, HEIGHT, CHANNELS).
    :param img: Input image
    :param sizes: Kernel size in format (BATCH, WIDTH, HEIGHT, CHANNEL)
    :param strides: Strides in format (BATCH, WIDTH, HEIGHT, CHANNEL)
    :param padding: VALID or SAME. If VALID, only patches which are fully contained in the input image are included.
    If SAME, all patches whose starting point is inside the input are included,
    and areas outside the input default to zero.
    """
    img = update_image_shape(img)
    channels = img.shape[-1]

    patches = tf.image.extract_patches(img,
                                       sizes=sizes,
                                       strides=strides,
                                       rates=(1, 1, 1, 1),
                                       padding=padding)
    batch, rows, cols, img_values = patches.shape
    patches = tf.reshape(patches, (batch, rows, cols, sizes[1], sizes[2], channels))
    return patches


def patches_to_batch(p):
    """Reformat patches p of format  (BATCH, ROWS, COLUMNS, WIDTH, HEIGHT, CHANNELS)
    into (BATCH, CROP, WIDTH, HEIGHT, CHANNELS)"""
    batch, row, col, w, h, c = p.shape
    patches_batch = tf.reshape(p, (row * col, w, h, c))
    return patches_batch


def update_image_shape(img):
    """Image bust be shaped as: (batch, width, height, channels)"""
    if len(img.shape) == 2:
        return img[tf.newaxis, ..., tf.newaxis]
    elif len(img.shape) == 3:
        return img[tf.newaxis, ...]
    elif len(img.shape) == 4:
        return img
    else:
        raise ValueError("Invalid image shape")


def calc_padding(i, o, k, s):
    """
    Calculate the resulting SAME padding.
    :param i: input shape (height, width)
    :param o: output shape (height, width)
    :param k: kernel size (height, width)
    :param s: stride (height, width)
    :return: (Float tuple) Padding Height, Padding Width
    """
    ih, iw = i
    oh, ow = o
    kh, kw = k
    sh, sw = s
    padding_h = 0.5 * (kh + oh * sh - sh - ih)
    padding_w = 0.5 * (kw + ow * sw - sw - iw)
    return padding_h, padding_w


def run_patched_prediction(tfmodel, img, kernel=(1, 224, 224, 1), stride=(1, 200, 200, 1)):
    """
    Run prediction on patches of an image. Patches are calculated based on kernel and stride.
    :param tfmodel: Tensorflow model object.
    :param img: Input image.
    :param kernel: Kernel to generate patches of shape (BATCH, WDITH, HEIGHT, CHANNEL)
    :param stride: Stride to generate patches of shape (BATCH, WDITH, HEIGHT, CHANNEL)
    :return: Points and scores in input image (img) coordinates.
    """
    patches = patch_image(img, kernel, stride)
    batch = patches_to_batch(patches)

    output_shape = patches.shape[1:3]
    ph, pw = calc_padding(img.shape[:2], output_shape, kernel[1:3], stride[1:3])

    predictions = run_inference_on_batch(tfmodel, batch)

    boxes = predictions.get("detection_boxes")
    scores = predictions.get("detection_scores")

    # Final points and scores
    p, s = [], []
    i = 0
    for row in range(output_shape[0]):
        for col in range(output_shape[1]):

            # Extract points and scores
            detection_box = boxes[i]
            score = scores[i]
            points = box.boxes_to_center_points(detection_box)

            # Subtract padding
            padding = np.array([pw, ph])
            points = points - padding

            # Add stride shift
            shift_right = (col * kernel[1]) - (col * (kernel[1] - stride[1]))
            shift_down = (row * kernel[2]) - (row * (kernel[2] - stride[2]))
            points = points + np.array([shift_right, shift_down])

            s.append(score)
            p.append(points)
            i += 1

    scores = np.concatenate(s)
    points = np.concatenate(p)
    scores = scores[np.where((points[:, 0] >= 0) & (points[:, 1] >= 0))]
    points = points[np.where((points[:, 0] >= 0) & (points[:, 1] >= 0))]

    return points, scores



def _normalize_bbox_coordinates(bboxes, col, row, width, height):
    new_xmin = col * width
    new_ymin = row * height
    new_xmax = col * width
    new_ymax = row * height

    new_bboxes = bboxes + np.array([new_xmin, new_ymin, new_xmax, new_ymax])
    return new_bboxes


"""
ARGPARSE
"""


def _image_path(p):
    formats = [".png", ".tif", ".tiff", ".jpg", ".jpeg"]
    p = Path(p)
    if p.is_file() and p.suffix in formats:
        return p
    else:
        raise ValueError(f"Image must be one of the following formats: {formats}")


def _model_path(p):
    p = Path(p)
    if p.is_dir() and p.joinpath("saved_model").is_dir():
        return p
    else:
        raise ValueError("Model must be saved tensorflow model containing \"saved_model\" subfolder.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=_model_path, help="Path to save tensorflow model.")
    parser.add_argument("-i", "--image", type=_image_path, help="Runs prediction on single image.")
    parser.add_argument("-o", "--output", help="Output path. Will be created if does not exist.")
    parser.add_argument("-v", "--visualize", default=True,
                        help="Wheather prediction should be visualized. DEFAULT=True")
    args = parser.parse_args()

    if args.image:
        model = load_model(str(args.model))
        image = cv2.imread(str(args.image))
        print("Predicting...")
        prediction = run_inference_for_single_image(model, image)
        if args.output:
            out_path = Path(args.output)
            out_path.mkdir(parents=True, exist_ok=True)
        else:
            out_path = Path().cwd()
        pickle.dump(prediction, open(str(out_path.joinpath("raw_predictions.p")), "wb"))
        print(f"Saved prediction in: {out_path}")

        if args.visualize:
            detection = prediction.get("detection_boxes")[prediction.get("detection_scores") >= 0.5]
            out_img = draw_circles_from_boxes(image, detection)
            cv2.imwrite(str(out_path.joinpath("res_image.png")), out_img)
