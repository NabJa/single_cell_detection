"""
Script to predict confluency in brightfield images. Prediction is based in simple image filters.
"""

import numpy as np
import cv2
from pathlib import Path


def simple_segement(image):
    """
    Segement image (brightfield) based on edge detection.
    :return: segmented mask
    """
    kernel = np.ones(9).reshape((3, 3))

    edges = cv2.Canny(image, 15, 50)
    dilation = cv2.dilate(edges, kernel, iterations=2)
    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel, iterations=2)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    erosion = cv2.erode(closing, kernel, iterations=1)

    return erosion


def get_mask_confluency(mask):
    bool_mask = mask.astype(np.bool)
    total = len(bool_mask.flatten())
    foreground = np.sum(bool_mask)
    return foreground/total


def get_image_confluency(image):
    mask = simple_segement(image)
    return get_mask_confluency(mask)


def arbitary_min_max_norm(x, tmax, tmin=0):
    """
    Normalize x to new range(tmin, tmax).
    """
    x = np.array(x)
    rmin, rmax = np.min(x), np.max(x)
    r_diff = rmax - rmin
    t_diff = tmax - tmin
    x = (x - rmin) / r_diff
    return x * t_diff + tmin


def get_image_confluencys(path, normalize=True, image_type=".png"):
    """
    Returns image confluency for every image in path.

    :param path: Path to image directory
    :param normalize: Normalize images to confluence min/max
    :param image_type: Type of images in path
    :return:
    """
    confluencys = []
    path = Path(path)
    images = [x for x in path.iterdir() if x.suffix == image_type]

    for img_path in images:

        image = cv2.imread(str(img_path))

        # Segment image
        segmented = simple_segement(image).astype(np.bool)

        # Get confluence in %
        confluency = np.round(get_mask_confluency(segmented), 4) * 100

        confluencys.append(confluency)
    confluencys = np.array(confluencys)

    if normalize and len(confluencys) == 0:
        conf_max = np.max(confluencys)
        conf_min = np.min(confluencys)
        confluencys = arbitary_min_max_norm(confluencys, conf_max, conf_min)

    return confluencys
