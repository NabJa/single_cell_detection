"""
Utility functions to slice images and bounding boxes.
"""

import argparse
import numpy as np
from numba import jit
from tf_record_loading import load_tf_dataset

def split_image(image, tiles=4):

    split = tiles / 2

    height, width, *_ = image.shape

    new_height = height / split
    new_width = width / split

    tile_height = new_height

    crops = []

    for row in range(int(split)):
        tile_width = new_width
        for col in range(int(split)):

            ymin = int(tile_height-new_height)
            ymax = int(tile_height)
            xmin = int(tile_width-new_width)
            xmax = int(tile_width)

            crops.append(image[ymin:ymax, xmin:xmax])

            tile_width = tile_width + new_width
        tile_height = tile_height + new_height

    return crops


def split_image_with_bboxes(bboxes, image, tiles=4):
    """
    Split an image with its corresponding bboxes into the requested number of tiles.

    bboxes (list): Bboxes must be in (xmin, ymin, xmax, ymax) format.
    image (np.array): Image to split.
    tiles (int): Number of tiles to split into.
    """

    if tiles == 0:
        return {(0, 0): {"image": image, "bboxes": bboxes}}
    assert tiles % 2 == 0, "Error in splitting images. Uneven number of images requested."

    split = tiles / 2

    height, width, *_ = image.shape

    new_height = height / split
    new_width = width / split

    tiles = {}

    tile_height = new_height

    for row in range(int(split)):
        tile_width = new_width
        for col in range(int(split)):

            # Create image with true values on tile
            canvas = np.zeros_like(image)
            tile_start = (int(tile_height-new_height), int(tile_width-new_width))
            tile_end = (int(tile_height), int(tile_width))
            canvas[tile_start[0]:tile_end[0], tile_start[1]:tile_end[1]] = 1

            new_bboxes = []
            for bbox in bboxes:

                xmin, ymin, xmax, ymax = bbox

                # Overlap of image tile and bbox
                bbox_image = np.zeros_like(image)
                bbox_image[ymin:ymax, xmin:xmax] = 1

                overlap = np.logical_and(canvas, bbox_image)

                if np.sum(overlap) < 1:
                    continue

                overlap_index = np.argwhere(overlap)

                overlap_xmin, overlap_ymin = overlap_index[0][1], overlap_index[0][0]
                overlap_xmax, overlap_ymax = overlap_index[-1][1]+1, overlap_index[-1][0]+1

                new_xmin = overlap_xmin - col * new_width
                new_ymin = overlap_ymin - row * new_height
                new_xmax = overlap_xmax - col * new_width
                new_ymax = overlap_ymax - row * new_height

                new_bbox = (new_xmin, new_ymin, new_xmax, new_ymax)

                new_bboxes.append(new_bbox)

            cropped_image = image[tile_start[0]:tile_end[0], tile_start[1]:tile_end[1]]
            tiles[(row, col)] = {"image": cropped_image, "bboxes": new_bboxes}

            tile_width = tile_width + new_width
        tile_height = tile_height + new_height

    return tiles


def split_image_with_bboxes_efficient(bboxes, image, bbox_size=50, tiles=4):
    """
    Split an image with its corresponding bboxes into the requested number of tiles.

    bboxes (list): Bboxes must be in (xmin, ymin, xmax, ymax) format.
    image (np.array): Image to split.
    tiles (int): Number of tiles to split into.
    """

    if tiles == 0:
        return {(0, 0): {"image": image, "bboxes": bboxes}}

    split = tiles / 2

    height, width, *_ = image.shape

    new_height = height / split
    new_width = width / split

    tiles = {}

    tile_height = new_height

    for row in range(int(split)):
        tile_width = new_width
        for col in range(int(split)):

            # Create image with true values on tile
            canvas = np.zeros_like(image)

            ymin = int(tile_height-new_height)
            ymax = int(tile_height)
            xmin = int(tile_width-new_width)
            xmax = int(tile_width)

            canvas[ymin:ymax, xmin:xmax] = 1

            query_bboxes = find_query_boxes(bboxes, xmin, xmax, ymin, ymax, bbox_size)

            new_bboxes = find_overlaps(canvas, query_bboxes, col, row, new_width, new_height)

            cropped_image = image[ymin:ymax, xmin:xmax]

            tiles[(row, col)] = {"image": cropped_image, "bboxes": new_bboxes}

            tile_width = tile_width + new_width
        tile_height = tile_height + new_height

    return tiles


def find_overlaps(canvas, bboxes, col, row, new_width, new_height):

    new_bboxes = []

    for bbox in bboxes:

        xmin, ymin, xmax, ymax = bbox

        # Overlap of image tile and bbox
        bbox_image = np.zeros_like(canvas)
        bbox_image[ymin:ymax, xmin:xmax] = 1

        overlap = np.logical_and(canvas, bbox_image)

        if np.sum(overlap) < 1:
            continue

        new_bbox = find_bbox_from_overlap(overlap, col, row, new_width, new_height)

        new_bboxes.append(new_bbox)

    return new_bboxes


@jit
def find_bbox_from_overlap(overlap, col, row, width, height):
    overlap_index = np.argwhere(overlap)

    overlap_xmin, overlap_ymin = overlap_index[0][1], overlap_index[0][0]
    overlap_xmax, overlap_ymax = overlap_index[-1][1]+1, overlap_index[-1][0]+1

    new_xmin = overlap_xmin - col * width
    new_ymin = overlap_ymin - row * height
    new_xmax = overlap_xmax - col * width
    new_ymax = overlap_ymax - row * height

    new_bbox = (new_xmin, new_ymin, new_xmax, new_ymax)
    return new_bbox


def find_query_boxes(bboxes, xmin, xmax, ymin, ymax, bbox_size=50):
    """
    bbox: (xmin, ymin, xmax, ymax)
    ymin: canvas
    ymax: canvas
    xmin: canvas
    xmax: canvas

    (ymin, xmin) ------------ (ymin, xmax)
      |                           |
      |                           |
      |                           |
      |                           |
    (ymax, xmin) ------------ (ymax, xmax)
    """
    bboxes_np = np.array(bboxes)

    bboxes_minus_ymin = bboxes_np - ymin + bbox_size # Everything above canvas < 0
    bboxes_minus_ymax = bboxes_np - ymax - bbox_size # Everything above or inside canvas < 0

    bboxes_minus_xmin = bboxes_np - xmin + bbox_size # Everything left of canvas < 0
    bboxes_minus_xmax = bboxes_np - xmax - bbox_size # Everything left of or inside canvas < 0

    bboxes_idxs = (np.argwhere((bboxes_minus_ymin[..., 1] >= 0) &
                               (bboxes_minus_ymax[..., 3] <= 0) &
                               (bboxes_minus_xmin[..., 0] >= 0) &
                               (bboxes_minus_xmax[..., 2] <= 0)
                              )).flatten()

    return bboxes_np[bboxes_idxs].tolist()
