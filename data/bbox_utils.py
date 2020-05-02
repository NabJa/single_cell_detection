"""
Utils to transform Bounding boxes.
"""

import numpy as np
import cv2

def bbox_xy1xy2_to_xywh(bbox):
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin
    return np.array([xmin, ymin, width, height])


def bbox_yx1yx2_to_xywh(bbox):
    ymin, xmin, ymax, xmax = bbox
    width = xmax - xmin
    height = ymax - ymin
    return np.array([xmin, ymin, width, height])


def bbox_xywh_to_yx1yx2(bbox):
    xmin, ymin, width, height = bbox
    xmax = xmin + width
    ymax = ymin + height
    return np.array([ymin, xmin, ymax, xmax])


def bbox_xy1xy2_to_yx1yx2(bbox):
    xmin, ymin, xmax, ymax = bbox
    return np.array([ymin, xmin, ymax, xmax])


def bbox_yx1yx2_to_xy1xy2(bbox):
    ymin, xmin, ymax, xmax = bbox
    return np.array([xmin, ymin, xmax, ymax])


def normalize_bboxes_to_image(bboxes, image):
    """
    Transform normalized bbox coordinates back to image coordinates.
    """
    normalized_bboxes = bboxes.copy()

    img_height, img_width, *_ = image.shape
    shape_matrix = np.array([img_height, img_width, img_height, img_width])

    normalized_bboxes = bboxes * shape_matrix

    return normalized_bboxes


def calc_bbox_from_mask(mask, rotation=False):
    """
    Calculates bounding box for a given mask.
    Returns binary image with bounding box and rectangle coordinates (optionally with rotation).
    """
    img = np.zeros_like(mask)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # From all contours, select largest
    contour_areas = [cv2.contourArea(i) for i in contours]
    contour = contours[np.argmax(contour_areas)]

    if rotation:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img, [box], 0, (255, 0, 0), -1)

    else:
        rect = cv2.boundingRect(contour)
        x, y, w, h = rect
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), -1)

    return img, rect


def boxes_to_center_points(boxes, bbox_format="xy1xy2"):
    """
    Calculate all center points of all boxes.
    """
    xmin, ymin, xmax, ymax = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    height = ymax - ymin
    width = xmax - xmin

    x_offset = width/2
    y_offset = height/2

    points = np.zeros((boxes.shape[0], 2))

    points[:, 0] = xmin + x_offset
    points[:, 1] = ymin + y_offset

    return points


def box_to_point(box, bbox_format="xy1xy2"):
    """
    Calculate the center point of a box.
    """
    xmin, ymin, xmax, ymax = box

    height = ymax - ymin
    width = xmax - xmin

    x_offset = width/2
    y_offset = height/2

    return xmin + x_offset, ymin + y_offset


def point_to_box(point, size=30):
    """Transform point (x, y) to fixed sized box (x1, y1, x2, y2)."""
    offset = size // 2

    px, py = point[0], point[1]

    x1 = px - offset if px - offset > 0 else 0
    y1 = py - offset if py - offset > 0 else 0
    x2 = px + offset
    y2 = py + offset

    return (x1, y1, x2, y2)
