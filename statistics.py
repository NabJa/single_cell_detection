"""
Statistical utils for object detection tasks.
"""

import numpy as np
from scipy.spatial import distance
from data import bbox_utils as box
from sklearn.metrics import auc
import visualization


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


def compute_iou(query_box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    query_box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(query_box[0], boxes[:, 0])
    y2 = np.minimum(query_box[2], boxes[:, 2])
    x1 = np.maximum(query_box[1], boxes[:, 1])
    x2 = np.minimum(query_box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_matches(pred_boxes, gt_boxes, iou_threshold, score_threshold=0.0):
    """
    Compute matches of prediction to gt.

    :return pred_match: indices of matches in gt
    :return gt_match: indices of matches in pred
    :return overlaps: matrix of iou values of every pred, gt combination
    """
    overlaps = compute_overlaps(pred_boxes, gt_boxes)

    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > -1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            match_count += 1
            gt_match[j] = i
            pred_match[i] = j
            break
    return gt_match, pred_match, overlaps


def compute_ap(pred_boxes, gt_boxes, iou_threshold=0.5, bbox_format="xy1xy2"):
    """Compute Average Precision at a set IoU threshold (default 0.5).

    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """

    if bbox_format == "xy1xy2":
        pred_boxes = np.apply_along_axis(box.bbox_xy1xy2_to_yx1yx2, 1, pred_boxes)
        gt_boxes = np.apply_along_axis(box.bbox_xy1xy2_to_yx1yx2, 1, gt_boxes)

    # Get matches and overlaps
    gt_match, pred_match, overlaps = compute_matches(
        pred_boxes, gt_boxes, iou_threshold)

    precisions, recalls = get_precisions_recalls_from_matches(pred_match, gt_match)

    mAP = compute_mean_ap(precisions, recalls)

    return mAP, precisions, recalls, overlaps


def compute_mean_ap(precisions, recalls):
    """
    Compute the mean average precision for a list of precisions, recalls.
    """
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    return np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])


def get_precisions_recalls_from_matches(pred_match, gt_match):
    """
    Given matches between prediction and gt, predict precisions and recalls at each prediction step.

    :param pred_match: Matches of prediction to gt. (As computed by compute_matches())
    :param gt_match: Matches of gt to pred. (As computed by compute_matches())
    """
    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])
    return precisions, recalls


def get_performance_metrics(pred, gt, iou):
    """
    Get performance metrics as defined in:
        https://en.wikipedia.org/wiki/Receiver_operating_characteristic

    :param pred: Predicted bboxes [N, (x1, y1, x2, y2)]
    :param gt: Ground truth bboxes [N, (x1, y1, x2, y2)]
    :param iou: IoU threshold to define true detection

    :return: Dictionary of performance metrics
    """
    gt_match, pred_match, overlaps = compute_matches(pred, gt, iou)

    tp = np.sum(pred_match > -1)
    fn = np.sum(gt_match == -1)
    fp = np.sum(pred_match == -1)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    precisions, recalls = get_precisions_recalls_from_matches(pred_match, gt_match)
    mAP = compute_mean_ap(precisions, recalls)

    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "recall": recall,
        "precision": precision,
        "recalls": recalls,
        "precisions": precisions,
        "fnr": fn / (fn + tp),
        "fdr": fp / (fp + tp),
        "map": mAP,
        "auc": auc(recalls, precisions),
        "fscore": 2 * ((precision*recall) / (precision+recall))
    }


def find_n_closest_points(distances, n):
    """
    Find the n closest points for every point in a distance matrix.

    distances: MxN matrix of distances between all points
    n: Number of cloesest point to find
    """

    # Self similarity is taken into account
    n = n + 1

    close_points_val = np.zeros((distances.shape[0], n))

    for i in range(distances.shape[0]):

        row = distances[i, ...]
        index = np.argpartition(row, n)[:n]
        close_points_val[i, :] = row[index]

    # Remove 0 for closest to itself
    close_points_val = np.sort(close_points_val, axis=1)[:, 1:]

    return close_points_val


def evaluate_distance_cutoffs(pred, gt, cutoffs, image=None, k=3):
    """
    Evaluate prediction based on different distance cutoffs.
    Use to evaluate model capability of predicting dense objects.

    :pred: [N, (xmin, ymin, xmax, ymax)] predicted bboxes
    :gt: [M, (xmin, ymin, xmax, ymax)] ground truth bboxes
    :cutoffs: (list) k-nearest neighbour distance cutoffs to evaluate
    :k: (int) number of nearests neighbours
    """

    gt_points = box.boxes_to_center_points(gt)
    pred_points = box.boxes_to_center_points(pred)

    gt_distance = distance.cdist(gt_points, gt_points, metric="euclidean")
    pred_distance = distance.cdist(pred_points, gt_points, metric="euclidean")

    gt_k_closest = find_n_closest_points(gt_distance, k)
    gt_mean_dist = np.mean(gt_k_closest, axis=1)

    precisions = []
    recalls = []
    images_of_predictions = []

    for c in cutoffs:

        # Extract boxes with dist < c
        predictions_with_gt_smaller_c = pred_distance[:, np.where(gt_mean_dist <= c)[0]]
        query_pred_boxes = pred[np.argmin(predictions_with_gt_smaller_c, axis=0)]

        query_gt_boxes = gt[gt_mean_dist <= c]

        query_pred_points = box.boxes_to_center_points(query_pred_boxes)
        query_gt_points = box.boxes_to_center_points(query_gt_boxes)

        if image is not None:
            # images_of_predictions.append(draw_bboxes_on_image(image, query_gt_boxes, query_pred_boxes))
            img = visualization.draw_circles_on_image(image, query_gt_points)
            img = visualization.draw_circles_on_image(img, query_pred_points)
            images_of_predictions.append(img)

        _, prec, rec, _ = compute_ap(query_pred_boxes, query_gt_boxes, 0.5)

        precisions.append(prec)
        recalls.append(rec)

    return {"precisions": precisions, "recalls": recalls, "images": images_of_predictions}


def rescale_min_max(values, nmin, nmax):
    vmin, vmax = values.min(), values.max()
    scaled = (values - vmin)/(vmax - vmin)
    scaled = scaled * (nmax - nmin) + nmin
    return scaled
