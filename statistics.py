"""
Statistical utils for object detection tasks.
"""

import numpy as np
from scipy.spatial import distance
from data import bbox_utils as box

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


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_matches(pred_boxes, gt_boxes, iou_threshold, score_threshold=0.0):
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


def compute_ap(pred_boxes, gt_boxes, iou_threshold=0.5):
    """Compute Average Precision at a set IoU threshold (default 0.5).

    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Get matches and overlaps
    gt_match, pred_match, overlaps = compute_matches(
        pred_boxes, gt_boxes, iou_threshold)

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

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])

    return mAP, precisions, recalls, overlaps


def compute_precision_recall(pred_boxes, gt_boxes, iou):
    """Compute the recall and precision at the given IoU threshold.

    pred_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    gt_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    """
    # Measure overlaps
    overlaps = compute_overlaps(pred_boxes, gt_boxes)
    iou_max = np.max(overlaps, axis=1)
    iou_argmax = np.argmax(overlaps, axis=1)

    positive_ids = np.where(iou_max >= iou)[0]
    negative_ids = np.where(iou_max < iou)[0]

    matched_gt_boxes = iou_argmax[positive_ids]
    unmatched_gt_boxes = iou_argmax[negative_ids]

    true_positives = len(set(matched_gt_boxes))
    false_negatives = len(set(unmatched_gt_boxes))

    recall = true_positives / gt_boxes.shape[0]
    precision = true_positives / (true_positives+false_negatives)

    return recall, precision, positive_ids


def find_n_closest_points(distances, n):
    """
    Find the n closest points for every point in a distance matrix.

    distances: MxN matrix of distances between all points
    n: Number of cloesest point to find
    """

    # Self similarity is taken into account
    n = n + 1

    close_points_val = np.zeros((distances.shape[0] ,n))

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
            img = draw_circles_on_image(image, query_gt_points)
            img = draw_circles_on_image(img, query_pred_points, default_color=(0, 0, 255))
            images_of_predictions.append(img)

        _, prec, rec, _ = compute_ap(query_pred_boxes, query_gt_boxes, 0.5)

        precisions.append(prec)
        recalls.append(rec)

    return {"precisions": precisions, "recalls": recalls, "images": images_of_predictions}


# Must be at the end to workaround cyclic import with visualization.py
from visualization import draw_bboxes_on_image, draw_circles_on_image
