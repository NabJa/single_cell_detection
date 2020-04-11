import numpy as np


def nms(predictions, scores, thresh):
    """
    Base function for Non-maximum suppression.

    :param predictions: Bbox predictions in format: [N, (X1, Y1, X2, Y2)]
    :param scores: Prediction scores [N, 1]
    :param thresh: IoU threshold for NMS
    :return: Indices of predictions to keep.
    """
    x1 = predictions[:, 0]
    y1 = predictions[:, 1]
    x2 = predictions[:, 2]
    y2 = predictions[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= thresh)[0]
        order = order[inds + 1]
    return keep


def greedy_nms(predictions, scores, thresh):
    """
    Suppressed boxes get assigned score=0.
    """
    keep = nms(predictions, scores, thresh)
    weights = np.zeros_like(scores)
    weights[keep] = 1
    new_scores = scores * weights
    return new_scores


def soft_nms(predictions, scores, thresh, sigma=0.3):
    """
    Suppressed boxes get assigned new score penalized based on IoU and sigma.
    """
    x1 = predictions[:, 0]
    y1 = predictions[:, 1]
    x2 = predictions[:, 2]
    y2 = predictions[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    new_scores = scores.copy()
    while order.size > 0:
        i = order[0]
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= thresh)[0]
        iou = iou[inds]

        weight = np.exp(-(iou * iou) / sigma)

        new_scores[inds] = scores[inds] * weight
        order = order[inds + 1]

    return new_scores


def t_nms(predictions, scores, window=3, iou=0.6, det=0.5, nms_func=greedy_nms):
    """
    Post process detections with temporal NMS.
    Concatenate all predeictions in a given window and apply NMS on the combined predictions.
    For more details: https://projet.liris.cnrs.fr/imagine/pub/proceedings/ICPR-2014/data/5209c239.pdf

    :param predictions: List of predicted bboxes. [M, [N, (x1, y1, x2, y2)]]
    :param scores: List of predicted scores. [M, [N, score]]
    :param window: Window size to concatenate predictions
    :param iou: Intersection over Union threshold.
    :param det: Detection score threshold.
    :param nms_func: NMS function. Arguments: (predictions, scores, iou) -> Returns: scores
    :return: Final detections and corresponding scores.
    """

    # Apply NMS on all predictions
    greedy_nms_scores = [nms_func(p, s, iou) for p, s in zip(predictions, scores)]
    processed_p = [p[nms_score >= det] for p, nms_score in zip(predictions, greedy_nms_scores)]
    processed_s = [s[nms_score >= det] for s, nms_score in zip(scores, greedy_nms_scores)]

    temporal_detections = []
    temporal_scores = []
    for i, (prediction, score) in enumerate(zip(processed_p, processed_s)):
        if i < window:
            temporal_detections.append(prediction)
            temporal_scores.append(score)
            continue

        start = i - window + 1

        merged_predictions = processed_p[start:i] + [predictions[i]]
        merged_scores = processed_s[start:i] + [scores[i]]

        merged_predictions = np.concatenate(merged_predictions)
        merged_scores = np.concatenate(merged_scores)

        t_nms_score = nms_func(merged_predictions, merged_scores, iou)

        thresh_pred = merged_predictions[t_nms_score >= det]
        thresh_score = merged_scores[t_nms_score >= det]

        temporal_detections.append(thresh_pred)
        temporal_scores.append(thresh_score)

    return temporal_detections, temporal_scores
