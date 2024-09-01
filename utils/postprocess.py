import cv2
import numpy as np


def get_valid_outputs(outputs, conf_threshold):
    valid_outputs = outputs[outputs[:, 4] > conf_threshold]

    boxes = valid_outputs[:, 0:4]
    confidences = valid_outputs[:, 4]

    return boxes.astype(np.int32), confidences


def non_max_suppression(outputs, conf_threshold, iou_threshold):
    boxes, scores = get_valid_outputs(outputs, conf_threshold)

    boxes[:, 0] -= boxes[:, 2] >> 1
    boxes[:, 1] -= boxes[:, 3] >> 1

    for index in cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold, eta=0.5):
        yield boxes[index]
