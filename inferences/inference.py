import cv2
import numpy as np
import onnxruntime as ort
import yaml


with open('configs/inference.yaml', 'r') as configs:
    configs = yaml.load(configs, Loader=yaml.FullLoader)

    precision = configs['precision']
    providers = configs['providers']

    conf_threshold = configs['conf-threshold']
    iou_threshold = configs['iou-threshold']

    character_codes = configs['character-codes']

detection_session = ort.InferenceSession(f'inferences/models/detection-{precision}.onnx', providers=providers)
character_session = ort.InferenceSession(f'inferences/models/character-{precision}.onnx', providers=providers)


def letterbox(image, size, padding):
    current_size = max(image.shape[0], image.shape[1])

    x1 = (current_size - image.shape[1]) >> 1
    y1 = (current_size - image.shape[0]) >> 1

    x2 = x1 + image.shape[1]
    y2 = y1 + image.shape[0]

    background = np.full((current_size, current_size, 3), padding, dtype=np.uint8)
    background[y1:y2, x1:x2] = image

    return cv2.resize(background, (size, size))


def preprocess(image):
    inputs = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))
    inputs = inputs / 255.0
    inputs = np.expand_dims(inputs, axis=0)

    if precision == 'fp16':
        return inputs.astype(np.float16)
    else:
        return inputs.astype(np.float32)
    

def get_valid_outputs(outputs):
    valid_outputs = outputs[outputs[:, 4] > conf_threshold]

    bboxes = valid_outputs[:, 0:4]
    scores = valid_outputs[:, 4]

    return bboxes.astype(np.int32), scores.astype(np.float32)


def non_max_suppression(outputs):
    bboxes, scores = get_valid_outputs(outputs)

    bboxes[:, 0] -= bboxes[:, 2] >> 1
    bboxes[:, 1] -= bboxes[:, 3] >> 1

    for index in cv2.dnn.NMSBoxes(bboxes, scores, conf_threshold, iou_threshold, eta=0.5):
        bboxes[index, 2] += bboxes[index, 0]
        bboxes[index, 3] += bboxes[index, 1]

        yield bboxes[index]


def detection_inference(image):
    outputs = detection_session.run(['output0'], {'images': preprocess(image)})
    outputs = outputs[0]
    outputs = outputs.squeeze().transpose()

    return non_max_suppression(outputs)


def character_inference(character_area):
    outputs = character_session.run(['output0'], {'images': preprocess(letterbox(character_area, size=64, padding=0))})
    outputs = outputs[0]
    outputs = outputs.squeeze()

    return character_codes[np.argmax(outputs)]


def paint_result(image, bbox, character_code):
    point1 = bbox[0:2]
    point2 = bbox[2:4]

    image = cv2.rectangle(image, point1, point2, (0, 255, 0))
    image = cv2.putText(image, character_code, point1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    return image


def inference(image):
    letterbox_image = letterbox(image, size=640, padding=255)

    for bbox in detection_inference(letterbox_image):
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]

        letterbox_image = paint_result(letterbox_image, bbox, character_inference(letterbox_image[y1:y2, x1:x2]))

    return letterbox_image
