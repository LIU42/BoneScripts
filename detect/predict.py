import onnxruntime as ort

import utils.preprocess as preprocess
import utils.postprocess as postprocess

from wrappers import BoneScript


class ScriptDetector:
    def __init__(self, configs):
        if configs['device'] == 'GPU':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        self.configs = configs
        self.session = ort.InferenceSession(f'detector/weights/product/detect-{self.precision}.onnx', providers=providers)

    def __call__(self, image):
        inputs = preprocess.convert(image, precision=self.precision)

        outputs = self.session.run([], inputs)
        outputs = self.postprocessing(outputs)

        results = postprocess.non_max_suppression(outputs, self.conf_threshold, self.iou_threshold)

        return [BoneScript.from_box(box) for box in results]

    @property
    def precision(self):
        return self.configs['precision']

    @property
    def conf_threshold(self):
        return self.configs['conf-threshold']

    @property
    def iou_threshold(self):
        return self.configs['iou-threshold']

    @staticmethod
    def postprocessing(outputs):
        return outputs[0].squeeze().transpose()
