import onnxruntime as ort

from wrappers import ScriptBuilder
from utils import ImageUtils
from utils import ResultUtils


class ScriptDetector:

    def __init__(self, config):
        if config['device'] == 'GPU':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        self.config = config
        self.session = ort.InferenceSession(f'detector/weights/product/detect-{self.precision}.onnx', providers=providers)

    def __call__(self, image):
        inputs = ImageUtils.convert(image, precision=self.precision)

        outputs = self.session.run(None, {
            'images': inputs,
        })
        outputs = outputs[0].squeeze()
        outputs = outputs.transpose()
        
        results = ResultUtils.non_max_suppression(outputs, self.conf_threshold, self.iou_threshold)

        return [ScriptBuilder.box(box) for box in results]

    @property
    def precision(self):
        return self.config['precision']

    @property
    def conf_threshold(self):
        return self.config['conf-threshold']

    @property
    def iou_threshold(self):
        return self.config['iou-threshold']
