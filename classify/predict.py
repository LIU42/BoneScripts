import numpy as np
import onnxruntime as ort

import utils.process as process


class ScriptClassifier:
    def __init__(self, configs):
        if configs['device'] == 'CUDA':
            providers = ['CUDAExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        self.configs = configs
        self.session = ort.InferenceSession(f'classify/weights/classify-{self.precision}.onnx', providers=providers)

    def __call__(self, scripts, image):
        for script in scripts:
            x1 = script.x1
            y1 = script.y1
            x2 = script.x2
            y2 = script.y2

            inputs = process.preprocess(image[y1:y2, x1:x2], size=64, padding_color=0, precision=self.precision)

            outputs = self.session.run(None, inputs)
            outputs = self.reshape(outputs)

            script.code = self.codes[np.argmax(outputs)]

        return scripts

    @property
    def precision(self):
        return self.configs['precision']

    @property
    def codes(self):
        return self.configs['script-codes']

    @staticmethod
    def reshape(outputs):
        return outputs[0].squeeze()
