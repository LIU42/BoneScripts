import utils.paint as paint
import utils.preprocess as preprocess

from detect import ScriptDetector
from classify import ScriptClassifier


class ScriptIdentifier:
    def __init__(self, configs):
        self.detector = ScriptDetector(configs)
        self.classifier = ScriptClassifier(configs)

    def __call__(self, image):
        preprocessed_image = preprocess.letterbox(image, size=640, padding_color=255)

        for script in self.classifier(self.detector(preprocessed_image), preprocessed_image):
            paint.script(preprocessed_image, script)

        return preprocessed_image
