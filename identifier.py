import utils.paint as paint
import utils.process as process

from detect import ScriptDetector
from classify import ScriptClassifier


class ScriptIdentifier:
    def __init__(self, configs):
        self.detector = ScriptDetector(configs)
        self.classifier = ScriptClassifier(configs)

    def __call__(self, image):
        preprocessed_image = process.letterbox(image, size=640, padding_color=255)

        for script in self.classifier(self.detector(preprocessed_image), preprocessed_image):
            paint.script(preprocessed_image, script)

        return preprocessed_image
