from detector import ScriptDetector
from classifier import ScriptClassifier

from utils import ImageUtils
from utils import MarkingUtils


class ScriptIdentifier:

    def __init__(self, configs):
        self.detector = ScriptDetector(configs)
        self.classifier = ScriptClassifier(configs)

    def __call__(self, image):
        preprocessed_image = ImageUtils.letterbox(image, size=640, padding_color=255)

        for script in self.classifier(self.detector(preprocessed_image), preprocessed_image):
            MarkingUtils.script(preprocessed_image, script)

        return preprocessed_image
