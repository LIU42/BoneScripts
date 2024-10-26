import cv2

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPainter

from PyQt5.QtWidgets import QAction
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QMainWindow

import ui.languages as languages
import inferences.inference as inference


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.source_image = None
        self.output_image = None

        self.setWindowTitle(languages.main_title)
        self.setMinimumSize(640, 640)

        self.open_action = QAction(languages.open_action, self)
        self.save_action = QAction(languages.save_action, self)
        self.exit_action = QAction(languages.exit_action, self)

        self.open_action.setShortcut('Ctrl+O')
        self.save_action.setShortcut('Ctrl+S')

        self.open_action.setEnabled(True)
        self.save_action.setEnabled(False)

        self.open_action.triggered.connect(self.inference)
        self.save_action.triggered.connect(self.save)
        self.exit_action.triggered.connect(self.close)

        menuber = self.menuBar()

        file_menu = menuber.addMenu(languages.file_menu)
        help_menu = menuber.addMenu(languages.help_menu)

        file_menu.addAction(self.open_action)
        file_menu.addAction(self.save_action)
        help_menu.addAction(self.exit_action)

    @property
    def converted_output(self):
        if self.output_image is not None:
            data = self.output_image.data

            w = self.output_image.shape[1]
            h = self.output_image.shape[0]

            return QImage(data, w, h, QImage.Format_RGB888)

    def inference(self):
        selected_path, _ = QFileDialog.getOpenFileName(self, languages.open_title, '.', languages.types_description)

        if selected_path.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            self.open_action.setEnabled(False)
            self.save_action.setEnabled(False)

            self.source_image = cv2.imread(selected_path)
            self.output_image = inference.inference(self.source_image)

            self.open_action.setEnabled(True)
            self.save_action.setEnabled(True)

    def save(self):
        selected_path, _ = QFileDialog.getSaveFileName(self, languages.save_title, '.', languages.types_description)

        if selected_path.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            self.open_action.setEnabled(False)
            self.save_action.setEnabled(False)

            cv2.imwrite(selected_path, self.output_image)

            self.open_action.setEnabled(True)
            self.save_action.setEnabled(True)

    def paintEvent(self, _):
        painter = QPainter(self)

        if self.output_image is not None:
            scaled_image = self.converted_output.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

            x = (self.width() - scaled_image.width()) // 2
            y = (self.height() - scaled_image.height()) // 2

            painter.drawImage(x, y, scaled_image)
