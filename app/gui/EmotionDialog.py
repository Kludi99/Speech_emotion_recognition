from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QVBoxLayout,
    QLabel,
    QComboBox,
    QHBoxLayout,
)
from PyQt6.QtCore import pyqtSignal as Signal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import seaborn as sns


class EmotionDialog(QDialog):
    emotion_chosen = Signal(str)

    def __init__(self, labels):
        super().__init__()
        self.labels = labels
        self.setWindowTitle("Choose emotion for spectrogram")

        QBtn = QDialogButtonBox.StandardButton.Ok

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)

        self.layout = QVBoxLayout()
        self.emotions = QComboBox()
        self.emotions.addItems([x for x in self.labels])
        self.layout.addWidget(QLabel("Choose emotion"))
        self.layout.addWidget(self.emotions)

        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

    def accept(self):
        chosen_emotion = self.emotions.currentText()

        if chosen_emotion != None:
            print(f"Spectrogram Dialog:: Selected emotion: {chosen_emotion}")
            self.emotion_chosen.emit(chosen_emotion)

        self.close()
