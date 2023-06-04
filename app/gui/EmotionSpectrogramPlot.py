import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QDialog
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import librosa
import librosa.display
import numpy as np
from matplotlib.cm import ScalarMappable


class EmotionSpectrogramPlot(QDialog):
    def __init__(self, path, emotion):
        super().__init__()

        self.figure = Figure(figsize=(10, 4))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.canvas)
        data, sample_rate = librosa.load(path)
        print(path)
        self.plot_spectrogram(data, sample_rate, emotion)

    def plot_spectrogram(self, data, sample_rate, emotion):
        x = librosa.stft(data)
        xdb = librosa.amplitude_to_db(np.abs(x), ref=np.max)

        self.ax.clear()
        self.ax.set_title(emotion, size=20)
        duration = len(data) / sample_rate
        im = self.ax.imshow(
            xdb,
            origin="lower",
            aspect="auto",
            cmap="plasma",
            extent=[0, duration, 0, sample_rate / 2],
        )

        colorbar = self.figure.colorbar(im, format="%+2.0f dB")
        colorbar.set_label("Amplitude (dB)")

        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Frequency")

        self.canvas.draw()
