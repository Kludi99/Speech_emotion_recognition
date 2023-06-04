import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QDialog
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import core.const as const
import core.emotions as emotion
import numpy as np


class ConfusionMatrix(QDialog):
    def __init__(self, conf_mat, type, curr_path):
        super().__init__()

        # Create a Figure and Axes for the countplot
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)

        # Draw the countplot on the canvas
        self.confusion_matrix_plot(conf_mat, type, curr_path)

    def get_labels(self, path):
        if path == const.AGH:
            return emotion.labels_agh
        elif path == const.EMO_DB:
            return emotion.labels_emoDB
        elif path == const.RAVDESS:
            return emotion.labels_ravdess
        elif path == const.TESS:
            return emotion.labels_tess

    def get_colors(self, type):
        match type:
            case const.RANDOM_FOREST:
                return plt.cm.Greens
            case const.SVM:
                return plt.cm.Oranges
            case const.KNN:
                return plt.cm.Blues

    def confusion_matrix_plot(self, conf_mat, type, curr_path):
        matrix = conf_mat
        matrix = matrix.astype("float") / matrix.sum(axis=1)[:, np.newaxis]

        # Build the plot
        ax = self.figure.add_subplot(111)
        sns.set(font_scale=1.4)
        sns.heatmap(
            matrix,
            annot=True,
            annot_kws={"size": 10},
            cmap=self.get_colors(type),
            linewidths=0.2,
            ax=ax,
        )

        # Add labels to the plot
        class_names = self.get_labels(curr_path)
        tick_marks = np.arange(len(class_names))
        tick_marks2 = tick_marks + 0.5
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=25)
        ax.set_yticks(tick_marks2)
        ax.set_yticklabels(class_names, rotation=0)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(f"Confusion Matrix for {type} Model")

        self.resize(1200, 900)
