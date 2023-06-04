import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QDialog
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class CountPlot(QDialog):
    def __init__(self, data):
        super().__init__()

        # Create a Figure and Axes for the countplot
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)

        # Create a countplot using Seaborn
        ax = self.figure.add_subplot(111)
        sns.countplot(data, ax=ax)

        # Draw the countplot on the canvas
        self.canvas.draw()
