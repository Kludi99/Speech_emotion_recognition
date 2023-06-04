from PyQt6.QtWidgets import (
    QMainWindow,
    QTableView,
    QStatusBar,
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QRadioButton,
    QLabel,
)
from PyQt6.QtCore import Qt


class RadioButton(QWidget):
    def __init__(self, title, options):
        super().__init__()
        self.title = title
        self.options = options
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)

        self.radio_layout = QVBoxLayout()
        # self.radio_layout.addWidget(QLabel(self.title))
        self.radio_group = QGroupBox(self.title)
        # self.radio_layout.addWidget(self.radio_group)
        # self.radio_group.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.radio_group.setLayout(self.radio_layout)
        self.radio_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.radio_buttons = []
        for option in self.options:
            radio_button = QRadioButton(option)

            self.radio_layout.addWidget(radio_button)
            self.radio_buttons.append(radio_button)

        self.setLayout(self.radio_layout)
