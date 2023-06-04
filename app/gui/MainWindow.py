from PyQt6.QtWidgets import (
    QMainWindow,
    QTableView,
    QStatusBar,
    QMessageBox,
    QHBoxLayout,
    QWidget,
    QRadioButton,
    QLabel,
    QVBoxLayout,
    QCheckBox,
    QPushButton,
    QGridLayout,
)
from PyQt6.QtCore import Qt
from PyQt6.QtCore import pyqtSignal as Signal
from gui.RadioButton import RadioButton
import pandas as pd

from core.core import *
import core.const as const
from gui.CountPlot import CountPlot
from gui.ConfusionMatrixPlot import ConfusionMatrix
from gui.EmotionDialog import EmotionDialog
from gui.EmotionSpectrogramPlot import EmotionSpectrogramPlot


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data = None
        self.database_toggled = ""
        self.classifier_toggled = ""
        self.options_toggled = ""
        self.csv_checked = False
        self.df = pd.DataFrame()
        self.submit_button = Signal(str, str, str, bool)
        # initialize main window
        self.setWindowTitle(
            "Analysis of emotional state of the speaker based on the speech signal"
        )
        self.setGeometry(100, 100, 800, 300)
        self.labels = []
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QGridLayout(central_widget)

        layout.addWidget(
            QLabel("Speech emotion recognition using MFCC"),
            0,
            0,
            alignment=Qt.AlignmentFlag.AlignLeft,
        )

        databases_header = QLabel("Databases")

        layout.addWidget(databases_header, 1, 0, alignment=Qt.AlignmentFlag.AlignCenter)

        databases = [const.AGH, const.EMO_DB, const.RAVDESS, const.TESS]
        db_radio = RadioButton("Databases", databases)

        layout.addWidget(db_radio, 2, 0, alignment=Qt.AlignmentFlag.AlignCenter)

        classifiers_header = QLabel("Classifiers")
        classifiers_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(
            classifiers_header, 1, 1, alignment=Qt.AlignmentFlag.AlignCenter
        )

        classifiers = [const.SVM, const.RANDOM_FOREST, const.KNN]
        clf_radio = RadioButton("Classifiers", classifiers)

        layout.addWidget(clf_radio, 2, 1, alignment=Qt.AlignmentFlag.AlignCenter)

        self.options_header = QLabel("Options")
        self.options_header.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(
            self.options_header, 1, 2, alignment=Qt.AlignmentFlag.AlignCenter
        )

        options_radio = RadioButton("Options", [])

        layout.addWidget(options_radio, 2, 2)

        self.checkbox = QCheckBox("Generate csv")
        # row3_layout.addWidget(self.checkbox)
        layout.addWidget(self.checkbox, 4, 0)

        button = QPushButton("Start classification")
        # row3_layout.addWidget(button)
        layout.addWidget(button, 4, 2)

        button_counterplot = QPushButton("CounterPlot")
        # button_counterplot.adjustSize()
        button_counterplot.setMinimumWidth(100)
        button_counterplot.setMinimumHeight(30)
        layout.addWidget(
            button_counterplot, 0, 1, alignment=Qt.AlignmentFlag.AlignCenter
        )

        button_emotion = QPushButton("Spectrogram")
        # button_emotion.adjustSize()
        button_emotion.setMinimumWidth(100)
        button_emotion.setMinimumHeight(30)

        layout.addWidget(button_emotion, 0, 2, alignment=Qt.AlignmentFlag.AlignCenter)

        # Connect the second column's radio buttons to update the options in the third column
        for radio_button in clf_radio.radio_buttons:
            radio_button.toggled.connect(
                lambda state, rb=radio_button: self.onClassifierToggled(
                    state, rb, options_radio
                )
            )

        # Connect the second column's radio buttons to update the options in the third column
        for radio_button in db_radio.radio_buttons:
            radio_button.toggled.connect(
                lambda state, rb=radio_button: self.onDatabaseToggled(state, rb)
            )
        button.pressed.connect(self.save_button_clicked)
        button_counterplot.pressed.connect(self.show_countplot)
        button_emotion.pressed.connect(self.spectrogram_clicked)

    def onClassifierToggled(self, state, radio_button, options_radio):
        print("state ", state)
        print("radio button ", radio_button.text())
        print("options radio ", options_radio)
        if radio_button.text() == const.SVM and radio_button.isChecked():
            options_radio.options = ["0.01", "0.1", "1", "10", "100"]
            self.options_header.setText("Option - 'c' value")
        elif radio_button.text() == const.RANDOM_FOREST and radio_button.isChecked():
            options_radio.options = ["10", "100", "1000", "10000"]
            self.options_header.setText("Option - number of trees")
        else:
            options_radio.options = []
            self.options_header.setText("Options")

        if radio_button.isChecked():
            self.classifier_toggled = radio_button.text()

        # Clear the current layout of options_radio
        while options_radio.layout().count():
            child = options_radio.layout().takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            self.options_toggled = ""

        # Add the new radio buttons
        for option in options_radio.options:
            radio_button = QRadioButton(option)
            radio_button.toggled.connect(
                lambda state, rb=radio_button: self.onOptionToggled(state, rb)
            )
            options_radio.layout().addWidget(radio_button)

        # options_radio.layout().addStretch()

    def onDatabaseToggled(self, state, radio_button):
        if radio_button.isChecked():
            self.df = pd.DataFrame()
            self.database_toggled = radio_button.text()
            if self.df.empty:
                self.paths, self.labels = append_paths(self.database_toggled)

                self.df["speech"] = self.paths
                self.df["labels"] = self.labels

    def onOptionToggled(self, state, radio_button):
        if radio_button.isChecked():
            self.options_toggled = radio_button.text()
        else:
            self.options_toggled = ""

    def show_countplot(self):
        if self.database_toggled == "":
            QMessageBox.critical(
                self,
                "Error",
                "Choose database!",
                buttons=QMessageBox.StandardButton.Discard,
                defaultButton=QMessageBox.StandardButton.Discard,
            )
            return
        if self.df.empty:
            self.paths, self.labels = append_paths(self.database_toggled)

            self.df["speech"] = self.paths
            self.df["labels"] = self.labels
        print(self.df["labels"].value_counts())
        dlg = CountPlot(self.df["labels"])
        dlg.exec()

    def save_button_clicked(self):
        print("DB: ", self.database_toggled)
        print("Option: ", self.options_toggled)
        print("Classifier: ", self.classifier_toggled)
        print("Save csv: ", self.checkbox.isChecked())
        if self.database_toggled == "" or self.classifier_toggled == "":
            if self.classifier_toggled != const.KNN and self.options_toggled == "":
                QMessageBox.critical(
                    self,
                    "Error",
                    "Choose all values!",
                    buttons=QMessageBox.StandardButton.Discard,
                    defaultButton=QMessageBox.StandardButton.Discard,
                )
                return
        if self.df.empty:
            print("EMPTY")
            self.paths, self.labels = append_paths(self.database_toggled)
            self.df["speech"] = self.paths
            self.df["labels"] = self.labels

        data_frame_result = create_dataframe_mfcc(self.paths, self.database_toggled)
        if self.checkbox.isChecked():
            save_to_csv(data_frame_result, self.database_toggled)

        y = data_frame_result["emotion"]
        X = data_frame_result.drop("emotion", axis=1)

        if self.classifier_toggled == const.KNN:
            conf_matrix = classify_knn(X, y, self.database_toggled)
            print(conf_matrix)
        elif self.classifier_toggled == const.SVM:
            conf_matrix = classify_svm(
                X, y, float(self.options_toggled), self.database_toggled
            )
            print(conf_matrix)
        elif self.classifier_toggled == const.RANDOM_FOREST:
            conf_matrix = classify_random_forest(
                X, y, int(self.options_toggled), self.database_toggled
            )
            print(conf_matrix)

        dlg = ConfusionMatrix(
            conf_matrix, self.classifier_toggled, self.database_toggled
        )
        dlg.exec()

    def spectrogram_clicked(self):
        if self.database_toggled == "":
            QMessageBox.critical(
                self,
                "Error",
                "Choose database!",
                buttons=QMessageBox.StandardButton.Discard,
                defaultButton=QMessageBox.StandardButton.Discard,
            )
            return
        if self.df.empty:
            self.paths, self.labels = append_paths(self.database_toggled)

            self.df["speech"] = self.paths
            self.df["labels"] = self.labels
        dlg = EmotionDialog(set(self.labels))
        dlg.emotion_chosen.connect(self.emotion_spectrogram)
        dlg.exec()

    def emotion_spectrogram(self, emotion):
        print(f"Emotion choosen: {emotion}")
        path = np.array(self.df["speech"][self.df["labels"] == emotion])[0]
        dlg = EmotionSpectrogramPlot(path, emotion)
        dlg.exec()
