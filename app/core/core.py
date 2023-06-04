import core.const as const
import core.emotions as emotion

import numpy as np
import pandas as pd
import os
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import sklearn as sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import (
    cross_val_score,
    cross_validate,
    KFold,
    cross_val_predict,
)
from scipy import signal
from scipy.io import wavfile
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm


def get_label(filename, path):
    if path == const.AGH:
        return filename.split("_")[1]
    elif path == const.TESS:
        label = filename.split("_")[-1]
        return (label.split(".")[0]).lower()
    elif path == const.EMO_DB:
        label = filename.split(".")[0]
        return label[5]
    elif path == const.RAVDESS:
        label = filename.split("-")[2]
        return label


def get_index(path):
    if path == const.AGH:
        return const.agh_index
    elif path == const.EMO_DB:
        return const.emoDB_index
    elif path == const.RAVDESS:
        return const.ravdess_index
    elif path == const.TESS:
        return const.tess_index


def switch(path, filename):
    index = get_index(path)
    if emotion.irony[index] != None and emotion.irony[index] in filename:
        return "irony"
    elif emotion.neutral[index] != None and emotion.neutral[index] in filename:
        return "neutral"
    elif emotion.happy[index] != None and emotion.happy[index] in filename:
        return "happy"
    elif emotion.sad[index] != None and emotion.sad[index] in filename:
        return "sad"
    elif emotion.fear[index] != None and emotion.fear[index] in filename:
        return "fear"
    elif emotion.suprise[index] != None and emotion.suprise[index] in filename:
        return "surprise"
    elif emotion.angry[index] != None and emotion.angry[index] in filename:
        return "angry"
    elif emotion.boredom[index] != None and emotion.boredom[index] in filename:
        return "boredom"
    elif emotion.disgust[index] != None and emotion.disgust[index] in filename:
        return "disgust"
    elif emotion.calm[index] != None and emotion.calm[index] in filename:
        return "calm"


def append_paths(path):
    paths = []
    labels = []
    for dirname, _, filenames in os.walk(const.DIR_PATH + path):
        for filename in filenames:
            paths.append(os.path.join(dirname, filename))
            label = get_label(filename, path)
            labels.append(switch(path, label))
    print("Dataset is loaded")
    return paths, labels


# Spectrograms
def spectogram(data, sample_rate, emotion):
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x), ref=np.max)
    plt.figure(figsize=(10, 4))
    plt.title(emotion, size=20)
    librosa.display.specshow(
        xdb, sr=sample_rate, x_axis="time", y_axis="log", cmap="plasma"
    )
    plt.colorbar(format="%+2.0f dB")
    plt.show()


def melspectogram(data, sample_rate, emotion):
    x = librosa.stft(data)
    S = librosa.feature.melspectrogram(y=data, sr=sample_rate, n_mels=128, fmax=8000)
    fig, ax = plt.subplots()
    plt.figure(figsize=(10, 4))
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(
        S_dB,
        x_axis="time",
        y_axis="mel",
        sr=sample_rate,
        fmax=8000,
        ax=ax,
        cmap="plasma",
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set(title="Mel-frequency spectrogram of " + emotion)
    plt.show()


def show_diagrams(emotion, df):
    path = np.array(df["speech"][df["labels"] == emotion])[0]
    print(path)
    data, sampling_rate = librosa.load(path)
    spectogram(data, sampling_rate, emotion)
    melspectogram(data, sampling_rate, emotion)


# region MFCC ekstrakcja
def extract_var_mfcc(filename):
    y, sr = librosa.load(filename)

    y_filt = librosa.effects.preemphasis(y)
    mfcc = np.var(librosa.feature.mfcc(y=y_filt, sr=sr).T, axis=0)
    return mfcc


def extract_mean_mfcc(filename, path):
    y, sr = librosa.load(filename)
    y_filt = librosa.effects.preemphasis(y)
    mfcc = np.mean(librosa.feature.mfcc(y=y_filt, sr=sr).T, axis=0)

    filename = filename.split("\\")[-1]
    label = get_label(filename, path)
    label = switch(path, label)
    mfcc = np.append(mfcc, label)
    return mfcc


def extract_mfcc_features(paths, curr_path):
    result_arr = []
    for file in paths:
        var_mfcc = extract_var_mfcc(file)
        mean_mfcc = extract_mean_mfcc(file, curr_path)

        value = np.concatenate((var_mfcc, mean_mfcc))

        result_arr.append(value)
    return result_arr


def create_header():
    mean_mfcc_labels = []
    var_mfcc_labels = []
    for i in range(1, 21):
        mean_mfcc_labels.append(f"mean_mfcc_{i}")
        var_mfcc_labels.append(f"var_mfcc_{i}")
    header = np.concatenate((var_mfcc_labels, mean_mfcc_labels))
    header = np.append(header, "emotion")
    return header


def create_dataframe_mfcc(paths, curr_path):
    result = extract_mfcc_features(paths, curr_path)
    header = create_header()
    dataFrame = pd.DataFrame(result, columns=header)
    return dataFrame


# endregion


def save_to_csv(dataframe, curr_path):
    os.makedirs("output/csvData", exist_ok=True)
    current_date = datetime.date.today()
    dataframe.to_csv(
        "output/csvData/" + curr_path + "_" + current_date.strftime("%d_%m_%y") + ".csv"
    )


# region confusion matrix
def get_labels(path):
    if path == const.AGH:
        return emotion.labels_agh
    elif path == const.EMO_DB:
        return emotion.labels_emoDB
    elif path == const.RAVDESS:
        return emotion.labels_ravdess
    elif path == const.TESS:
        return emotion.labels_tess


def get_colors(type):
    match type:
        case const.RANDOM_FOREST:
            return plt.cm.Greens
        case const.SVM:
            return plt.cm.Oranges
        case const.KNN:
            return plt.cm.Blues


def confusion_matrix_plot(conf_mat, type, curr_path):
    matrix = conf_mat
    matrix = matrix.astype("float") / matrix.sum(axis=1)[:, np.newaxis]

    # Build the plot
    plt.figure(figsize=(16, 7))
    sns.set(font_scale=1.4)
    sns.heatmap(
        matrix,
        annot=True,
        annot_kws={"size": 10},
        cmap=get_colors(type),
        linewidths=0.2,
    )

    # Add labels to the plot
    class_names = get_labels(curr_path)
    tick_marks = np.arange(len(class_names))
    tick_marks2 = tick_marks + 0.5
    plt.xticks(tick_marks, class_names, rotation=25)
    plt.yticks(tick_marks2, class_names, rotation=0)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion Matrix for {type} Model")
    plt.show()


# endregion
# region Random Forest
def classify_random_forest(X, y, option, curr_path):
    forest = RandomForestClassifier(n_estimators=option)
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    scores = cross_val_score(forest, X, y, scoring="accuracy", cv=cv)
    print("Accuracy: ", np.mean(scores) * 100, "%")

    y_pred = cross_val_predict(forest, X, y, cv=cv)
    conf_mat = confusion_matrix(y, y_pred, labels=get_labels(curr_path))
    return conf_mat


# endregion


# region knn
def find_best_k(X, y):
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    best_k = 1
    first_mean = 0
    k_range = range(1, 41)
    k_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores_knn = cross_val_score(knn, X, y, cv=cv, scoring="accuracy")
        mean = np.mean(scores_knn)
        if mean > first_mean:
            first_mean = mean
            best_k = k
        k_scores.append(mean)

    print("Best k: ", best_k)
    return best_k


def classify_knn(X, y, curr_path):
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    best_k = find_best_k(X, y)
    knn = KNeighborsClassifier(n_neighbors=best_k)
    y_pred = cross_val_predict(knn, X, y, cv=cv)
    conf_mat = confusion_matrix(y, y_pred, labels=get_labels(curr_path))

    knn_best = KNeighborsClassifier(n_neighbors=best_k)
    scores_knn_best = cross_val_score(knn_best, X, y, cv=cv, scoring="accuracy")
    print("Accuracy: ", np.mean(scores_knn_best) * 100, "%")
    return conf_mat


# endregion


# region svm
def classify_svm(X, y, option, curr_path):
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    svmModel = svm.SVC(kernel="linear", C=option)
    scores = cross_val_score(svmModel, X, y, scoring="accuracy", cv=cv)
    print("Accuracy: ", np.mean(scores) * 100, "%")
    y_pred = cross_val_predict(svmModel, X, y, cv=cv)
    conf_mat = confusion_matrix(y, y_pred, labels=get_labels(curr_path))
    return conf_mat


# endregion
