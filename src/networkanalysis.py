from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


def generate_confusion_matrix(true_values, prediction_values, cfm_title):
    cfm = confusion_matrix(true_values, prediction_values)
    disp = ConfusionMatrixDisplay(confusion_matrix=cfm)
    disp.plot()
    plt.title(cfm_title)
    plt.show()


def generate_classifier_report(true_values, prediction_values):
    target_names = ['class 0', 'class 1', 'class 2']
    cr = classification_report(true_values, prediction_values, target_names=target_names)
    return cr


def run_analysis(network_runs):
    for network_run in network_runs:
        z_classes = network_run[0]
        n_shots = network_run[1]
        k_neighbors = network_run[2]
        predictions = network_run[3]
        true_labels = network_run[4]
        agree_labels = network_run[5]
        matrix_title = str(z_classes) + " Classes with " + str(n_shots) + "-shots and " + str(k_neighbors) + "-neighbors"
        generate_confusion_matrix(true_labels, predictions, matrix_title)
        cr = generate_classifier_report(true_labels, predictions)
