import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_training_history(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()

def plot_validation_history(history, string):
    # Determine the plot size
    plt.figure(figsize=(12, 6))
    # Try to load label names for the legend
    label_names = None
    try:
        label_mapping_path = os.path.join(os.getcwd(), '..', 'data/processed', 'label_mapping.csv')
        label_mapping = pd.read_csv(label_mapping_path)
        label_names = label_mapping['label'].tolist()
    except FileNotFoundError:
        pass

    val_key = 'val_' + string
    val_history = history.history[val_key]
    plt.plot(val_history)
    plt.xlabel("Epochs")
    plt.ylabel(val_key)
    if label_names and np.ndim(val_history[0]) > 0 and len(val_history[0]) == len(label_names):
        plt.legend(label_names)
    else:
        plt.legend([val_key])
    plt.show()

def plot_confusion_matrix(test_dataset, model):
    # Get true labels and predictions from the test dataset
    y_true = []
    y_pred = []

    for batch in test_dataset:
        X_batch, y_batch = batch
        preds = model.predict(X_batch)
        y_true.extend(y_batch.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    # Use the labels from label_mapping.csv if available in x and y axis
    try:
        label_mapping_path = os.path.join(os.getcwd(), '..', 'data/processed', 'label_mapping.csv')
        label_mapping = pd.read_csv(label_mapping_path)
        plt.xticks(ticks=np.arange(len(label_mapping)), labels=label_mapping['label'], rotation=90)
        plt.yticks(ticks=np.arange(len(label_mapping)), labels=label_mapping['label'])
    except FileNotFoundError:
        print("Label mapping file not found.")

    plt.title("Confusion Matrix")
    plt.show()