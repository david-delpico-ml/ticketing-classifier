import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def plot_training_history(history, string):
    """
    Generate a plot of loss and validation loss for epoch.
    
    Args:
        history: TensorFlow model containing fit results
        string: Variable inside history to show
    
    Returns:
        plot.
    """
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()

def plot_validation_history(history, string):
    """
    Generate a plot of validation F1Score for each class.
    
    Args:
        history: TensorFlow model containing fit results
        string: Variable inside history to show
    
    Returns:
        plot.
    """
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

    val_history = history.history[string]
    plt.plot(val_history)
    plt.xlabel("Epochs")
    plt.ylabel(string)
    if label_names and np.ndim(val_history[0]) > 0 and len(val_history[0]) == len(label_names):
        plt.legend(label_names, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.legend([string], bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

def plot_confusion_matrix(test_dataset, model):
    """
    Generate a plot of confusion matrix and return vectors, predictions and true labels DataFrame.
    
    Args:
        test_dataset: TensorFlow dataset containing test data
        model: Trained model for making predictions
    
    Returns:
        pd.DataFrame: DataFrame with predictions and true labels
    """

    # Get true labels and predictions from the test dataset
    y_true = []
    y_pred = []
    x_text = []

    for batch in test_dataset:
        X_batch, y_batch = batch
        preds = model.predict(X_batch, verbose=0)
        # If y_batch is one-hot encoded, convert to class indices
        if len(y_batch.shape) > 1 and y_batch.shape[1] > 1:
            y_true.extend(np.argmax(y_batch.numpy(), axis=1))
        else:
            y_true.extend(y_batch.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
        x_text.extend(X_batch.numpy())

    # Load label mapping
    label_mapping_path = os.path.join(os.getcwd(), '..', 'data/processed', 'label_mapping.csv')
    label_mapping = pd.read_csv(label_mapping_path)

    cm = confusion_matrix(y_true, y_pred, normalize='true')*100
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_mapping['label'])
    disp.plot(cmap='Blues', values_format='.0f', xticks_rotation=90)

    plt.title("Confusion Matrix")
    plt.show()

    dataset_pred = pd.DataFrame({'x_text':x_text, 'y_pred':y_pred, 'y_true':y_true})

    return dataset_pred

from sklearn.metrics import classification_report
import numpy as np


def plot_classification_report(test_dataset, model):
    """
    Generate a classification report.
    
    Args:
        test_dataset: TensorFlow dataset containing test data
        model: Trained model for making predictions
    
    Returns:
        dict: Classification report as dictionary
    """
    # Get true labels and predictions from the test dataset
    y_true = []
    y_pred = []
    x_text = []

    for batch in test_dataset:
        X_batch, y_batch = batch
        preds = model.predict(X_batch, verbose=0)
        # If y_batch is one-hot encoded, convert to class indices
        if len(y_batch.shape) > 1 and y_batch.shape[1] > 1:
            y_true.extend(np.argmax(y_batch.numpy(), axis=1))
        else:
            y_true.extend(y_batch.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
        x_text.extend(X_batch.numpy())

    # Load label mapping
    label_mapping_path = os.path.join(os.getcwd(), '..', 'data/processed', 'label_mapping.csv')
    label_mapping = pd.read_csv(label_mapping_path)
    
    # Get class labels
    class_labels = label_mapping['label'].tolist()
    
    # Generate classification report
    report_dict = classification_report(
        y_true, 
        y_pred, 
        target_names=class_labels,
        output_dict=True
    )
    # Print detailed classification report
    print("\nDetailed Classification Report:")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_labels))
    
    return pd.DataFrame(report_dict).T