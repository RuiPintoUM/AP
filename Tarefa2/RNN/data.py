import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

class Data:
    def __init__(self, X, y=None, features=None, label=None):
        if X is None:
            raise ValueError("X cannot be None")
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if features is not None and len(X[0]) != len(features):
            raise ValueError("Number of features must match the number of columns in X")
        if features is None:
            features = [f"feature_{i}" for i in range(X.shape[1])]
        if y is not None and label is None:
            label = "y"
        self.X = X
        self.y = y
        self.features = features
        self.label = label

    def shape(self):
        return self.X.shape

    def has_label(self):
        return self.y is not None

    def get_classes(self):
        if self.has_label():
            return np.unique(self.y)
        else:
            raise ValueError("Dataset does not have a label")

    def summary(self):
        data = {
            "mean": np.nanmean(self.X, axis=0),
            "median": np.nanmedian(self.X, axis=0),
            "min": np.nanmin(self.X, axis=0),
            "max": np.nanmax(self.X, axis=0),
            "var": np.nanvar(self.X, axis=0)
        }
        return pd.DataFrame.from_dict(data, orient="index", columns=self.features)

# Updated function to read CSV and process data
def read_csv(filename, sep=',', text_column='Text', label_column='Id', vectorizer=None, sequence_length=10):
    """
    Lê um CSV e converte o texto para uma sequência numérica.
    """
    data = pd.read_csv(filename, sep=sep, quotechar='"', on_bad_lines='skip')

    if vectorizer is None:
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(data[text_column].values).toarray()
    else:
        X = vectorizer.transform(data[text_column].values).toarray()

    # Criar sequências com comprimento fixo
    X_seq = []
    y_seq = []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(data[label_column].iloc[i+sequence_length-1])  # The label for the last element in the sequence

    X_seq = np.array(X_seq)  # (samples, time_steps, features)
    y_seq = np.array(y_seq)  # Adjusted labels for each sequence

    # Processar labels
    label_encoder = LabelEncoder()
    y_seq = label_encoder.fit_transform(y_seq)  # Encode the labels for the sequences

    return Data(X=X_seq, y=y_seq, features=None, label=label_column), vectorizer

# Example usage with the provided dataset
if __name__ == '__main__':
    filename = '../datasets/combined_dataset.csv'
    dataset, vectorizer = read_csv(filename, text_column='Text', label_column='Label')
    print(dataset.shape())
    print("Has label:", dataset.has_label())
    print("Classes:", dataset.get_classes())
    print(dataset.summary())
