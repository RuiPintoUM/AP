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
def read_csv(filename, sep=',', text_column='Text', label_column='Label', vectorizer=None):
    """
    Reads a CSV file and converts the text data into a numerical representation using CountVectorizer.
    Uses LabelEncoder for label encoding.
    """
    data = pd.read_csv(filename, sep=sep, quotechar='"', on_bad_lines='skip')

    if vectorizer is None:
        # Apenas no treino: Criar e ajustar o vocabulário
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(data[text_column].values).toarray()
        features = vectorizer.get_feature_names_out()
        
    else:
        # No teste: Apenas transformar os dados usando vocabulário do treino
        X = vectorizer.transform(data[text_column].values).toarray()
        features = vectorizer.get_feature_names_out()

    # Use LabelEncoder to convert labels to numerical values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data[label_column].values)

    return Data(X=X, y=y, features=features, label=label_column) , vectorizer

# Example usage with the provided dataset
if __name__ == '__main__':
    filename = '../datasets/combined_dataset.csv'
    dataset = read_csv(filename)
    print(dataset.shape())
    print("Has label:", dataset.has_label())
    print("Classes:", dataset.get_classes())
    print(dataset.summary())
