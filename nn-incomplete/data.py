#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import re

class Data:
    def __init__(self, X, y = None, features = None, label= None):
        if X is None:
            raise ValueError("X cannot be None")
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if features is not None and len(X[0]) != len(features):
            raise ValueError("Number of features must match the number of columns in X")
        if features is None:
            features = [f"feat_{str(i)}" for i in range(X.shape[1])]
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

    def get_mean(self):
        return np.nanmean(self.X, axis=0)

    def get_variance(self):
        return np.nanvar(self.X, axis=0)

    def get_median(self):
        return np.nanmedian(self.X, axis=0)

    def get_min(self):
        return np.nanmin(self.X, axis=0)

    def get_max(self):
        return np.nanmax(self.X, axis=0)

    def summary(self):
        data = {
            "mean": self.get_mean(),
            "median": self.get_median(),
            "min": self.get_min(),
            "max": self.get_max(),
            "var": self.get_variance()
        }
        return pd.DataFrame.from_dict(data, orient="index", columns=self.features)


def tokenize(text):
    """
    Tokeniza o texto removendo pontuações e separando por espaços.
    Converte tudo para minúsculas para normalização.
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove pontuação
    tokens = text.split()  # Divide o texto em palavras
    return tokens

def build_vocabulary(texts):
    """
    Cria um vocabulário único de todas as palavras nos textos.
    Retorna um dicionário que mapeia cada palavra para um índice.
    """
    vocabulary = {}
    for text in texts:
        tokens = tokenize(text)
        for token in tokens:
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)  # Atribui um índice único a cada palavra
    return vocabulary

def text_to_bow(text, vocabulary):
    """
    Converte um texto em um vetor Bag of Words (BoW) com base no vocabulário.
    """
    tokens = tokenize(text)
    vector = np.zeros(len(vocabulary))
    for token in tokens:
        if token in vocabulary:
            index = vocabulary[token]
            vector[index] += 1  # Conta a frequência da palavra no texto
    return vector

def read_csv(filename, sep=',', text_column='text', label_column='label'):
    """
    Lê um CSV e converte o texto para uma representação numérica usando apenas numpy.
    """
    data = pd.read_csv(filename, sep=sep)

    # Cria o vocabulário com todas as palavras únicas
    vocabulary = build_vocabulary(data[text_column].values)

    # Converte cada texto para um vetor Bag of Words (BoW)
    X = np.array([text_to_bow(text, vocabulary) for text in data[text_column].values])

    # Converte os rótulos para valores numéricos (e.g., 'human' -> 0, 'ai' -> 1)
    y = np.where(data[label_column].values == 'human', 0, 1)

    features = list(vocabulary.keys())

    return Data(X=X, y=y, features=features, label=label_column)


if __name__ == '__main__':
    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([1, 2])
    features = np.array(['a', 'b', 'c'])
    label = 'y'
    dataset = Data(X, y, features, label)
    print(dataset.shape())
    print(dataset.has_label())
    print(dataset.get_classes())
    print(dataset.get_mean())
    print(dataset.get_variance())
    print(dataset.get_median())
    print(dataset.get_min())
    print(dataset.get_max())
    print(dataset.summary())
