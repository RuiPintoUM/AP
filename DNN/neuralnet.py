#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pickle as pickle

from layers import DenseLayer
from losses import LossFunction, MeanSquaredError, BinaryCrossEntropy
from optimizer import Optimizer
from metrics import mse


class NeuralNetwork:
 
    def __init__(self, epochs = 100, batch_size = 128, optimizer = None,
                 learning_rate = 0.01, momentum = 0.90, verbose = False, 
                 loss = MeanSquaredError,
                 metric:callable = mse):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = Optimizer(learning_rate=learning_rate, momentum= momentum)
        self.verbose = verbose
        self.loss = loss()
        self.metric = metric

        # attributes
        self.layers = []
        self.history = {}

    def add(self, layer):
        if self.layers:
            layer.set_input_shape(input_shape=self.layers[-1].output_shape())
        if hasattr(layer, 'initialize'):
            layer.initialize(self.optimizer)
        self.layers.append(layer)
        return self

    def get_mini_batches(self, X, y = None,shuffle = True):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        assert self.batch_size <= n_samples, "Batch size cannot be greater than the number of samples"
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, n_samples - self.batch_size + 1, self.batch_size):
            if y is not None:
                yield X[indices[start:start + self.batch_size]], y[indices[start:start + self.batch_size]]
            else:
                yield X[indices[start:start + self.batch_size]], None

    def forward_propagation(self, X, training):
        output = X
        for layer in self.layers:
            output = layer.forward_propagation(output, training)
        return output

    def backward_propagation(self, output_error):
        error = output_error
        for layer in reversed(self.layers):
            error = layer.backward_propagation(error)
        return error

    def fit(self, dataset):
        X = dataset.X
        y = dataset.y
        if np.ndim(y) == 1:
            y = np.expand_dims(y, axis=1)

        self.history = {}
        for epoch in range(1, self.epochs + 1):
            # store mini-batch data for epoch loss and quality metrics calculation
            output_x_ = []
            y_ = []
            for X_batch, y_batch in self.get_mini_batches(X, y):
                # Forward propagation
                output = self.forward_propagation(X_batch, training=True)
                # Backward propagation
                error = self.loss.derivative(y_batch, output)
                self.backward_propagation(error)

                output_x_.append(output)
                y_.append(y_batch)

            output_x_all = np.concatenate(output_x_)
            y_all = np.concatenate(y_)

            # compute loss
            loss = self.loss.loss(y_all, output_x_all)

            if self.metric is not None:
                metric = self.metric(y_all, output_x_all)
                metric_s = f"{self.metric.__name__}: {metric:.4f}"
            else:
                metric_s = "NA"
                metric = 'NA'

            # save loss and metric for each epoch
            self.history[epoch] = {'loss': loss, 'metric': metric}

            if self.verbose:
                print(f"Epoch {epoch}/{self.epochs} - loss: {loss:.4f} - {metric_s}")

        return self

    def predict(self, dataset):
        return self.forward_propagation(dataset.X, training=False)

    def score(self, dataset, predictions):
        if self.metric is not None:
            return self.metric(dataset.y, predictions)
        else:
            raise ValueError("No metric specified for the neural network.")

    def save(self, file_path: str, vectorizer=None):
        """
        Save model weights, biases, and vectorizer using pickle.
        """
        model_data = {
            "weights": [layer.weights for layer in self.layers if hasattr(layer, "weights")],
            "biases": [layer.biases for layer in self.layers if hasattr(layer, "biases")],
            "vectorizer": vectorizer  # Salvar o vectorizer junto
        }
        
        with open(file_path, "wb") as f:
            pickle.dump(model_data, f)
        print(f"Model and vectorizer saved successfully to {file_path}")

    def load(self, file_path: str):
        """
        Load model weights, biases, and vectorizer using pickle.
        Returns the vectorizer as well.
        """
        with open(file_path, "rb") as f:
            model_data = pickle.load(f)
        weights = model_data["weights"]
        biases = model_data["biases"]
        vectorizer = model_data.get("vectorizer")  # Carregar o vectorizer, se existir
        index = 0
        for layer in self.layers:
            if hasattr(layer, "weights"):
                layer.weights = weights[index]
                layer.biases = biases[index]
                index += 1
        print(f"Model loaded successfully from {file_path}")
        return vectorizer

class Dataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        

if __name__ == '__main__':
    from activation import SigmoidActivation, ReLUActivation
    from metrics import mse, accuracy
    from data_v2 import read_csv
    from layers import DropoutLayer
    
    from sklearn.model_selection import train_test_split


    # training data
    dataset_treino , vectorizer = read_csv('../datasets/combined_dataset_treino.csv', 
                                sep=',', 
                                text_column='Text',  # Nome exato da coluna de texto no CSV
                                label_column='Label')  # Nome exato da coluna de rótulos no CSV
    
    #dataset_test , _ = read_csv('../datasets/combined_dataset_test.csv', 
    #                    sep=',', 
    #                    text_column='Text',  # Nome exato da coluna de texto no CSV
    #                    label_column='Label', # Nome exato da coluna de rótulos no CSV
    #                    vectorizer=vectorizer)  

    # Dividir os dados corretamente
    X_train, X_test, y_train, y_test = train_test_split(
        dataset_treino.X, dataset_treino.y, 
        test_size=0.3, random_state=42, stratify=dataset_treino.y
    )

    # Atualizar os datasets corretamente
    dataset_treino = Dataset(X_train, y_train)
    dataset_test = Dataset(X_test, y_test)
    
    # network
    net = NeuralNetwork(epochs=100, batch_size=16, learning_rate=0.005, verbose=True,
                        optimizer=Optimizer(learning_rate=0.005, momentum=0.9, weight_decay=1e-5),
                        loss=BinaryCrossEntropy, metric=accuracy)
    
    n_features = dataset_treino.X.shape[1]
    net.add(DenseLayer(32, (n_features,)))
    net.add(ReLUActivation())
    net.add(DropoutLayer(0.3)) 
    net.add(DenseLayer(16))
    net.add(ReLUActivation())
    net.add(DropoutLayer(0.4))
    net.add(DenseLayer(1))
    net.add(SigmoidActivation())
    
    print(f"Training set shape: {dataset_treino.X.shape}")
    print(f"Test set shape: {dataset_test.X.shape}")

    # train
    net.fit(dataset_treino)
    net.save("../models/dnn.pkl", vectorizer=vectorizer)
    

