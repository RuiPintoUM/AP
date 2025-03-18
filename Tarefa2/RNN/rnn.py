import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from RNN.layers import RecurrentLayer, DenseLayer, DropoutLayer  # Update import
from activation import SigmoidActivation, ReLUActivation
from metrics import accuracy
from losses import BinaryCrossEntropy
from LogisticRegression import LogisticRegression
from data import read_csv  # Função para ler o CSV
from optimizer import Optimizer  # Make sure the path is correct

import pickle

class RecurrentNeuralNetwork:
    def __init__(self, epochs=100, batch_size=16, optimizer=None, learning_rate=0.01, verbose=False, loss=BinaryCrossEntropy, metric=accuracy):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = Optimizer(learning_rate=learning_rate, momentum=0.9)
        self.verbose = verbose
        self.loss = loss()
        self.metric = metric
        self.layers = []
        self.history = {}

    def add(self, layer):
        if self.layers:
            layer.set_input_shape(input_shape=self.layers[-1].output_shape())
        if hasattr(layer, 'initialize'):
            layer.initialize(self.optimizer)
        self.layers.append(layer)

    def forward_propagation(self, X, training=True):
        output = X
        for layer in self.layers:
            output = layer.forward_propagation(output, training)
        return output

    def backward_propagation(self, output_error):
        error = output_error
        for layer in reversed(self.layers):
            error = layer.backward_propagation(error)
        return error

    def get_mini_batches(self, X, y=None, shuffle=True):
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

    def fit(self, dataset):
        X = dataset.X
        y = dataset.y
        if np.ndim(y) == 1:
            y = np.expand_dims(y, axis=1)  # (samples, 1)

        self.history = {}
        for epoch in range(1, self.epochs + 1):
            output_x_, y_ = [], []
            for X_batch, y_batch in self.get_mini_batches(X, y):
                output = self.forward_propagation(X_batch, training=True)  # (batch_size, 1)
                error = self.loss.derivative(y_batch, output)
                self.backward_propagation(error)

                output_x_.append(output)
                y_.append(y_batch)

            output_x_all = np.concatenate(output_x_)  # (n_samples, 1)
            y_all = np.concatenate(y_)  # (n_samples, 1)

            loss = self.loss.loss(y_all, output_x_all)
            metric = self.metric(y_all, output_x_all) if self.metric else "NA"

            self.history[epoch] = {'loss': loss, 'metric': metric}

            if self.verbose:
                print(f"Epoch {epoch}/{self.epochs} - loss: {loss:.4f} - accuracy: {metric:.4f}")

        return self

    def predict(self, dataset):
        return self.forward_propagation(dataset.X, training=False)

    def score(self, dataset, predictions):
        if self.metric:
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


if __name__ == '__main__':
    # Carregar dados
    dataset_treino, vectorizer = read_csv('../datasets/ds/balanced_output.csv', text_column='Text', label_column='Label')
    #dataset_test, _ = read_csv('../datasets/dataset2_inputs.csv', vectorizer=vectorizer)

    # Criar modelo RNN
    rnn = RecurrentNeuralNetwork(epochs=100, batch_size=16, learning_rate=0.005, verbose=True)

    # Adicionar camadas RNN
    n_features = dataset_treino.X.shape[2]  # (samples, time_steps, features)
    rnn.add(RecurrentLayer(32, (n_features,)))  # Camada recorrente
    rnn.add(DropoutLayer(0.3))
    rnn.add(DenseLayer(16))  # Camada densa
    rnn.add(ReLUActivation())  
    rnn.add(DropoutLayer(0.4))
    rnn.add(DenseLayer(1))  # Saída final
    rnn.add(SigmoidActivation())  

    # Treinar o modelo
    rnn.fit(dataset_treino)
    
    # Obter representações da RNN
    rnn_train_output = rnn.predict(dataset_treino)
    #rnn_test_output = rnn.predict(dataset_test)

    # Treinar o modelo de regressão logística usando as representações da RNN
    logistic_regression = LogisticRegression(learning_rate=0.01, num_iterations=1000)
    logistic_regression.fit(rnn_train_output, dataset_treino.y)

    # Testar o modelo de regressão logística
    #logistic_predictions = logistic_regression.predict(rnn_test_output)
    #logistic_score = logistic_regression.score(rnn_test_output, dataset_test.y)

    #print(f"Score da regressão logística: {logistic_score}")

    # Testar o modelo
    #out = rnn.predict(dataset_test)
    #print(f"Score: {rnn.score(dataset_test, out)}")
    
    # Salvar o modelo
    rnn.save("../models/rnn.pkl", vectorizer=vectorizer)
