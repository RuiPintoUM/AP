import numpy as np
from layers import RecurrentLayer, DenseLayer
from losses import BinaryCrossEntropy
from optimizer import Optimizer
from metrics import accuracy

class FlattenLayer:
    def set_input_shape(self, input_shape):
        # input_shape aqui representa o shape de uma amostra (ex: (time_steps, features))
        self.input_shape = input_shape
    
    def forward_propagation(self, X, training=True):
        # X possui shape (batch, ...) e retornamos um tensor 2D
        return X.reshape(X.shape[0], -1)
    
    def backward_propagation(self, error):
        # Reshape para o formato original (mantendo o batch)
        return error.reshape((error.shape[0],) + self.input_shape)
    
    def output_shape(self):
        # Retorna o shape de uma amostra após o flatten (sem o batch)
        return (np.prod(self.input_shape),)

class RecurrentNeuralNetwork:
    def __init__(self, epochs=100, batch_size=16, learning_rate=0.01, verbose=False, loss=BinaryCrossEntropy, metric=accuracy):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = Optimizer(learning_rate=learning_rate, momentum=0.9)
        self.verbose = verbose
        self.loss = loss()
        self.metric = metric
        self.layers = []
        self.history = {}

    def add(self, layer):
        # Se a camada tiver o método 'set_input_shape', usamos-o para definir o shape de entrada com base na camada anterior.
        if self.layers and hasattr(layer, 'set_input_shape'):
            layer.set_input_shape(self.layers[-1].output_shape())
        if hasattr(layer, 'initialize'):
            layer.initialize(self.optimizer)
        self.layers.append(layer)

    def forward_propagation(self, X, training=True):
        output = X
        for layer in self.layers:
            output = layer.forward_propagation(output, training)
        return output

    def backward_propagation(self, error):
        for layer in reversed(self.layers):
            error = layer.backward_propagation(error)

    def get_mini_batches(self, X, y=None, shuffle=True):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            if y is not None:
                yield X[indices[start:end]], y[indices[start:end]]
            else:
                yield X[indices[start:end]], None

    def fit(self, dataset):
        X = dataset.X
        y = dataset.y
        if np.ndim(y) == 1:
            y = np.expand_dims(y, axis=1)

        self.history = {}
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0
            epoch_metric = 0
            batch_count = 0

            for X_batch, y_batch in self.get_mini_batches(X, y):
                output = self.forward_propagation(X_batch, training=True)
                # A saída já está achatada pela FlattenLayer
                final_output = output
                
                batch_loss = self.loss.loss(y_batch, final_output)
                batch_metric = self.metric(y_batch, final_output)
                epoch_loss += batch_loss
                epoch_metric += batch_metric

                error = self.loss.derivative(y_batch, final_output)
                self.backward_propagation(error)
                batch_count += 1

            avg_loss = epoch_loss / batch_count
            avg_metric = epoch_metric / batch_count
            self.history[epoch] = {'loss': avg_loss, 'metric': avg_metric}

            if self.verbose:
                print(f"Epoch {epoch}/{self.epochs} - loss: {avg_loss:.4f} - accuracy: {avg_metric:.4f}")

        return self

    def predict(self, dataset):
        output = self.forward_propagation(dataset.X, training=False)
        return output

    def score(self, dataset, predictions):
        if self.metric:
            return self.metric(dataset.y, predictions)
        else:
            raise ValueError("No metric specified for the neural network.")

if __name__ == '__main__':
    from activation import SigmoidActivation, ReLUActivation
    from metrics import accuracy
    from data import read_csv

    # Carrega os datasets de treino e teste
    dataset_train, vectorizer = read_csv('../datasets/combined_dataset_treino.csv', text_column='Text', label_column='Label')
    dataset_test, _ = read_csv('../datasets/combined_dataset_test.csv', text_column='Text', label_column='Label', vectorizer=vectorizer)

    # Cria o modelo RNN
    rnn = RecurrentNeuralNetwork(epochs=50, batch_size=16, learning_rate=0.005, verbose=True)

    # Supondo que dataset_train.X tenha shape: (batch, time_steps, features)
    n_features = dataset_train.X.shape[2]

    # Monta a arquitetura: camada recorrente -> flatten -> densas
    rnn.add(RecurrentLayer(32, input_shape=(n_features,)))
    rnn.add(FlattenLayer())
    rnn.add(DenseLayer(16))
    rnn.add(ReLUActivation())
    rnn.add(DenseLayer(1))
    rnn.add(SigmoidActivation())

    # Treina o modelo
    rnn.fit(dataset_train)

    # Faz predições e avalia
    predictions = rnn.predict(dataset_test)
    score = rnn.score(dataset_test, predictions)
    print(f"Score: {score}")