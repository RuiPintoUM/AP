from abc import ABCMeta, abstractmethod
import numpy as np
import copy

class Layer(metaclass=ABCMeta):

    @abstractmethod
    def forward_propagation(self, input):
        raise NotImplementedError
    
    @abstractmethod
    def backward_propagation(self, error):
        raise NotImplementedError
    
    @abstractmethod
    def output_shape(self):
        raise NotImplementedError
    
    @abstractmethod
    def parameters(self):
        raise NotImplementedError
    
    def set_input_shape(self, input_shape):
        self._input_shape = input_shape

    def input_shape(self):
        return self._input_shape
    
    def layer_name(self):
        return self.__class__.__name__


class RecurrentLayer(Layer):
    
    def __init__(self, n_units, input_shape=None):
        super().__init__()
        self.n_units = n_units
        self._input_shape = input_shape
        self.input = None
        self.output = None
        self.weights_x = None  # Pesos da entrada
        self.weights_h = None  # Pesos do estado oculto
        self.biases = None
        self.h_prev = None  # Estado oculto anterior
    
    def initialize(self, optimizer):
        self.weights_x = np.random.rand(self.input_shape()[0], self.n_units) - 0.5
        self.weights_h = np.random.rand(self.n_units, self.n_units) - 0.5
        self.biases = np.zeros((1, self.n_units))
        self.w_x_opt = copy.deepcopy(optimizer)
        self.w_h_opt = copy.deepcopy(optimizer)
        self.b_opt = copy.deepcopy(optimizer)
    
    def forward_propagation(self, inputs, training=True):
        batch_size, time_steps, input_dim = inputs.shape
        self.h_prev = np.zeros((batch_size, self.n_units))
        self.outputs = []

        for t in range(time_steps):
            x_t = inputs[:, t, :]
            self.h_prev = np.tanh(np.dot(x_t, self.weights_x) + np.dot(self.h_prev, self.weights_h) + self.biases)
            self.outputs.append(self.h_prev)

        return np.array(self.outputs).transpose(1, 0, 2)  # (batch, time_steps, units)

    def backward_propagation(self, output_error):
        batch_size, time_steps, _ = output_error.shape
        dW_x, dW_h, dB = np.zeros_like(self.weights_x), np.zeros_like(self.weights_h), np.zeros_like(self.biases)
        dh_next = np.zeros((batch_size, self.n_units))

        for t in reversed(range(time_steps)):
            dht = output_error[:, t, :] + dh_next
            dht_raw = (1 - self.h_prev**2) * dht
            dW_x += np.dot(self.inputs[:, t, :].T, dht_raw)
            dW_h += np.dot(self.h_prev.T, dht_raw)
            dB += np.sum(dht_raw, axis=0, keepdims=True)
            dh_next = np.dot(dht_raw, self.weights_h.T)

        self.weights_x = self.w_x_opt.update(self.weights_x, dW_x)
        self.weights_h = self.w_h_opt.update(self.weights_h, dW_h)
        self.biases = self.b_opt.update(self.biases, dB)

        return dh_next  # Erro para a camada anterior

    def output_shape(self):
        return (self.n_units, )
    
    
class DropoutLayer(Layer):
    
    def __init__(self, drop_rate):
        """
        drop_rate: percentual de neurônios a serem desativados (ex: 0.5 = 50%)
        """
        super().__init__()
        self.drop_rate = drop_rate
        self.mask = None  # Máscara de dropout

    def forward_propagation(self, inputs, training=True):
        if training:
            # Criar máscara de dropout (0 para desativado, 1 para ativado)
            self.mask = np.random.binomial(1, 1 - self.drop_rate, size=inputs.shape)
            return inputs * self.mask  # Aplica dropout
        return inputs  # Na inferência, não aplica dropout

    def backward_propagation(self, output_error):
        return output_error * self.mask  # Propaga apenas os neurônios ativos

    def output_shape(self):
        return self._input_shape

    def parameters(self):
        return 0  # Dropout não tem parâmetros treináveis