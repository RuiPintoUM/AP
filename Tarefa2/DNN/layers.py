#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    

class DenseLayer (Layer):
    
    def __init__(self, n_units, input_shape = None):
        super().__init__()
        self.n_units = n_units
        self._input_shape = input_shape

        self.input = None
        self.output = None
        self.weights = None
        self.biases = None
        
    def initialize(self, optimizer):
        # initialize weights from a 0 centered uniform distribution [-0.5, 0.5)
        self.weights = np.random.rand(self.input_shape()[0], self.n_units) - 0.5
        # initialize biases to 0
        self.biases = np.zeros((1, self.n_units))
        self.w_opt = copy.deepcopy(optimizer)
        self.b_opt = copy.deepcopy(optimizer)
        return self
    
    def parameters(self):
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

    def forward_propagation(self, inputs, training):
        self.input = inputs
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output
 
    def backward_propagation(self, output_error):
         # computes the layer input error (the output error from the previous layer),
         # dE/dX, to pass on to the previous layer
         # SHAPES: (batch_size, input_columns) = (batch_size, output_columns) * (output_columns, input_columns)
         input_error = np.dot(output_error, self.weights.T)
    
         # computes the weight error: dE/dW = X.T * dE/dY
         # SHAPES: (input_columns, output_columns) = (input_columns, batch_size) * (batch_size, output_columns)
         weights_error = np.dot(self.input.T, output_error)
         
         # computes the bias error: dE/dB = dE/dY
         # SHAPES: (1, output_columns) = SUM over the rows of a matrix of shape (batch_size, output_columns)
         bias_error = np.sum(output_error, axis=0, keepdims=True)
    
         # updates parameters
         self.weights = self.w_opt.update(self.weights, weights_error)
         self.biases = self.b_opt.update(self.biases, bias_error)
         return input_error
 
    def output_shape(self):
         return (self.n_units,) 
     
     
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
        self.inputs = inputs  # Store inputs
        self.h_prev = np.zeros((batch_size, self.n_units))
        self.outputs = []

        for t in range(time_steps):
            x_t = inputs[:, t, :]
            self.h_prev = np.tanh(np.dot(x_t, self.weights_x) + np.dot(self.h_prev, self.weights_h) + self.biases)
            self.outputs.append(self.h_prev)

        output = np.array(self.outputs).transpose(1, 0, 2)
        return output[:, -1, :]  # (batch_size, n_units)

    def backward_propagation(self, output_error):
        batch_size, n_units = output_error.shape  # 2D error: (batch_size, n_units)
        dW_x = np.zeros_like(self.weights_x)
        dW_h = np.zeros_like(self.weights_h)
        dB = np.zeros_like(self.biases)
        dh_next = np.zeros((batch_size, self.n_units))

        # Since we only use the last time step's output, compute gradients based on that
        time_steps = self.inputs.shape[1]
        t = time_steps - 1  # Last time step
        h_prev = self.outputs[t - 1] if t > 0 else np.zeros((batch_size, self.n_units))  # h(t-1)
        h_current = self.outputs[t]  # h(t), the output we used

        # Compute gradients for the last time step
        dht = output_error + dh_next  # Error from output + next layer (if any)
        dht_raw = (1 - h_current**2) * dht  # Derivative of tanh
        dW_x = np.dot(self.inputs[:, t, :].T, dht_raw)  # Gradient w.r.t. input weights
        dW_h = np.dot(h_prev.T, dht_raw)  # Gradient w.r.t. hidden weights
        dB = np.sum(dht_raw, axis=0, keepdims=True)  # Gradient w.r.t. biases
        dh_next = np.dot(dht_raw, self.weights_h.T)  # Error to propagate back

        # Update weights and biases
        self.weights_x = self.w_x_opt.update(self.weights_x, dW_x)
        self.weights_h = self.w_h_opt.update(self.weights_h, dW_h)
        self.biases = self.b_opt.update(self.biases, dB)

        return dh_next  # Error for the previous layer
    def output_shape(self):
        return (self.n_units, )
    
    def parameters(self):
        # Return the total number of trainable parameters
        if self.weights_x is None or self.weights_h is None or self.biases is None:
            return 0  # Before initialization
        return (np.prod(self.weights_x.shape) + 
                np.prod(self.weights_h.shape) + 
                np.prod(self.biases.shape))


    
