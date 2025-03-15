#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class Optimizer:

    def __init__(self, learning_rate=0.01, momentum=0.90, weight_decay=0.0):
        self.retained_gradient = None
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay  # Adiciona o weight decay

    def update(self, w, grad_loss_w):
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros(np.shape(w))

        # Atualiza gradiente com momentum
        self.retained_gradient = self.momentum * self.retained_gradient + (1 - self.momentum) * grad_loss_w

        # Aplica regularização L2 (Weight Decay)
        if self.weight_decay > 0:
            w = w - self.weight_decay * w  # Penaliza pesos grandes
        
        return w - self.learning_rate * self.retained_gradient
