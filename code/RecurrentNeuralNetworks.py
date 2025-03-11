import numpy as np

class RecurrentNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.initialize_parameters()
    
    def initialize_parameters(self):
        np.random.seed(42)
        self.Wax = np.random.randn(self.hidden_size, self.input_size) * 0.01
        self.Waa = np.random.randn(self.hidden_size, self.hidden_size) * 0.01
        self.Wya = np.random.randn(self.output_size, self.hidden_size) * 0.01
        self.ba = np.zeros((self.hidden_size, 1))
        self.by = np.zeros((self.output_size, 1))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def softmax(self, x):
        shifted_x = x - np.max(x, axis=0, keepdims=True)
        exp_x = np.exp(shifted_x)
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    def forward_step(self, x_t, a_prev):
        a_next = self.tanh(np.dot(self.Wax, x_t) + np.dot(self.Waa, a_prev) + self.ba)
        z = np.dot(self.Wya, a_next) + self.by
        y_pred = self.softmax(z)
        return a_next, y_pred, (a_next, a_prev, x_t, y_pred)
    
    def forward(self, X):
        sequence_length = X.shape[1]
        a = np.zeros((self.hidden_size, sequence_length + 1))
        y_preds = np.zeros((self.output_size, sequence_length))
        caches = []
        a[:, 0] = np.zeros((self.hidden_size,))
        for t in range(sequence_length):
            x_t = X[:, t].reshape(-1, 1)
            a_prev = a[:, t].reshape(-1, 1)
            a_next, y_pred, cache = self.forward_step(x_t, a_prev)
            a[:, t+1] = a_next.reshape(-1)
            y_preds[:, t] = y_pred.reshape(-1)
            caches.append(cache)
        return y_preds, caches, a
    
    def compute_loss(self, y_preds, Y):
        loss = -np.sum(Y * np.log(y_preds + 1e-9)) / Y.shape[1]
        return loss
    
    def backward_step(self, dy, cache):
        a_next, a_prev, x_t, y_pred = cache
        dz = dy
        dWya = np.dot(dz, a_next.T)
        dby = dz
        da_next = np.dot(self.Wya.T, dz)
        dtanh = (1 - a_next ** 2) * da_next
        dWax = np.dot(dtanh, x_t.T)
        dWaa = np.dot(dtanh, a_prev.T)
        dba = dtanh
        da_prev = np.dot(self.Waa.T, dtanh)
        return (dWax, dWaa, dWya, dba, dby), da_prev
    
    def backward(self, Y, y_preds, caches, a):
        dWax = np.zeros_like(self.Wax)
        dWaa = np.zeros_like(self.Waa)
        dWya = np.zeros_like(self.Wya)
        dba = np.zeros_like(self.ba)
        dby = np.zeros_like(self.by)
        da_next = np.zeros((self.hidden_size, 1))
        for t in reversed(range(Y.shape[1])):
            dy = y_preds[:, t].reshape(-1, 1) - Y[:, t].reshape(-1, 1)
            gradients, da_next = self.backward_step(dy, caches[t])
            dWax += gradients[0]
            dWaa += gradients[1]
            dWya += gradients[2]
            dba += gradients[3]
            dby += gradients[4]
        for gradient in [dWax, dWaa, dWya, dba, dby]:
            np.clip(gradient, -5, 5, out=gradient)
        return {"dWax": dWax, "dWaa": dWaa, "dWya": dWya, "dba": dba, "dby": dby}
    
    def update_parameters(self, gradients):
        self.Wax -= self.learning_rate * gradients["dWax"]
        self.Waa -= self.learning_rate * gradients["dWaa"]
        self.Wya -= self.learning_rate * gradients["dWya"]
        self.ba -= self.learning_rate * gradients["dba"]
        self.by -= self.learning_rate * gradients["dby"]
    
    def train(self, X, Y, num_epochs=100, print_every=10):
        for epoch in range(num_epochs):
            y_preds, caches, a = self.forward(X)
            cost = self.compute_loss(y_preds, Y)
            gradients = self.backward(Y, y_preds, caches, a)
            self.update_parameters(gradients)
            if epoch % print_every == 0:
                print(f"Cost after epoch {epoch}: {cost}")
    
    def predict(self, X):
        y_preds, _, _ = self.forward(X)
        predictions = np.argmax(y_preds, axis=0)
        return predictions
