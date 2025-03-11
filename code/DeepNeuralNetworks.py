import numpy as np

class DeepNeuralNetwork:
    def __init__(self, layer_dims, learning_rate=0.01):
        """
        Initialize a Deep Neural Network model
        
        Parameters:
            layer_dims: list containing the dimensions of each layer
            learning_rate: learning rate for gradient descent
        """
        self.layer_dims = layer_dims
        self.parameters = {}
        self.learning_rate = learning_rate
        self.L = len(layer_dims) - 1  # number of layers (excluding input)
        self.initialize_parameters()
        
    def initialize_parameters(self):
        """
        Initialize weights and biases for all layers
        """
        np.random.seed(42)
        
        for l in range(1, self.L + 1):
            # He initialization for weights
            self.parameters['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * np.sqrt(2 / self.layer_dims[l-1])
            self.parameters['b' + str(l)] = np.zeros((self.layer_dims[l], 1))
    
    def relu(self, Z):
        """ReLU activation function"""
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z):
        """Derivative of ReLU function"""
        return Z > 0
    
    def sigmoid(self, Z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(Z, -709, 709)))  # Clip to avoid overflow
    
    def sigmoid_derivative(self, Z):
        """Derivative of sigmoid function"""
        sig = self.sigmoid(Z)
        return sig * (1 - sig)
    
    def softmax(self, Z):
        """Softmax activation function for multi-class classification"""
        # Shift values to avoid numerical issues
        shifted_Z = Z - np.max(Z, axis=0, keepdims=True)
        exp_Z = np.exp(shifted_Z)
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    
    def forward_propagation(self, X):
        """
        Forward propagation for L-layer neural network
        
        Parameters:
            X: input data of shape (features, num_examples)
            
        Returns:
            AL: final activation value
            caches: list of tuples containing the intermediate values
        """
        caches = []
        A = X
        
        # Iterate through layers 1 to L-1 (using ReLU)
        for l in range(1, self.L):
            A_prev = A
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]
            
            Z = np.dot(W, A_prev) + b
            A = self.relu(Z)
            
            cache = (A_prev, W, b, Z)
            caches.append(cache)
        
        # Output layer (using sigmoid for binary or softmax for multi-class)
        WL = self.parameters['W' + str(self.L)]
        bL = self.parameters['b' + str(self.L)]
        
        ZL = np.dot(WL, A) + bL
        
        # Use softmax for multi-class (when output dimension > 1)
        if self.layer_dims[-1] > 1:
            AL = self.softmax(ZL)
        else:
            AL = self.sigmoid(ZL)
        
        cache = (A, WL, bL, ZL)
        caches.append(cache)
        
        return AL, caches
    
    def compute_cost(self, AL, Y):
        """
        Compute cost function
        
        Parameters:
            AL: output of the forward propagation
            Y: true labels
            
        Returns:
            cost: cross-entropy cost
        """
        m = Y.shape[1]
        
        # Binary classification (sigmoid output)
        if self.layer_dims[-1] == 1:
            # Binary cross-entropy loss
            cost = -1/m * np.sum(Y * np.log(AL + 1e-9) + (1 - Y) * np.log(1 - AL + 1e-9))
        else:
            # Multi-class categorical cross-entropy loss
            # Y should be one-hot encoded
            cost = -1/m * np.sum(Y * np.log(AL + 1e-9))
        
        return np.squeeze(cost)
    
    def backward_propagation(self, AL, Y, caches):
        """
        Backward propagation for L-layer neural network
        
        Parameters:
            AL: output of the forward propagation
            Y: true labels
            caches: list of caches containing values from forward propagation
            
        Returns:
            gradients: dictionary containing gradients
        """
        gradients = {}
        m = Y.shape[1]
        L = len(caches)
        
        # Initialize backpropagation with output layer
        if self.layer_dims[-1] == 1:
            # Binary classification
            dAL = - (np.divide(Y, AL + 1e-9) - np.divide(1 - Y, 1 - AL + 1e-9))
        else:
            # Multi-class classification (softmax)
            dAL = AL - Y
        
        # Get cache for layer L
        current_cache = caches[L-1]
        A_prev, WL, bL, ZL = current_cache
        
        # Compute gradients for layer L
        if self.layer_dims[-1] == 1:
            dZL = dAL * self.sigmoid_derivative(ZL)
        else:
            # For softmax, dZ = dA
            dZL = dAL
            
        dWL = 1/m * np.dot(dZL, A_prev.T)
        dbL = 1/m * np.sum(dZL, axis=1, keepdims=True)
        dA_prev = np.dot(WL.T, dZL)
        
        gradients["dW" + str(L)] = dWL
        gradients["db" + str(L)] = dbL
        
        # Iterate through layers L-1 down to 1
        for l in reversed(range(1, L)):
            current_cache = caches[l-1]
            A_prev, W, b, Z = current_cache
            
            dZ = dA_prev * self.relu_derivative(Z)
            dW = 1/m * np.dot(dZ, A_prev.T)
            db = 1/m * np.sum(dZ, axis=1, keepdims=True)
            dA_prev = np.dot(W.T, dZ)
            
            gradients["dW" + str(l)] = dW
            gradients["db" + str(l)] = db
        
        return gradients
    
    def update_parameters(self, gradients):
        """
        Update parameters using gradient descent
        
        Parameters:
            gradients: dictionary containing gradients
        """
        for l in range(1, self.L + 1):
            self.parameters["W" + str(l)] -= self.learning_rate * gradients["dW" + str(l)]
            self.parameters["b" + str(l)] -= self.learning_rate * gradients["db" + str(l)]
    
    def fit(self, X, y, num_iterations=3000, batch_size=None, print_cost=True, print_every=100):
        """
        Train the L-layer neural network
        
        Parameters:
            X: input data, shape (num_examples, features)
            y: true labels, shape (num_examples, )
            num_iterations: number of iterations of the optimization loop
            batch_size: size of mini-batches
            print_cost: if True, print the cost every 100 iterations
            
        Returns:
            costs: list of costs
        """
        costs = []
        
        # Prepare data for DNN format (features, examples)
        X = X.T
        
        # Handle different label formats
        if self.layer_dims[-1] == 1:
            # Binary classification, shape (1, examples)
            if len(y.shape) == 1:
                Y = y.reshape(1, -1)
            else:
                Y = y.T
        else:
            # Multi-class, one-hot encode if needed
            if len(y.shape) == 1:
                # Convert to one-hot encoding
                num_classes = self.layer_dims[-1]
                m = y.shape[0]
                Y = np.zeros((num_classes, m))
                Y[y, np.arange(m)] = 1
            else:
                Y = y.T
        
        for i in range(num_iterations):
            # Forward propagation
            AL, caches = self.forward_propagation(X)
            
            # Compute cost
            cost = self.compute_cost(AL, Y)
            
            # Backward propagation
            gradients = self.backward_propagation(AL, Y, caches)
            
            # Update parameters
            self.update_parameters(gradients)
            
            # Print the cost every print_every iterations
            if print_cost and i % print_every == 0:
                print(f"Cost after iteration {i}: {cost}")
                costs.append(cost)
        
        return costs
    
    def predict(self, X):
        """
        Predict using the trained neural network
        
        Parameters:
            X: input data, shape (num_examples, features)
            
        Returns:
            predictions: class predictions
        """
        # Prepare data for DNN format
        X = X.T
        
        # Forward pass
        AL, _ = self.forward_propagation(X)
        
        # Process predictions based on output layer
        if self.layer_dims[-1] == 1:
            # Binary classification
            predictions = (AL > 0.5).astype(int).ravel()
        else:
            # Multi-class classification
            predictions = np.argmax(AL, axis=0)
            
        return predictions
    
    def score(self, X, y):
        """
        Calculate accuracy score
        
        Parameters:
            X: input data, shape (num_examples, features)
            y: true labels
            
        Returns:
            accuracy: proportion of correct predictions
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)