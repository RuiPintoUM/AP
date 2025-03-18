import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -709, 709)))
    
    def fit(self, X, y, print_cost=False, print_every=100):
        """
        Train logistic regression model
        
        Parameters:
            X: Training examples, shape (n_examples, n_features)
            y: Labels, shape (n_examples, )
        """
        # Initialize parameters
        n_features = X.shape[1]
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
        costs = []

        # Reshape y for matrix multiplication
        Y = y.reshape(-1, 1)
        
        # Gradient descent
        for i in range(self.num_iterations):
            # Forward pass
            Z = np.dot(X, self.weights) + self.bias
            A = self.sigmoid(Z)
            
            # Calculate cost
            m = X.shape[0]
            cost = -1/m * np.sum(Y * np.log(A + 1e-9) + (1 - Y) * np.log(1 - A + 1e-9))
            
            # Backward pass (compute gradients)
            dw = 1/m * np.dot(X.T, (A - Y))
            db = 1/m * np.sum(A - Y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Print cost
            if print_cost and i % print_every == 0:
                print(f"Cost after iteration {i}: {cost}")
                costs.append(cost)
        
        return costs
    
    def predict_proba(self, X):
        """Predict probability of class 1"""
        Z = np.dot(X, self.weights) + self.bias
        A = self.sigmoid(Z)
        return A
    
    def predict(self, X):
        """Predict class labels (0 or 1)"""
        return (self.predict_proba(X) > 0.5).astype(int).reshape(-1)
    
    def score(self, X, y):
        """Calculate accuracy score"""
        predictions = self.predict(X)
        return np.mean(predictions == y)