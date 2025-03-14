import numpy as np
from DeepNeuralNetworks import DeepNeuralNetwork
from RecurrentNeuralNetworks import RecurrentNeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

class Data:
    def __init__(self, X, y=None, features=None, label=None):
        if X is None:
            raise ValueError("X cannot be None")
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if features is not None and len(X[0]) != len(features):
            raise ValueError("Number of features must match the number of columns in X")
        if features is None:
            features = [f"feature_{i}" for i in range(X.shape[1])]
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
            
    def summary(self):
        data = {
            "mean": np.nanmean(self.X, axis=0),
            "median": np.nanmedian(self.X, axis=0),
            "min": np.nanmin(self.X, axis=0),
            "max": np.nanmax(self.X, axis=0),
            "var": np.nanvar(self.X, axis=0)
        }
        return pd.DataFrame.from_dict(data, orient="index", columns=self.features)
        
    def train_test_split(self, test_size=0.2, random_state=None):
        """Split data into training and test sets"""
        if not self.has_label():
            raise ValueError("Cannot split dataset without labels")
            
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        train_data = Data(X=X_train, y=y_train, features=self.features, label=self.label)
        test_data = Data(X=X_test, y=y_test, features=self.features, label=self.label)
        
        return train_data, test_data

# Function to read CSV and process data
def read_csv(filename, sep=',', text_column='Text', label_column='Label'):
    """
    Reads a CSV file and converts the text data into a numerical representation using CountVectorizer.
    Uses LabelEncoder for label encoding.
    """
    
    data = pd.read_csv(filename, sep=sep, quotechar='"', on_bad_lines='skip')
    
    # Clean data - handle NaN values in text column
    print(f"Original data shape: {data.shape}")
    
    # Check for and handle NaN values in the text column
    nan_count = data[text_column].isna().sum()
    if nan_count > 0:
        print(f"Found {nan_count} NaN values in '{text_column}' column")
        # Fill NaN values with an empty string
        data[text_column] = data[text_column].fillna('')
    
    # Check for and handle NaN values in the label column
    if data[label_column].isna().sum() > 0:
        print(f"Removing rows with NaN values in '{label_column}' column")
        data = data.dropna(subset=[label_column])
    
    print(f"Cleaned data shape: {data.shape}")
    
    # Use CountVectorizer for Bag of Words transformation
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data[text_column].values).toarray()
    features = vectorizer.get_feature_names_out()
    
    # Use LabelEncoder to convert labels to numerical values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data[label_column].values)
    
    # Store label mapping for interpretation
    label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    
    dataset = Data(X=X, y=y, features=features, label=label_column)
    dataset.label_mapping = label_mapping
    dataset.vectorizer = vectorizer
    
    return dataset , vectorizer

if __name__ == '__main__':
    try:
        # Load dataset
        filename = '../datasets/combined_dataset2.csv'
        print(f"Loading dataset from: {filename}")
        dataset = pd.read_csv(filename)
        
        # Basic dataset info
        print("Dataset shape:", dataset.shape)
        print("Columns:", dataset.columns)

        # Assume dataset has 'text' and 'label' columns
        X_text = dataset['Text'].values
        y_labels = dataset['Label'].values
        
        # Encode labels (0 for human, 1 for AI)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_labels)

        # Convert text to feature vectors using CountVectorizer
        vectorizer = CountVectorizer(binary=True, stop_words='english', max_features=5000)
        X = vectorizer.fit_transform(X_text).toarray()

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Get feature and class counts
        n_features = X_train.shape[1]
        n_classes = len(np.unique(y))

        # ------------------ DNN Model ------------------
        print("\nTraining Deep Neural Network...")
        dnn_layers = [n_features, 128, 64, 32, 1] if n_classes == 2 else [n_features, 128, 64, 32, n_classes]
        dnn = DeepNeuralNetwork(layer_dims=dnn_layers, learning_rate=0.01)
        dnn.fit(X_train, y_train, num_iterations=1000, print_cost=True, print_every=100)

        # Evaluate DNN
        dnn_train_acc = dnn.score(X_train, y_train)
        dnn_test_acc = dnn.score(X_test, y_test)

        print(f"\nDNN Training accuracy: {dnn_train_acc:.4f}")
        print(f"DNN Test accuracy: {dnn_test_acc:.4f}")

        # ------------------ RNN Model ------------------
        print("\nTraining Recurrent Neural Network...")

        # Convert data to RNN-compatible shape
        X_train_rnn = X_train.T  # (features, samples)
        X_test_rnn = X_test.T
        y_train_rnn = np.eye(n_classes)[y_train].T  # One-hot encoding
        y_test_rnn = np.eye(n_classes)[y_test].T

        rnn = RecurrentNeuralNetwork(input_size=n_features, hidden_size=128, output_size=n_classes, learning_rate=0.01)
        rnn.train(X_train_rnn, y_train_rnn, num_epochs=100, print_every=10)

        # Evaluate RNN
        y_pred_train_rnn = rnn.predict(X_train_rnn)
        y_pred_test_rnn = rnn.predict(X_test_rnn)

        rnn_train_acc = np.mean(y_pred_train_rnn == y_train)
        rnn_test_acc = np.mean(y_pred_test_rnn == y_test)

        print(f"\nRNN Training accuracy: {rnn_train_acc:.4f}")
        print(f"RNN Test accuracy: {rnn_test_acc:.4f}")
        


        # ------------------ Model Comparison ------------------
        print("\nModel Performance Comparison:")
        print(f"DNN Test Accuracy: {dnn_test_acc:.4f}")
        print(f"RNN Test Accuracy: {rnn_test_acc:.4f}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
