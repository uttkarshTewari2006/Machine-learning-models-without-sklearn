import numpy as np
import math

class logistic_regression():
    def __init__(self, lr = 0.001, n_iter = 1000):
        self.lr = lr
        self.n_iter = n_iter
         #exceptions:
        self.len_not_equal = Exception("invalid arguments: number of features and labels not equal")
        self.invalid_argument = Exception("invalid arguments: expected numpy arrays")
        self.incorrect_ndim = Exception("incorrect dimensions: expected feature array with 2 dim and label array with 1 dim")
    
    def fit(self, X, Y):
        if (isinstance(X, np.ndarray) and isinstance(Y, np.ndarray)):
            if (X.ndim != 2 or Y.ndim != 1):
                raise self.incorrect_ndim
            if (X.shape[0] != Y.shape[0]):
                raise self.len_not_equal
        else:
            raise self.invalid_argument
        
        num_samples, num_features = X.shape

        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.n_iter):
            Y_pred = 1 / (1 + np.exp(-np.dot(X, self.weights) - self.bias))

            dw = np.dot(X.T , Y_pred - Y) / num_samples
            db = np.sum(Y_pred - Y) / num_samples

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return self
    
    def predict(self, X):
        if (isinstance(X, np.ndarray)):
            if (X.ndim != 2):
                raise self.incorrect_ndim
        else:
            raise self.invalid_argument
                    
        probabilities = 1 / (1 + np.exp(-np.dot(X, self.weights) - self.bias))
        Y_pred = [probabilities >= 0.5]
        return Y_pred


