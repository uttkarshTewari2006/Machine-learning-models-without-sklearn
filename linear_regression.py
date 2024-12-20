import numpy as np
import math
from collections import defaultdict


class linear_regression():
  def __init__(self, lr = 0.01, n_iter = 1000):
    self.lr = lr
    self.weights = None
    self.bias = None
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

    for i in range(self.n_iter):
        Y_pred = np.dot(X, self.weights) + self.bias
        error_total = Y - Y_pred
        dw = -2 * np.dot(X.T, error_total) / num_samples
        db = -2 * np.sum(error_total) / num_samples
        self.weights -= self.lr * dw
        self.bias -= self.lr * db
    
    return self

  def predict(self, X):
    return np.dot(X, self.weights) + self.bias