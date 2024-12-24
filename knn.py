import numpy as np
import math
from collections import defaultdict
from sklearn import datasets

class Knn():
    def __init__(self, n=5, distance="L2", type="voting"):
        self.train_X = None
        self.train_Y = None
        self.n = n
        if distance == "L1":
            self.distance = 1
        elif distance == "L2":
            self.distance = 2
        else:
            raise Exception('Invalid argument: distance only accepts "L1" and "L2"')

        if type == "voting":
            self.type = 1
        elif type == "average":
            self.type = 2
        else:
            raise Exception('Invalid argument: type only accepts "voting" and "average"')

    def _L2(self, x1, x2):
        return math.sqrt(np.sum((x1 - x2) ** 2))

    def _L1(self, x1, x2):
        return np.sum(np.abs(x1 - x2))

    def fit(self, X, Y):
        if isinstance(X, np.ndarray) and isinstance(Y, np.ndarray):
            if X.ndim != 2 or Y.ndim != 1:
                raise self.incorrect_ndim
            if X.shape[0] != Y.shape[0]:
                raise self.len_not_equal
        else:
            raise self.invalid_argument
        if X.shape[0] < self.n:
            raise self.not_enough_samples
        self.X_train = X
        self.Y_train = Y

    def predict(self, X):
        Y_pred = []
        for x_test in X:
            distances = []
            labels = []

            # Finding n closest data points and their corresponding labels
            for x_train, y_train in zip(self.X_train, self.Y_train):
                if self.distance == 1:
                    d = self._L1(x_train, x_test)
                else:
                    d = self._L2(x_train, x_test)

                distances.append(d)
                labels.append(y_train)

            distance_indices = np.argsort(distances)[:self.n]
            n_distances = [distances[i] for i in distance_indices]
            n_labels = [labels[i] for i in distance_indices]

            # Finding the most common label if type is voting
            if self.type == 1:
                unique_labels, count = np.unique(n_labels, return_counts=True)
                Y_pred.append(unique_labels[np.argmax(count)])
            else:
                weight = 1 / (np.array(n_distances) + 0.000001)
                weighted_average = np.dot(n_labels, weight)
                total_weight = np.sum(weight)
                Y_pred.append(weighted_average / total_weight)

        return np.array(Y_pred)