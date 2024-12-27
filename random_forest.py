import numpy as np
from desision_tree import DecisionTree
from typing import Literal
class RandomForest():

    def __init__(self, n_trees : int = 4, min_sample_split : int = 2, sample_percent : float = 0.6, max_depth: int = 3, ignore_index: list = [], type: Literal["voting", "average"] = "voting"):
        if (sample_percent <= 0 or sample_percent > 1):
            raise ValueError("sample_percent must be in the range (0.0, 1.0]")
        self.n_trees = n_trees
        self.min_sample_split = min_sample_split
        self.ignore_index = ignore_index
        self.max_depth = max_depth
        self.trees = []
        self.type = type
        self.sample_percent = sample_percent
        
    def _get_samples(self, X, Y, n_samples):
        indc = np.random.choice(len(X), n_samples, replace = True)
        return X[indc], Y[indc]

    def fit(self, X : np.ndarray, Y : np.ndarray):
        num_samples = Y.shape[0]
        for i in range(self.n_trees):
            dt = DecisionTree(max_depth = self.max_depth, min_samples_split= self.min_sample_split, ignore_indxs = self.ignore_index)
            X_sample, Y_sample = self._get_samples(X, Y, int(self.sample_percent * num_samples))
            dt.fit(X_sample, Y_sample)
            self.trees.append(dt)


    def predict(self, X : np.ndarray):
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        if self.type == "voting":
            return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr = tree_predictions)
        else:
            return np.mean(tree_predictions, axis=0)
        








    