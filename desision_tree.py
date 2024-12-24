import numpy as np
import math
from collections import defaultdict
class Node():
    def __init__(self, feature = None, threshold = None, left = None, right = None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

class DecisionTree():

    ''' Initilize the Decision tree model with parameters. Max depth is the highest size of tree, min_samples_split is the lowest amount of samples that won't be split, ignore_indxs include any features that won't be used by the model.'''
    def __init__(self, max_depth = 100, min_samples_split = 2, ignore_indxs = []):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.ignore_indxs = ignore_indxs
    

    def _entropy(self, Y):
        pt = np.sum(Y == 1) / len(Y)
        pf = np.sum(Y == 0) / len(Y)
        
        if pt == 0 or pf == 0:
            return 0
        
        return -1 * (pt * math.log2(pt) + pf * math.log2(pf))
    
    def _information_gain(self, Y, Y_left, Y_right):
        parent_entropy = self._entropy(Y)
        left_entropy = self._entropy(Y_left)
        right_entropy = self._entropy(Y_right)
        
        children_weighted = len(Y_left) / len(Y) * left_entropy + len(Y_right) / len(Y) * right_entropy
        return parent_entropy - children_weighted

    def _most_common_label(self, Y):
        count = defaultdict(int)
        for i in Y:
            count[i] += 1
        return max(count, key = count.get)


        
    def _grow_tree(self, X, Y, depth = 0):
        num_samples, num_features = X.shape

        #if ending criteria is met, return a leaf node and find its value
        if (depth >= self.max_depth or num_samples < self.min_samples_split) or len(np.unique(Y)) == 1:
            leaf_value = self._most_common_label(Y)
            return Node(value = leaf_value)
        
        #if not a leaf node, find which feature to split on
        max_IG = -1
        best_feature = None
        best_tresh = None
        best_split_indx = None
        for feature in range(num_features):
            tresholds = np.unique(X[:, feature])
            for treshold in tresholds:
                left_indc = X[:, feature] > treshold
                right_indc = X[:, feature] <= treshold
                if len(np.unique(Y[left_indc])) == 1 or len(np.unique(Y[right_indc])) == 1:
                    continue
                Y_left = Y[left_indc]
                Y_right = Y[right_indc]
                IG = self._information_gain(Y, Y_left, Y_right)
                if (IG > max_IG):
                    max_IG = IG
                    best_feature = feature
                    best_tresh = treshold
                    best_split_indx = feature

        #if no split found, return a leaf node with its value again
        if max_IG == -1:
            leaf_value = self._most_common_label(Y)
            return Node(value=leaf_value)
        
        #splitting the data into left and right data
        left_idxs = X[:, best_split_indx] < best_tresh
        right_idxs = X[:, best_split_indx] >= best_tresh

        X_left = X[left_idxs]
        Y_left = Y[left_idxs]

        X_right = X[right_idxs]
        Y_right = Y[right_idxs]

        #initilizing the current node and using recursion to find its left and right nodes
        left_tree, right_tree = None, None
        if len(Y_right > 0):
            right_tree = self._grow_tree(X_right, Y_right, depth = depth + 1)
        if len(Y_left > 0):
            left_tree = self._grow_tree(X_left, Y_left, depth = depth + 1)
        if (left_tree == None and right_tree == None):
            leaf_value = self._most_common_label(Y)
            return Node(value = leaf_value)
        else:
            return Node(feature = best_feature, threshold = best_tresh, left = left_tree, right = right_tree)
        

    ''' Trains the decision tree model with a X 2d features numpy array with a Y 1d labels array, returns the model'''
    def fit(self, X: np.ndarray, Y: np.ndarray):
        if (X.ndim != 2 and Y.ndim != 1):
            raise Exception("Value error: Incorrect dimensions in either X or Y")
        if (X.shape[0] == Y.shape):
            raise Exception("Value error: number of features not equal to number of labels")
        X_copy = np.copy(X)
        X_copy = np.delete(X_copy, self.ignore_indxs, axis=1)
        self.root = self._grow_tree(X_copy, Y, depth = 0)
        return self

    '''Predicts the output of each of the values in the X 2d features np array, returns a 1D array with the predicted labels'''
    def predict(self, X: np.ndarray):
        if (X.ndim != 2):
            raise Exception("Value error: X needs to be a 2d array")
        Y_pred = []
        for x in X:
            node = self.root
            while (not node.is_leaf()):
                feature = node.feature
                thresh = node.threshold

                if x[feature] < thresh:
                    node = node.left
                else:
                    node = node.right
            Y_pred.append(node.value)
        return Y_pred






 


     