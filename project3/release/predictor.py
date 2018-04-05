import numpy as np
from cs475_types import Predictor


class AdaBoost(Predictor):
    def __init__(self, num_iter):
        self.num_iter = num_iter
        self.weights = None

    def select_h(self, X, y, W):
        X = X.toarray()
        for i in range(X.shape[1]):
            compute_error(X[:, i], y, W)
            break

    def compute_error(self, feat, y, W):
        unique, unique_index, unique_count = np.unique(feat, return_index = True, return_counts = True)
        feat_sorted = np.argsort(feat)
        # index of different cutoff value
        # indices are the first occurences of the unique values
        for i in range(unique_index):
            c = np.where(feat_sorted == unique_index[i])[0]
            # indices of the two children of the decision stump
            left, right = np.split(feat_sorted, c + unique_count[i])
            np.maximum(y[left])
            
            

    def train(self, X, y):
        # initialize weights
        self.weights = np.full(X.shape[0], 1/X.shape[0])
        self.select_h(X, y, self.weights)


    def test(self, X):
        pass
