import numpy as np
from cs475_types import Predictor
import math
from scipy import stats

class AdaBoost(Predictor):
    def __init__(self, num_iter):
        self.num_iter = num_iter
        # used for selection of h
        self.cache = []
        self.alphas = []
        # store the best h_t for each t
        # including j, c and decision rule
        self.h_t = []

    def h_predict(self, X, j, c, h_rule):
        feat = X[:, j]
        pred = np.where(feat <= c, h_rule[0], h_rule[1])
        return pred
        
    # compute the prediction for a particular h given feature vector and cutoff
    # return the prediction and the decision rule based on the given feature (feat)
    # and the cutoff value c
    def compute_h(self, feat, y, c):
        feat_sorted = np.argsort(feat)
        unique, unique_index, unique_count = np.unique(feat, 
            return_index = True, return_counts = True)
    
        loc = np.where(unique == c)[0]
        # the index of the first occurence of c in the feature vector
        c_index = unique_index[loc]
        # the counts of c in the feature vector
        c_count = unique_count[loc]

        # indices of the two children of the decision stump
        # left is the index of every value <= c
        # right is the index of every value > c
        left, right = np.split(feat_sorted, np.where(feat_sorted == c_index)[0] + c_count)
        # prediction made by the current h
        h_pred = np.empty(y.shape[0])
        # the prediction of every value <= c is based on the most common
        # y among those values
        h_pred_left = stats.mode(y[left], axis = None)[0][0]
        h_pred[left] = h_pred_left
        
        # right could be empty if c is the largest value of the feature vector
        if y[right].shape[0] != 0:
            h_pred_right= stats.mode(y[right], axis = None)[0][0]
            h_pred[right] = h_pred_right
        # just so the return is consistent
        else:
            h_pred_right = h_pred_left

        # used as margin, the cutoff value stored should be half of the current
        # c and the next c
        if loc < len(unique)-1:
            c = .5 * (c + unique[loc+1])
        return (h_pred, h_pred_left, h_pred_right, c)

    # iterate over the dataset, compute and store necessary information for each h
    # including j (feature index), c (cutoff value), h_pred(prediction made)
    # left, right (decision rule)
    def select_h(self, X, y):
        # iterate over features
        for j in range(X.shape[1]):
            feat = X[:, j]
            # iterate over cutoff values
            for c in np.unique(feat):
                h_pred, left, right, c = self.compute_h(feat, y, c)
                # cache a tuple of 3 elements
                # 1. a tuple of (j, c)
                # 2. the prediction made by h
                # 3. decision rule of h
                self.cache.append(((j, c), h_pred, (left, right)))

    def update(self, X, y, D):
        min_error = np.inf
        index = (0, 0) # store j and c of the best h
        h_rule = (0, 0) # the decision rule of the best h
        h_best_pred = np.empty(y.shape[0]) # the prediction made by the best h
        stop = False
        alpha = 0
        D_new = np.empty(D.shape[0]) # new weights

        for h in self.cache:
            (j, c), h_pred, (left, right) = h[0], h[1], h[2]
            error = self.calc_error(h_pred, y, D)
            if error < min_error:
                min_error = error
                index = (j, c)
                h_rule = (left, right)
                h_best_pred = h_pred
        
        # early stopping
        if min_error < 1e-6:
            stop = True
        else: 
            alpha = .5 * math.log((1-min_error) / min_error)
            Z = 0 # normalizing factor
            for i, d in enumerate(D):
                D_new[i] = d * np.exp(-alpha * y[i] * h_best_pred[i])
                Z += D_new[i]
            D_new = np.multiply(D_new, np.full(D.shape[0], 1/Z)) 
        
        return (stop, alpha, index, h_rule, D_new)            

    def calc_error(self, h_pred, y, D):
        error = np.dot(np.where(h_pred != y, 1, 0), D)
        return error
    
    def train(self, X, y):
        X = X.toarray()
        y = np.where(y == 0, -1, 1)
        # initialize weights
        D = np.full(X.shape[0], 1/X.shape[0])
        self.select_h(X, y)
        for i in range(self.num_iter):
            stop, alpha, index, h_rule, D = self.update(X, y, D)
            if stop:
                break
            else:
                self.alphas.append(alpha)
                self.h_t.append((index, h_rule))

    def test(self, X):
        X = X.toarray()
        h_preds = []
        for h in self.h_t:
            j, c = h[0][0], h[0][1]
            h_pred = self.h_predict(X, j, c, h[1])
            h_preds.append(h_pred)
        
        # if adaboost quits in the first iteration, we have no strong classifier
        # so return all 1 for prediction
        if len(h_preds) == 0:
            y_pred = np.full(X.shape[0], 1)
        else:
            h_preds = np.transpose(np.asarray(h_preds))
            self.alphas = np.asarray(self.alphas)
            y_pred = np.dot(h_preds, self.alphas)
            y_pred = np.where(y_pred <= 0, 0, 1)

        return y_pred.reshape(X.shape[0])
