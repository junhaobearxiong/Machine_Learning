import numpy as np
from cs475_types import Predictor
import math
from scipy import stats

class AdaBoost(Predictor):
    def __init__(self, num_iter):
        self.num_iter = num_iter
        self.alphas = np.empty((self.num_iter, 1))
        self.hs = []

    def h_predict(self, X, j, c, h_rule):
        feat = X[:, j]
        pred = np.where(feat <= c, h_rule[0], h_rule[1])
        return pred

    # return the prediction and the decision rule based on the given feature (feat)
    # and the cutoff value c
    def select_h(self, feat, y, c):
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
        left, right = np.split(feat_sorted, c_index + c_count)
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
        
        return (h_pred, h_pred_left, h_pred_right)

    def calc_error(self, h_pred, y, D):
        error = np.dot(np.where(h_pred != y, 1, 0), D)
        return error
        

    def update(self, X, y, D):
        min_error = np.inf
        h_t = (0, 0) # store j and c of the best h
        h_rule = None # the decision rule of the best h
        h_best_pred = None # the prediction made by the best h

        # iterate over features
        for j in range(X.shape[1]):
            feat = X[:, j]
            # iterate over cutoff values
            for c in np.unique(feat):
                h_pred, left, right = self.select_h(feat, y, c)
                #TODO each iteration everything not using D should be the same
                error = self.calc_error(h_pred, y, D)
                if error < min_error:
                    min_error = error
                    h_t = (j, c)
                    h_rule = (left, right)
                    h_best_pred = h_pred
                
        alpha = .5 * math.log((1-min_error) / min_error)
        D_new = np.empty(D.shape[0]) # new weights
        Z = 0 # normalizing factor
        for i, d in enumerate(D):
            D_new[i] = d * np.exp(-alpha * y[i] * h_best_pred[i])
            Z += D_new[i]
        D_new = np.multiply(D_new, np.full(D.shape[0], 1/Z)) 
        return (alpha, h_t, h_rule, D_new)            

    def train(self, X, y):
        X = X.toarray()
        y = np.where(y == 0, -1, 1)
        # initialize weights
        D = np.full(X.shape[0], 1/X.shape[0])
        for i in range(self.num_iter):
            alpha, h_t, h_rule, D = self.update(X, y, D)
            self.alphas[i] = alpha
            self.hs.append((h_t, h_rule))

    def test(self, X):
        X = X.toarray()
        h_preds = []
        for h in self.hs:
            j, c = h[0][0], h[0][1]
            h_pred = self.h_predict(X, j, c, h[1])
            h_preds.append(h_pred)
       
        h_preds = np.transpose(np.asarray(h_preds))
        y_pred = np.dot(h_preds, self.alphas)
        y_pred = np.where(y_pred <= 0, 0, 1)

        return y_pred.reshape(X.shape[0])
