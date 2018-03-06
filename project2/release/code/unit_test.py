import numpy as np
from models import LogisticRegression

X = np.matrix([[2, 2, 3], [1, 2, 4], [2, 3, 3], [1, 2, 4]])
Y = np.array([5, 5, 5, 4])


# calculate conditional entropy for one feature
def calc_cond_entropy(x, y):
    cond_ent = 0
    
    for i in np.unique(y):
        index_y = np.where(y == i)
        p_yi = np.shape(index_y)[1] / np.shape(y)[0]

        # elements of the column where the corrsponding y is y_i
        z = x[index_y]
        for j in np.unique(np.array(z)): # unique only takes array not matrix
            index_z = np.where(z == j)
            # condition upon a given y value
            p_xj_cond = np.shape(index_z)[1] / np.shape(index_y)[1]
            p_joint = p_xj_cond * p_yi
            index_x = np.where(x == j)
            # marginal is the probability over the entire x
            p_xj_marg = np.shape(index_x)[1] / np.shape(y)[0]
            cond_ent += p_joint * np.log(p_joint / p_xj_marg)
    
    return -cond_ent

# given the input feature, labels and number of features to select
# return: 1. the index of features in the input matrix
# 2. the input matrix composed of only the selected features
def feature_selection(X, y, num_feat):
    feats = [] # index of features to be selected
    cond_ent = [] # conditional entropy of each features 
    # iterate over columns 
    for i in range(X.shape[1]):
        x = X[:, i]
        cond_ent.append(calc_cond_entropy(x, y))
    
    # since we want to maximize -H(Y|X)
    # equivalently we minimize H(Y|X)
    # so we pick the num_feat # of features that have the smallest 
    # conditional entropy
    feats = np.argsort(cond_ent)[:num_feat]
    return (feats, X[:, feats])


print(feature_selection(X, Y, 2)[1])



