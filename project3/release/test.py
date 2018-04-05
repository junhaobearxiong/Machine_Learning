import numpy as np
import math
from scipy import stats

feat = np.array([1, 2, 1, 1, 2])
y = np.array([0, 1, 0, 1, 1])
D = np.full(5, 1/5)

a = np.array([[6, 8, 3, 0],
                [3, 2, 1, 7],
               [8, 1, 8, 4],
               [5, 3, 0, 5],
               [4, 7, 5, 9]])


print(stats.mode(a, axis = 1)[0].reshape(5))

unique, unique_index, unique_count = np.unique(feat, return_index = True, return_counts = True)
feat_sorted = np.argsort(feat)
min_error = np.inf
h_t = None
hs = []
for i, u in enumerate(unique_index):
    c = np.where(feat_sorted == u)[0]
    # indices of the two children of the decision stump
    left, right = np.split(feat_sorted, c + unique_count[i])
    y_pred = np.empty(y.shape[0])
    if y[left].shape[0] != 0:
        y_pred_left = np.argmax(np.bincount(y[left]))
        y_pred[left] = y_pred_left
    if y[right].shape[0] != 0:
        y_pred_right= np.argmax(np.bincount(y[right]))
        y_pred[right] = y_pred_right
    error = np.sum(np.absolute(y_pred - y) * D)
    if error < min_error:
        min_error = error
        h_t = y_pred

hs.append(h_t)
hs = np.asarray(hs)
print(hs)

alpha = .5 * math.log((1-error) / error)
print(1e-6)
