import numpy as np

feat = np.array([1, 2, 1, 1, 2])
y = np.array([0, 1, 0, 1, 1])
W = np.full(5, 1/5)

unique, unique_index, unique_count = np.unique(feat, return_index = True, return_counts = True)
feat_sorted = np.argsort(feat)
# index of different cutoff value
# indices are the first occurences of the unique values
for i, u in enumerate(unique_index):
    c = np.where(feat_sorted == u)[0]

    # indices of the two children of the decision stump
    left, right = np.split(feat_sorted, c + unique_count[i])
    y_pred_left = np.argmax(np.bincount(y[left]))
    y_pred_right= np.argmax(np.bincount(y[right]))
    y_pred = np.empty(y.shape[0])
    y_pred[left] = y_pred_left
    y_pred[right] = y_pred_right
    error = np.sum(np.absolute(y_pred - y) * W)
    # also calculate new W and alpha
    break
