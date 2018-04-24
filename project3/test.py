import numpy as np
import math
from scipy import stats

x = np.array([0, 1, -1])
y = np.array([1, -1, -1])
def h1(x): return np.where(x > 1/2, 1, -1)
def h2(x): return np.where(x > -1/2, 1, -1)
def h3(x): return 1
D = np.full(3, 1/3)
hs = [h1, h2, h3]

for t in range(2):
    print('iteration {}'.format(t))
    min_error = np.inf
    h_best_pred = None
    h_t = None
    for i, h in enumerate(hs):
        h_pred = h(x)
        error = np.dot(np.where(h_pred != y, 1, 0), D)
        if error < min_error:
            min_error = error
            h_best_pred = h_pred
            h_t = i

    print('h_t: {}'.format(i))
    print('min_error: {}'.format(min_error))
    alpha = .5 * math.log((1-min_error) / min_error)
    print('alpha: {}'.format(alpha))
    Z = 0 # normalizing factor
    for i, d in enumerate(D):
        D[i] = d * np.exp(-alpha * y[i] * h_best_pred[i])
        Z += D[i]
    D = np.multiply(D, np.full(D.shape[0], 1/Z))
    print('D: {}'.format(D))
