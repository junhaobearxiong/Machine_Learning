import numpy as np

X = np.array([[0, 2, 0], [0, 4, 0]])
index = np.where(~X.any(axis = 0))[0]
print(X)
X = np.delete(X, index, axis = 1)
print(X)
print(np.reciprocal(np.array([0, 2])))
