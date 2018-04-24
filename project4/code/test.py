import numpy as np

X = np.array([[1, 2], [3, 4]])
print(X)
X[:, 1] = .5 * X[:, 1]
print(X)
