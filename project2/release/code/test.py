import numpy as np
from models import LogisticRegression

# Test apply sigmoid
model = LogisticRegression(0.01, 20)
weights = np.array([-1.5, -2, -3, -4])
X = np.random.rand(10, 4)
#print(model.apply_sigmoid(X, weights))
print(weights.shape)

# Test np.where
a = np.array([.25, 1, 0, .5, .75])
b = np.where(a >= .5, 1, 0)
print(a)
print(b)
