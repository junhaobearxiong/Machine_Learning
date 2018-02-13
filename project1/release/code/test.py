import numpy as np

A = np.ones((2, 4))
B = np.ones((2, 5))

def test_sum_features(X):
	num_examples, num_input_features = X.shape
	y_hat = np.empty([num_examples], dtype = np.int)
	for i in range(num_examples):
		front_sum = np.sum(X[i, 0:int(num_input_features / 2)])
		back_sum = np.sum(X[i, -1:-int(num_input_features / 2) - 1:-1])
		'''
		back_sum = 0
		if num_input_features % 2 == 1: # if num of features is odd
			back_sum = np.sum(X[i, int(num_input_features / 2) + 1:])
		else:
			back_num = np.sum(X[i, int(num_input_features / 2):])
		'''
		print(front_sum)
		print(back_sum)

test_sum_features(A)
test_sum_features(B)
'''
print(A)
print(B)
'''
