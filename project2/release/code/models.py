import numpy as np
import scipy as sp
from scipy.special import expit

class Model(object):

    def __init__(self):
        self.num_input_features = None

    def fit(self, X, y):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()

    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()


class Useless(Model):

    def __init__(self):
        super().__init__()
        self.reference_example = None
        self.reference_label = None

    def fit(self, X, y):
        self.num_input_features = X.shape[1]
        # Designate the first training example as the 'reference' example
        # It's shape is [1, num_features]
        self.reference_example = X[0, :]
        # Designate the first training label as the 'reference' label
        self.reference_label = y[0]
        self.opposite_label = 1 - self.reference_label

    def predict(self, X):
        if self.num_input_features is None:
            raise Exception('fit must be called before predict.')
        # Perhaps fewer features are seen at test time than train time, in
        # which case X.shape[1] < self.num_input_features. If this is the case,
        # we can simply 'grow' the rows of X with zeros. (The copy isn't
        # necessary here; it's just a simple way to avoid modifying the
        # argument X.)
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        # Or perhaps more features are seen at test time, in which case we will
        # simply ignore them.
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        # Compute the dot products between the reference example and X examples
        # The element-wise multiply relies on broadcasting; here, it's as if we first
        # replicate the reference example over rows to form a [num_examples, num_input_features]
        # array, but it's done more efficiently. This forms a [num_examples, num_input_features]
        # sparse matrix, which we then sum over axis 1.
        dot_products = X.multiply(self.reference_example).sum(axis=1)
        # dot_products is now a [num_examples, 1] dense matrix. We'll turn it into a
        # 1-D array with shape [num_examples], to be consistent with our desired predictions.
        dot_products = np.asarray(dot_products).flatten()
        # If positive, return the same label; otherwise return the opposite label.
        same_label_mask = dot_products >= 0
        opposite_label_mask = ~same_label_mask
        y_hat = np.empty([num_examples], dtype=np.int)
        y_hat[same_label_mask] = self.reference_label
        y_hat[opposite_label_mask] = self.opposite_label
        return y_hat


class LogisticRegression(Model):

    def __init__(self, eta, num_iter):
        super().__init__()
        self.eta = eta # learning rate, default to .01
        self.num_iter = num_iter # number of GD iterations, default to 20

    # vectorize the gradient update
    # we get partial_loss wrt w = transpose(X) * (y - mu)
    # where mu = sigmoid(X * w) where sigmoid is applied component wise
    def gradient_ascent(self, w, X, y):
        w_prime = w # define the new weights that we are going to update
        
        for i in range(self.num_iter): # perform GD for num_iter iterations
            w_prime = np.reshape(w_prime, (w_prime.shape[0], 1))
            # expit(x) is defined as expit(x) = 1/(1+exp(-x))
            mu = sp.special.expit(np.dot(X, w_prime))
            gradient = np.dot(np.transpose(X), np.subtract(y, mu))
            # update the weights 
            w_prime = np.add(w_prime, np.multiply(self.eta, gradient))         
            return w_prime

    def fit(self, X, y):
        num_examples, num_input_features = X.shape
        self.num_input_features = num_input_features # for test time checking

        self.weights = np.transpose(np.zeros([num_input_features])) # need modify for feature selections 
        y = np.reshape(y, (y.shape[0], 1))

        # turn the sparse matrix to a numpy array to do calculation
        X_arr = X.toarray()
        # perform gradient ascent
        self.weights = self.gradient_ascent(self.weights, X_arr, y)
    
    def predict(self, X):
        num_examples, num_input_features = X.shape
        
        # if test features is less than training features, 
        # grow test features and fill with zeros
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        # Or perhaps more features are seen at test time, in which case we will
        # simply ignore them.
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        
        X_arr = X.toarray()
        prob = sp.special.expit(np.dot(X_arr, self.weights))
        
        # when the prodiction probability >= .5, predict 1
        # otherwise predict 0
        y_hat = np.where(prob >= .5, 1, 0)
        return y_hat

