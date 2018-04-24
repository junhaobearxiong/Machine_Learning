import numpy as np


class Model(object):

    def __init__(self):
        self.num_input_features = None

    def fit(self, X, y, **kwargs):
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

    def fit(self, X, y, **kwargs):
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


class LambdaMeans(Model):

    def __init__(self):
        super().__init__()
        self.mu = None
        self.num_k = 0
        self.r_matrix = None

    def compute_dist(self, x, y):
        distance = np.sqrt(np.dot(np.transpose(x - y), (x - y)))
        return distance

    # grow numpy array by twice as many columns 
    def grow(self, arr, nrow, ncol):
        new_arr = np.empty((nrow, 2 * ncol))
        new_arr[:, :ncol] = arr[:, :ncol]
        return new_arr

    # find the minimum distance between column vector x and mu's
    # return the minimum distance, index of the cluster with the minimum distance
    def find_min_dist(self, x):
        for k in range(self.num_k):
            mu = self.mu[:, k]
            dist = self.compute_dist(x, mu)
            if k == 0:
                best_dist, best_k = dist, k
            # since we only update best_dist when the current dist is
            # STRICTLY less than best_dist, this ensures the tie breaking rule
            # that we pick the cluster with the lowest cluster index when
            # encountering a tie
            elif dist < best_dist:
                best_dist, best_k = dist, k
        return (best_dist, best_k)

    def E_step(self, X, lambda0):
        self.r_matrix = np.zeros((X.shape[0], self.num_k))
        for n in range(X.shape[0]):
            x = np.transpose(X[n, :])
            best_dist, best_k = self.find_min_dist(x)
            if best_dist <= lambda0:
                self.r_matrix[n, best_k] = 1
            else:
                r_new = np.zeros(X.shape[0])
                r_new[n] = 1
                #self.r_matrix = np.concatenate([self.r_matrix, r_new], axis = 1)
                if self.num_k == self.r_matrix.shape[1]:
                    self.r_matrix = self.grow(self.r_matrix, self.r_matrix.shape[0], self.r_matrix.shape[1])
                self.r_matrix[:, self.num_k] = r_new
                if self.num_k == self.mu.shape[1]:
                    self.mu = self.grow(self.mu, self.mu.shape[0], self.mu.shape[1])
                self.mu[:, self.num_k] = x
                self.num_k += 1
    
    def M_step(self, X):
        self.mu = np.dot(np.transpose(X), self.r_matrix[:, :self.num_k])
        cluster_size = np.sum(self.r_matrix[:, :self.num_k], axis = 0)
        self.mu = np.multiply(self.mu, np.reciprocal(cluster_size))

    def get_default_lambda(self, X, x_bar):
        total_dist = 0
        for n in range(X.shape[0]):
            total_dist += self.compute_dist(X[n, :], x_bar)
        return total_dist / X.shape[0]

    # set mu's for empty clusters to the zero vector
    def set_empty_cluster(self):
        # columns of r_matrix that have all zero entries
        index = np.where(~self.r_matrix[:, :self.num_k].any(axis = 0))[0]
        self.mu[:, index] = 0

    def fit(self, X, _, **kwargs):
        """  Fit the lambda means model  """
        assert 'lambda0' in kwargs, 'Need a value for lambda'
        assert 'iterations' in kwargs, 'Need the number of EM iterations'
        lambda0 = kwargs['lambda0']
        iterations = kwargs['iterations']
        X = X.toarray()
        self.mu = np.empty((X.shape[1], 10))
        x_bar = 1/X.shape[0] * np.sum(X, axis = 0)
        lambda0 = self.get_default_lambda(X, x_bar)
        self.mu[:, self.num_k] = x_bar
        self.num_k += 1
        for i in range(iterations):
            self.E_step(X, lambda0)
            self.M_step(X)
        self.set_empty_cluster()

    def predict(self, X):
        num_feats = self.mu.shape[0]
        if X.shape[1] > num_feats:
            X = X[:, :num_feats]
        elif X.shape[1] < num_feats:
            X._shape = (X.shape[0], num_feats)
        X = X.toarray()
        pred = np.empty(X.shape[0])
        for n in range(X.shape[0]):
            _, k = self.find_min_dist(np.transpose(X[n, :]))
            pred[n] = k
        return pred

class StochasticKMeans(Model):

    def __init__(self):
        super().__init__()
        self.mu = None
        self.p_mat = None

    def init_mu(self, X, num_clusters):
        if num_clusters == 1:
            self.mu[:, 0] = 1/X.shape[0] * np.sum(X, axis = 0)
        else:
            max_x = np.transpose(np.max(X, axis = 0))
            min_x = np.transpose(np.min(X, axis = 0))
            for k in range(num_clusters):
                self.mu[:, k] = (k/(num_clusters-1)) * max_x + (1 - k/(num_clusters-1)) * min_x
    
    def compute_dist(self, x, y):
        distance = np.sqrt(np.dot(np.transpose(x - y), (x - y)))
        return distance

    def E_step(self, X, num_clusters, beta):
        for n in range(X.shape[0]):
            x = np.transpose(X[n, :])
            d_vec = np.empty(num_clusters)
            for k in range(num_clusters):
                d_vec[k] = self.compute_dist(x, self.mu[:, k])
            d_hat = 1/num_clusters * np.sum(d_vec)
            denominator = 0
            for k in range(num_clusters):
                self.p_mat[n, k] = np.exp(-beta * d_vec[k] / d_hat)
                denominator += self.p_mat[n, k]
            self.p_mat[n, :] = 1/denominator * self.p_mat[n, :]

    def M_step(self, X):
        self.mu = np.dot(np.transpose(X), self.p_mat)
        denominator = np.sum(self.p_mat, axis = 0)
        self.mu = np.multiply(self.mu, np.reciprocal(denominator))

    def fit(self, X, _, **kwargs):
        assert 'num_clusters' in kwargs, 'Need the number of clusters (K)'
        assert 'iterations' in kwargs, 'Need the number of EM iterations'
        num_clusters = kwargs['num_clusters']
        iterations = kwargs['iterations']
        X = X.toarray()
        self.mu = np.empty((X.shape[1], num_clusters))
        self.p_mat = np.empty((X.shape[0], num_clusters))
        self.init_mu(X, num_clusters)
        c = 2
        beta = c
        for i in range(iterations):
            beta = c * (i + 1)
            self.E_step(X, num_clusters, beta)
            self.M_step(X)

    # find the minimum distance between column vector x and mu's
    # return the index of the cluster with the minimum distance
    def find_min_dist(self, x):
        for k in range(self.mu.shape[1]):
            mu = self.mu[:, k]
            dist = self.compute_dist(x, mu)
            if k == 0:
                best_dist, best_k = dist, k
            # since we only update best_dist when the current dist is
            # STRICTLY less than best_dist, this ensures the tie breaking rule
            # that we pick the cluster with the lowest cluster index when
            # encountering a tie
            elif dist < best_dist:
                best_dist, best_k = dist, k
        return best_k
    
    # same prediction function as lambda_means
    def predict(self, X):
        num_feats = self.mu.shape[0]
        if X.shape[1] > num_feats:
            X = X[:, :num_feats]
        elif X.shape[1] < num_feats:
            X._shape = (X.shape[0], num_feats)
        X = X.toarray()
        pred = np.empty(X.shape[0])
        for n in range(X.shape[0]):
            k = self.find_min_dist(np.transpose(X[n, :]))
            pred[n] = k
        return pred
        
