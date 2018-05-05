import numpy as np

class ChainMRFPotentials:
    def __init__(self, data_file):
        with open(data_file) as reader:
            for line in reader:
                if len(line.strip()) == 0:
                    continue

                split_line = line.split(" ")
                try:
                    self._n = int(split_line[0])
                except ValueError:
                    raise ValueError("Unable to convert " + split_line[0] + " to integer.")
                try:
                    self._k = int(split_line[1])
                except ValueError:
                    raise ValueError("Unable to convert " + split_line[1] + " to integer.")
                break

            # create an "(n+1) by (k+1)" list for unary potentials
            self._potentials1 = [[-1.0] * ( self._k + 1) for n in range(self._n + 1)]
            # create a "2n by (k+1) by (k+1)" list for binary potentials
            self._potentials2 = [[[-1.0] * (self._k + 1) for k in range(self._k + 1)] for n in range(2 * self._n)]

            for line in reader:
                if len(line.strip()) == 0:
                    continue

                split_line = line.split(" ")

                if len(split_line) == 3:
                    try:
                        i = int(split_line[0])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[0] + " to integer.")
                    try:
                        a = int(split_line[1])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[1] + " to integer.")
                    if i < 1 or i > self._n:
                        raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
                    if a < 1 or a > self._k:
                        raise Exception("given k=" + str(self._k) + ", illegal value for a: " + str(a))
                    if self._potentials1[i][a] >= 0.0:
                        raise Exception("ill-formed energy file: duplicate keys: " + line)
                    self._potentials1[i][a] = float(split_line[2])
                elif len(split_line) == 4:
                    try:
                        i = int(split_line[0])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[0] + " to integer.")
                    try:
                        a = int(split_line[1])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[1] + " to integer.")
                    try:
                        b = int(split_line[2])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[2] + " to integer.")
                    if i < self._n + 1 or i > 2 * self._n - 1:
                        raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
                    if a < 1 or a > self._k or b < 1 or b > self._k:
                        raise Exception("given k=" + self._k + ", illegal value for a=" + str(a) + " or b=" + str(b))
                    if self._potentials2[i][a][b] >= 0.0:
                        raise Exception("ill-formed energy file: duplicate keys: " + line)
                    self._potentials2[i][a][b] = float(split_line[3])
                else:
                    continue

            # check that all of the needed potentials were provided
            for i in range(1, self._n + 1):
                for a in range(1, self._k + 1):
                    if self._potentials1[i][a] < 0.0:
                        raise Exception("no potential provided for i=" + str(i) + ", a=" + str(a))
            for i in range(self._n + 1, 2 * self._n):
                for a in range(1, self._k + 1):
                    for b in range(1, self._k + 1):
                        if self._potentials2[i][a][b] < 0.0:
                            raise Exception("no potential provided for i=" + str(i) + ", a=" + str(a) + ", b=" + str(b))

    def chain_length(self):
        return self._n

    def num_x_values(self):
        return self._k

    def potential(self, i, a, b = None):
        if b is None:
            if i < 1 or i > self._n:
                raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
            if a < 1 or a > self._k:
                raise Exception("given k=" + str(self._k) + ", illegal value for a=" + str(a))
            return self._potentials1[i][a]

        if i < self._n + 1 or i > 2 * self._n - 1:
            raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
        if a < 1 or a > self._k or b < 1 or b > self._k:
            raise Exception("given k=" + str(self._k) + ", illegal value for a=" + str(a) + " or b=" + str(b))
        return self._potentials2[i][a][b]


class SumProduct:
    def __init__(self, p):
        self._potentials = p
        self.length = p.chain_length()
        self.dim = p.num_x_values()

        # store factor values in ndarray
        self.f_unary = {} # store unary factor values
        self.f_binary = {} # store binary factor values
        for i in range(1, self.length+1):
            f_a = np.zeros(self.dim)
            f_ab = np.zeros((self.dim, self.dim))
            for k in range(self.dim):
                f_a[k] = p.potential(i, k+1)
                for j in range(self.dim):
                    if i == self.length:
                        break
                    f_ab[k, j] = p.potential(self.length + i, k+1, j+1)
            self.f_unary[i] = f_a
            if i == self.length:
                break
            self.f_binary[self.length + i] = f_ab
    
        # store all the messages after one forward and backward pass
        # indexed by the node receiving the message
        # each entry is a list of messages passed to that node
        self.messages = {}
        self.forward_message(self.length*2 - 1)
        self.backward_message(self.length + 1)

    # y is the factor sending the messages
    # compute messages from node X_1 to X_n, from factors to variables
    # starting from the factor 2n-1
    def forward_message(self, y):
        # add the message from factor n since it is not computed in the recursion
        self.messages[self.length] = []
        self.messages[self.length].append(self.f_unary[self.length])
        if y <= self.length:
            self.messages[y] = []
            self.messages[y].append(self.f_unary[y])
            return self.f_unary[y]
        else:
            # message from the previous x
            if y-1 > self.length:
                mu_x = np.multiply(self.forward_message(y-self.length), self.forward_message(y-1))
            else:
                mu_x = self.forward_message(y-self.length)
            mu = np.dot(np.transpose(self.f_binary[y]), mu_x)
            self.messages[y-self.length+1].append(mu)
            return mu
    
    # compute messages from node X_n to X_1, factors to variables
    # starting from factor n+1
    def backward_message(self, y):
        if y <= self.length:
            return self.f_unary[y]
        else:
            if y+1 < self.length*2:
                mu_x = np.multiply(self.backward_message(y-self.length+1), self.backward_message(y+1))
            else:
                mu_x = self.backward_message(y-self.length+1)
            mu = np.dot(self.f_binary[y], mu_x)
            self.messages[y-self.length].append(mu)
            return mu
    
    # for the use of Max Sum
    def get_normalize_const(self):
        marg = np.ones(self.dim)
        for mu in self.messages[1]:
            marg = np.multiply(mu, marg)
        Z = np.sum(marg)
        return Z

    # return a python list of type float, with its length=k+1, and the first value 0
    def marginal_probability(self, x_i):
        marg = np.ones(self.dim)
        for mu in self.messages[x_i]:
            marg = np.multiply(mu, marg)
        Z = np.sum(marg)
        marg = np.divide(marg, Z)
        marginals = marg.tolist()
        marginals.insert(0, 0) 
        return marginals

class MaxSum:
    def __init__(self, p):
        self._potentials = p
        self._assignments = [0] * (p.chain_length() + 1)
        self.length = p.chain_length()
        self.dim = p.num_x_values()

        # store factor values in ndarray
        self.f_unary = {} # store unary factor values
        self.f_binary = {} # store binary factor values
        for i in range(1, self.length+1):
            f_a = np.zeros(self.dim)
            f_ab = np.zeros((self.dim, self.dim))
            for k in range(self.dim):
                f_a[k] = p.potential(i, k+1)
                for j in range(self.dim):
                    if i == self.length:
                        break
                    f_ab[k, j] = p.potential(self.length + i, k+1, j+1)
            self.f_unary[i] = f_a
            if i == self.length:
                break
            self.f_binary[self.length + i] = f_ab
    
        # store all the messages after one forward and backward pass
        # indexed by the node receiving the message
        # each entry is a list of messages passed to that node
        self.messages = {}
        # store the likely configurations
        self.configurations = {}
        self.forward_message(self.length*2 - 1)
        self.backward_message(self.length + 1)
        
        sp = SumProduct(p)
        Z = sp.get_normalize_const()
        # in log space, so subtract log of const
        self.max_log_prob = self.compute_max_log_prob() - np.log(Z)
        
    def forward_message(self, y):
        # add the message from factor n since it is not computed in the recursion
        self.messages[self.length] = []
        self.messages[self.length].append(self.f_unary[self.length])
        if y <= self.length:
            self.messages[y] = []
            self.messages[y].append(np.log(self.f_unary[y]))
            return np.log(self.f_unary[y])
        else:
            # message from the previous x
            if y-1 > self.length:
                mu_x = self.forward_message(y-self.length) + self.forward_message(y-1)
            else:
                mu_x = self.forward_message(y-self.length) 
            mu_sum = np.add(np.log(np.transpose(self.f_binary[y])), mu_x)
            mu = np.max(mu_sum, axis = 1)
            self.messages[y-self.length+1].append(mu)
            return mu
        
    # compute messages from node X_n to X_1, factors to variables
    # starting from factor n+1
    def backward_message(self, y):
        if y <= self.length:
            return np.log(self.f_unary[y])
        else:
            if y+1 < self.length*2:
                mu_x = self.backward_message(y-self.length+1) + self.backward_message(y+1)
            else:
                mu_x = self.backward_message(y-self.length+1)
            mu_sum = np.add(np.log(self.f_binary[y]), mu_x)
            mu = np.max(mu_sum, axis = 1)
            self.messages[y-self.length].append(mu)
            config = np.argmax(mu_sum, axis = 1)
            self.configurations[y-self.length] = config
            return mu

    # backtracking starting from X_1
    def backtrack(self, x):
        next_x = x
        for i in range(1, self.length+1):
            self._assignments[i] = next_x + 1
            if i == self.length:
                break
            next_x = self.configurations[i][next_x]

    # compute max log prob without normalizing constant
    # backtrack starting from X_1 to get all the assignments
    def compute_max_log_prob(self):
        mu_sum = np.zeros(self.dim)
        for mu in self.messages[1]:
            mu_sum += mu
        self.backtrack(np.argmax(mu_sum))
        return np.max(mu_sum)

    def get_assignments(self):
        return self._assignments

    def max_probability(self, x_i):
        return self.max_log_prob
