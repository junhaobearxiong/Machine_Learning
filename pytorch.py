import torch
from torch.autograd import Variable
import numpy as np

x = torch.Tensor(5, 3) # empty tensor
x = torch.rand(5, 3) # uniform random
x.numpy() # convert tensor to numpy array 

x = np.random.randn(5, 3) # random normal
torch.from_numpy(x) # convert numpy to torch

x.sum()
torch.ones(5)

x = Variable(torch.ones(2, 2), requires_grad = True)
y = x + 2
z = y * y * 3
out = z.mean()

print(z, out)
out.backward()

optimizer = optim.SGD(parameters, lr = LR)

for i in range(iterations):
    optimizer.zero_grad() # reset gradient
    out = loss(...)
    out.backward()
    optimizer.step()

# Perceptron
Xtrain = torch.randn(100, 6)
Xtest = torch.randn(100, 6)

w_ground_truth = torch

# reshape into 6x1 instead of 6
ytrain = torch.sign(Xtrain.mm(w_ground_truth.view(-1, 1)))) # view similar to reshape
ytest = torch.sign(Xtest.mm(w_ground_truth.view(-1, 1)))) # view similar to reshape

# cast to Variable object
Xtrain = Variable(Xtrain)
Xtest = Variable(Xtest)
ytrain = Variable(ytrain)
ytest = Variable(ytest)

w_ground_truth = Variable(w_ground_truth)

def loss(X, y, w):
    N = X.shape[0]
    S = 0
    for i in range(N):
        S += torch.max(Variable(torch.Tensor([0]), -y[i] * w.dot(X[i])))
    rteurn S/N

iterations = 100
LR = .1
w_cur = Variable(torch.randn(6), requires_grad = True)
print('initial training loss', loss(Xtrain, ytrain, w_cur))
# w_cur is the variables we want to optimize
optimizer = optim.SGD([w_cur], lr = LR)

for i in range(iterations):
    optimizer.zero_grad()
    out = loss(Xtrain, ytrain, w_cur)
    out.backward() # compute the gradient 
    optimizer.step() # actually change the variables
