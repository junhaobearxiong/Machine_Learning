import torch
import torch.nn as nn
import torchvision.datasets
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F

##TO-DO: Import data here:
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = torchvision.datasets.MNIST('./datasets', train = True, transform = transform)
mnist_test = torchvision.datasets.MNIST('./datasets', train = False, transform = transform)

batch_size = 30

train_loader = torch.utils.data.DataLoader(dataset = mnist_train,
    batch_size = batch_size)

test_loader = torch.utils.data.DataLoader(dataset = mnist_test, 
    batch_size = batch_size)

# input dimension is a mini batch of samples
input_dim = 28 * 28
output_dim = 10

##TO-DO: Define your model:
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        ##Define layers making use of torch.nn functions:
        # define hyperparameters
        self.num_hl_1 = 100
        self.num_hl_2 = 50
        self.num_hl_3 = 20

        self.hl_1 = nn.Linear(input_dim, self.num_hl_1)
        self.hl_2 = nn.Linear(self.num_hl_1, self.num_hl_2)
        self.hl_3 = nn.Linear(self.num_hl_2, self.num_hl_3)
        self.output = nn.Linear(self.num_hl_3, output_dim)
        
    # x is the input vector    
    def forward(self, x):

        ##Define how forward pass / inference is done:
        # reshape the tensor into the desired dimensions for input
        # -1 says we don't specify the number of rows
        x = x.view(-1, input_dim)
        x = F.relu(self.hl_1(x))
        x = F.relu(self.hl_2(x))
        x = F.relu(self.hl_3(x))
        x = self.output(x)
        x = F.log_softmax(x, dim=1)

        return x

my_net = torch.load('model.pkl')

# Testing
correct = 0
total = 0
for data in test_loader:
    x, y_hat = data
    output = my_net(Variable(x))
    # the second return value is the index of the max (argmax) which is what we want
    # torch.max finds the max of each row of the tensor, with given dimension
    # in this case it would find the max of each 1d row vector
    _, y_pred = torch.max(output.data, 1)
    # tensor.size returns a vector with each entry the size of each dimension
    # since y_hat is a 1d vector, we use size(0) to find its length
    total += y_hat.size(0)
    correct += (y_pred == y_hat).sum()

accuracy = 100 * correct / total
print('Test accuracy: {}'.format(accuracy))
