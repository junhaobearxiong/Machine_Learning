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

batch_size = 50

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
        # number of nodes in each hidden layers
        self.num_hl_1 = 500
        self.num_hl_2 = 200
        self.num_hl_3 = 50

        # initialize each hidden layer
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
        # softmax so we can use nll loss
        x = F.log_softmax(x, dim=1)

        return x

my_net = Net()

## Training

epoch = 10
optimizer = optim.SGD(my_net.parameters(), lr = .01, momentum = .5)


for i in range(epoch):
    for batch_num, mini_batch in enumerate(train_loader):
        # wrap in Variable
        x, y_hat = mini_batch
        x, y_hat = Variable(x), Variable(y_hat)

        # clear gradient buffer
        optimizer.zero_grad()
        # make prediction
        y_pred = my_net(x)
        # calculate loss
        loss = F.nll_loss(y_pred, y_hat)
        # compute gradient via backprop
        loss.backward()
        # updates model parameters (weights and biases)
        optimizer.step()
    print('Iteration {} completed'.format(i + 1))

#torch.save(my_net.state_dict(), 'model.pkl')
torch.save(my_net, 'model.pkl')
