import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
# installed: scikit-learn, pytorch, onnx, pandas

# Prepare the dataset
############################################################################################
# Load the iris dataset
iris = load_iris()

# Create X and y data
X = iris.data
y = iris.target

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create the model
#############################################################################################

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = nn.Linear(4, 50)
        self.lin2 = nn.Linear(50, 50)
        self.out = nn.Linear(50, 3)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.out(x)
        return x


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

# Train the network
#############################################################################################
for epoch in range(50):
    inputs = torch.autograd.Variable(torch.Tensor(X_train).float())
    targets = torch.autograd.Variable(torch.Tensor(y_train).long())

    optimizer.zero_grad()
    out = net(inputs)
    loss = criterion(out, targets)
    loss.backward()
    optimizer.step()

inputs = torch.autograd.Variable(torch.Tensor(X_test).float())
targets = torch.autograd.Variable(torch.Tensor(y_test).long())

optimizer.zero_grad()
out = net(inputs)
_, predicted = torch.max(out.data, 1)

error_count = y_test.size - np.count_nonzero((targets == predicted).numpy())
print('Errors: %d; Accuracy: %d%%' % (error_count, 100 * torch.sum(targets == predicted) / y_test.size))

# Save the model as ONNX
###############################################################################################
torch.onnx.export(net, inputs, "pytorch_iris.onnx")