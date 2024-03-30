from cgi import test
from cv2 import line
from sympy import plot
import torch 
import numpy as np
from torch import nn # neural network module of pytorch for building the computational graph
import matplotlib.pyplot as plt

from torch import optim # optimization module of pytorch for optimization algorithms like SGD, Adam, etc.

## Pytorch Workflow Chapter 1 in Course 

# 1. check pytorch version
__version__ = torch.__version__
print(__version__)
# 2. check if gpu is available and use it if it is
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# 3. generate some dummy data using a linear function
weight = 0.7
bias = 0.3

start = 0
end = 1
num_of_samples = 100

x = torch.linspace(start,end,num_of_samples).reshape(-1,1)
y = weight*x + bias # linear function

plt.figure(1)
plt.scatter(x,y)
plt.title('Dummy Data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# split data into train and test set
train_set_size = int(num_of_samples*0.8)
test_set_size = num_of_samples - train_set_size

X_train, Y_train = x[:train_set_size], y[:train_set_size]	
X_test, Y_test = x[train_set_size:], y[train_set_size:]

print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)


def plot_predictions(train_data, train_labels, test_data, test_labels, predictions=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c='b', label='Training data')
    plt.scatter(test_data, test_labels, c='g', label='Testing data')
    if predictions is not None:
        plt.scatter(test_data, predictions, c='r', label='Predictions')
    plt.legend()
    plt.show()

plot_predictions(X_train, Y_train, X_test, Y_test)

# 4. create a linear regression model using pytorch

class LinearRegression(nn.Module):
    def __init__(self): # constructor
        super().__init__() # inherit from nn.Module class
        self.weights = nn.Parameter(torch.randn(1,requires_grad=True,dtype=torch.float)) # initialize weights randomly
        self.bias = nn.Parameter(torch.randn(1,requires_grad=True,dtype=torch.float)) # initialize bias randomly

    def forward(self,x):
        return x * self.weights + self.bias
    

    
    
### Pytorch  Model Building Blocks 
# 1. torch.nn - contains all the building blocks to build a neural network computational graph
# 2. torch.nn.Module - base class for all neural network modules if you subclass it you have to implement the forward method
# 3. torch.nn.Parameter - a wrapper for tensors that tells a nn.Module that they have to be trained in many cases a Pytorch layer from torch.nn will set these for us automatically
# 4. torch.nn.functional - contains many useful functions like activation functions, loss functions, etc.
# 5. torch.optim - contains many useful optimization algorithms like SGD, Adam, etc.
# 6. torch.utils.data - contains useful classes for loading data like Dataset and DataLoader
# 7. torch.utils.tensorboard - contains utility classes for logging data to tensorboard
# 8. torch.utils.checkpoint - contains utility classes for checkpointing models

torch.manual_seed(42) # set random seed for reproducibility

linear_regression = LinearRegression().to(device) # create an instance of the LinearRegression class

# get the parameters of the model
print(linear_regression.state_dict()) # parameters are the weights and biases of the model the function is inherited from nn.Module

# 5. train the model

# 5.1 define the loss function to estimate the missmatch between the predictions and the labels
loss_function = nn.L1Loss() # L1 error
loss_function = nn.MSELoss() # mean squared error loss

# 5.2 define the optimizer to update the parameters of the model
optimizer = torch.optim.SGD(linear_regression.parameters(),lr=.1, weight_decay=0.001) # stochastic gradient descent optimizer

# 5.3 define the training loop
epochs = 1000  # number of epochs
# send train and test data to the device
X_train = X_train.to(device)
Y_train = Y_train.to(device)
X_test = X_test.to(device)
Y_test = Y_test.to(device)

# tracking experiments
train_losses = [] # list to store the tain losses for each epoch
test_losses = [] # list to store the test losses for each epoch
epoch_counter = [] # list to store the epoch number

for epoch in range(epochs):
    epoch_counter.append(epoch+1) # +1 because epoch starts at 0 and we want to start at 1
    linear_regression.train() # set the model to training mode
    # make a prediction using forward pass
    predictions = linear_regression(X_train)
    
    # calculate the loss using the defined loss function
    loss = loss_function(predictions,Y_train)
    train_losses.append(loss.item())
    
    # zero the gradients
    optimizer.zero_grad()
    
    # bread and butter steps of NN training are backpropagation and gradient descent
    
    
    # calculate the gradients using backward pass
    loss.backward() # backpropagation step
    
    # update the parameters of the model using the gradients and the optimizer
    optimizer.step() # gradient descent step
    
    linear_regression.eval() # set the model to evaluation mode
    with torch.inference_mode():
        predictions = linear_regression(X_test)
        test_loss = loss_function(predictions,Y_test) # calculate the loss on the test set
        test_losses.append(test_loss.item())
 
    
    # print the loss for each epoch
    print(f'Epoch: {epoch+1}/{epochs}, Loss: {loss.item():.3f}, Test Loss: {test_loss.item():.3f}')

# 5.4 plot the predictions after training the model
linear_regression.eval() # set the model to evaluation mode
with torch.inference_mode():
    predictions = linear_regression(X_test)
    test_loss = loss_function(predictions,Y_test) # calculate the loss on the test set
    print(test_loss.item())

# convert tensor to numpy array
predictions = predictions.detach().cpu().numpy()
plot_predictions(X_train.cpu().numpy(), Y_train.cpu().numpy(), X_test.cpu().numpy(), Y_test.cpu().numpy(), predictions)    

# show the train and test losses over the epochs
plt.figure(2)
plt.plot(train_losses,label='Train Loss')
plt.plot(test_losses,label='Test Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('L1 Loss')
plt.show()

# 6. save the model using torch.save
# torch.save - saves a serialized object to disk using pickle module
# torch.load - loads a serialized object from disk using pickle module
torch.save(linear_regression.state_dict(),'linear_regression.pth') # save the state dict of the model

torch.save(linear_regression,'linear_regression_model.pth') # save the entire model

# 7. load the model using torch.load
# load the state dict of the model
loaded_model = torch.load('linear_regression_model.pth')

loaded_model.eval() # set the model to evaluation mode
with torch.inference_mode():
    predictions = loaded_model(X_test)
    test_loss = loss_function(predictions,Y_test) # calculate the loss on the test set
    print(test_loss.item())
    
# convert tensor to numpy array
predictions = predictions.detach().cpu().numpy()
# plot the predictions from the loaded model
plot_predictions(X_train.cpu().numpy(), Y_train.cpu().numpy(), X_test.cpu().numpy(), Y_test.cpu().numpy(), predictions)


# Building the linear model class again but with use of nn.Linear
class LinearRegressionV2(nn.Module):
    def __init__(self): # constructor
        super().__init__() # inherit from nn.Module class
        self.linear = nn.Linear(in_features=1,out_features=1,bias=True) # initialize weights randomly

    def forward(self,x : torch.Tensor): # type hinting for the input to be of type torch.Tensor
        return self.linear(x)


# set random seed to 42

torch.manual_seed(42) # set random seed for reproducibility

my_linear_regressor_v2 = LinearRegressionV2().to(device) # create an instance of the LinearRegression class

# loss function 2 

loss_function_v2 = nn.MSELoss() # mean squared error loss

# optimzer for the second model

optimizer_v2 = torch.optim.SGD(my_linear_regressor_v2.parameters(),lr=.1, weight_decay=0.001) # stochastic gradient descent optimizer

# training loop again 
epochs = 1000  # number of epochs
# send train and test data to the device
X_train = X_train.to(device)
Y_train = Y_train.to(device)
X_test = X_test.to(device)
Y_test = Y_test.to(device)

# tracking experiments
train_losses = [] # list to store the tain losses for each epoch
test_losses = [] # list to store the test losses for each epoch
epoch_counter = [] # list to store the epoch number

for epoch in range(epochs):
    my_linear_regressor_v2.train() # set the model to training mode
    epoch_counter.append(epoch+1) # +1 because epoch starts at 0 and we want to start at 1
    # make a prediction using forward pass
    predictions = my_linear_regressor_v2(X_train)
    loss = loss_function_v2(predictions,Y_train) # calculate the loss using the defined loss function
    train_losses.append(loss.item())
    optimizer_v2.zero_grad() # zero the gradients
    loss.backward() # backpropagation step
    optimizer_v2.step() # gradient descent step
    my_linear_regressor_v2.eval() # set the model to evaluation mode
    with torch.inference_mode():
        predictions = my_linear_regressor_v2(X_test)
        test_loss = loss_function_v2(predictions,Y_test) # calculate the loss on the test set
        test_losses.append(test_loss.item())
    # print the train and test loss for each epoch 
    print(f'Epoch: {epoch+1}/{epochs}, Loss: {loss.item():.3f}, Test Loss: {test_loss.item():.3f}')


# plot learning curves
plt.figure(3)
plt.plot(train_losses,label='Train Loss')
plt.plot(test_losses,label='Test Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.show()

# plot predictions
my_linear_regressor_v2.eval() # set the model to evaluation mode
with torch.inference_mode():
    predictions = my_linear_regressor_v2(X_test)
    test_loss = loss_function_v2(predictions,Y_test) # calculate the loss on the test set
    print(test_loss.item())

# convert tensor to numpy array
predictions = predictions.detach().cpu().numpy()
# plot the predictions from the loaded model
plot_predictions(X_train.cpu().numpy(), Y_train.cpu().numpy(), X_test.cpu().numpy(), Y_test.cpu().numpy(), predictions)

