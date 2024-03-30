from cgi import test
from lib2to3.pytree import LeafPattern
from turtle import forward
from sympy import N, im
import torch
from torch import nn

import numpy as np

import torchvision # provide access to datasets, models, transforms, utils, etc see below
from torch.utils.data import DataLoader # gives easier dataset managment and creates mini batches
from torchvision import datasets # standard datasets 
from torchvision import transforms # transforms images 
from torchvision import models # pretrained models
from torchvision.transforms.transforms import ToTensor # transforms images to tensors

import matplotlib.pyplot as plt

# Check version of PyTorch and TorchVision
print(torch.__version__)
print(torchvision.__version__)

# Multi Class image classificaition using a Convolutional Neural Network (CNN)
#------------------------------------------------------------

# check if cuda is available
print(torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# add random seed for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)  

# Get a toy dataset from torchvision datasets
#------------------------------------------------------------

# get FashionMNIST dataset --> Training dataset
train_data = datasets.FashionMNIST(
    root="dataset/",
    train=True,
    transform=transforms.ToTensor(),
    download=True
)
# get FashionMNIST dataset --> Test dataset
test_data = datasets.FashionMNIST(
    root="dataset/",
    train=False,
    transform=transforms.ToTensor(),
    download=True
)

# show an image from the dataset
image, label = test_data[0]
plt.figure(1)
plt.imshow(image.squeeze(), cmap="gray")
# ad title to the image with the label and class name
plt.title(f"{label} --> {test_data.classes[label]}")
plt.show()
print("Label:", label)
print("Class:", test_data.classes[label])
print("Shape:", image.shape)

# visualize multiple images from the dataset in a 3x3 grid
fig, axs = plt.subplots(3, 3, figsize=(6, 6))
for i, ax in enumerate(axs.flat):
    image, label = test_data[i]
    ax.imshow(image.squeeze(), cmap="gray")
    ax.set_title(f"{label} --> {test_data.classes[label]}")
plt.tight_layout()
plt.show()


## Prepare the data for training and testing the model using DataLoaders
#------------------------------------------------------------
# Create DataLoaders for the training and test dataset
#------------------------------------------------------------
# define batch size
batch_size = 128

# create DataLoaders
train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# get the number of training and test batches!
N_train = len(train_data)
N_test = len(test_data)


# visualize a batch of images from the dataset in a 4x4 grid with labels 
# and class names
fig, axs = plt.subplots(4, 4, figsize=(6, 6))
for i, ax in enumerate(axs.flat):
    images, labels = next(iter(train_dl))
    ax.imshow(images[0].squeeze(), cmap="gray")
    ax.set_title(f"{labels[0]} --> {test_data.classes[labels[0]]}")
plt.tight_layout()
plt.show()

# Start with a simple classification model relying on fully connected layers
#------------------------------------------------------------

# define the model class 

class FahsionMNistV0(nn.Module):
    
    def __init__(self, input_shape : int, hidden_units : int, output_shape : int) :
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_shape, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_shape),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.layers(x)
    

# create an instance of the model for training
model = FahsionMNistV0(28*28, 128, 10).to(device) # 28*28 --> width * height of the image in pixels

# send data to the device
train_data.train_data.to(device)
train_data.train_labels.to(device)
test_data.test_data.to(device)
test_data.test_labels.to(device)

# define the loss function
loss_function = nn.CrossEntropyLoss() # loss function for multi class classification problems

# define the optimizer
Learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate) # Adam optimizer

# define the number of epochs
epochs = 1

# define lists for storing the loss and accuracy values
losses = []
test_losses = []
accuracies = []

# train the model for the defined number of epochs and batch size

for epoch in range(epochs):
    train_loss = 0
    batch_losses = []
    batch_accuracies = []
    for batch, (X,y) in enumerate(train_dl):
        model.train()
        y_pred = model(X.to(device))
        loss = loss_function(y_pred, y.to(device))
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
        batch_accuracies.append((y_pred.argmax(axis=1) == y.to(device)).float().mean())
    losses.append(train_loss / (N_train / batch_size))

    test_loss = 0
    test_batch_losses = []
    test_batch_accuracies = []
    
    for batch, (X,y) in enumerate(test_dl):
        model.eval()
        with torch.inference_mode():
            y_pred = model(X.to(device))
            loss = loss_function(y_pred, y.to(device))
            test_loss += loss.item()
            test_batch_losses.append(loss.item())
            test_batch_accuracies.append((y_pred.argmax(axis=1) == y.to(device)).cpu().float().mean())
    
    accuracies.append(np.mean(test_batch_accuracies))
    test_losses.append(test_loss / (N_test / batch_size))
    
    print(f"Epoch: {epoch+1}/{epochs}, loss: {losses[-1]:.4f}, test_loss: {test_losses[-1]:.4f}, accuracy: {accuracies[-1]:.4f}")
    

# plot the loss and accuracy values
plt.figure(2)
plt.plot(losses, label="train_loss")
plt.plot(test_losses, label="test_loss")
plt.plot(accuracies, label="accuracy")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# show a batch of images from the test dataset with the predicted labels
fig, axs = plt.subplots(4, 4, figsize=(6, 6))
for i, ax in enumerate(axs.flat):
    images, labels = next(iter(test_dl))
    y_pred = model(images.to(device))
    ax.imshow(images[0].squeeze(), cmap="gray")
    ax.set_title(f"{y_pred.argmax(axis=1)[0]} --> {test_data.classes[y_pred.argmax(axis=1)[0]]}")
plt.tight_layout()
plt.show()


# Improve the model by using convolutional layers
#------------------------------------------------------------
# define the model class

class FahsionMNistV1(nn.Module):
        
        def __init__(self, input_shape : int, output_shape : int, kernel_size : int = 3, stride : int = 2 , padding : int = 0) :
            super().__init__()
            # for calculation of output shape see: https://kvirajdatt.medium.com/calculating-output-dimensions-in-a-cnn-for-convolution-and-pooling-layers-with-keras-682960c73870
            self.hidden_units = 8
            self.cnn_layer_output_shape = np.floor(((28 - kernel_size + 2*padding) / stride) + 1).astype(int)
           
            self.layers = nn.Sequential(
                nn.Conv2d(input_shape, self.hidden_units, kernel_size=kernel_size, stride=stride, padding=padding),
                # I - F + 2P / S + 1 --> 28 - 3 + 2*0 / 2 + 1 = 13 --> 13x13x8
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # I - F / S + 1 --> 13 - 2 / 2 + 1 = 6 --> 6x6x8 
                nn.Flatten(),
                # 
                nn.Linear(self.hidden_units*6*6, output_shape), # 6*6*8 --> 6*6 --> output shape of the pooling layer 
                nn.Softmax(dim=1)
            )
        
        def forward(self, x):
            return self.layers(x)

# create an instance of the model for training
model2 = FahsionMNistV1(1, 10).to(device) # 28*28 --> width * height of the image in pixels

print(model2(torch.randn(1,1,28,28).to(device)).shape) # test the model with a random input

# define the loss function 

loss_fn = nn.CrossEntropyLoss()

# define optimizer 
Learning_rate = 1E-2
optim  = torch.optim.Adam(params = model2.parameters(), lr=Learning_rate) 


# define the number of epochs

epochs = 25

# define lists for storing the loss and accuracy values 
losses = []
test_losses = []
accuracies = []

# train the model for the defined number of epochs and batch size

for epoch in range(epochs):
    
    train_loss = 0
    batch_losses = []
    batch_accuracies = []
    
    for batch, (X,y) in enumerate(train_dl):
        
        model2.train()
        y_pred = model2(X.to(device))
        loss = loss_fn(y_pred, y.to(device))
        train_loss += loss.item()
        optim.zero_grad()
        loss.backward()
        optim.step()
        batch_losses.append(loss.item())
        batch_accuracies.append((y_pred.argmax(axis=1) == y.to(device)).float().mean())
        
    losses.append(train_loss / (N_train / batch_size))

    test_loss = 0
    test_batch_losses = []
    test_batch_accuracies = []
    
    for batch, (X,y) in enumerate(test_dl):
        
        model2.eval()
        with torch.inference_mode():
            y_pred = model2(X.to(device))
            loss = loss_fn(y_pred, y.to(device))
            test_loss += loss.item()
            test_batch_losses.append(loss.item())
            test_batch_accuracies.append((y_pred.argmax(axis=1) == y.to(device)).cpu().float().mean())
    
    accuracies.append(np.mean(test_batch_accuracies))
    test_losses.append(test_loss / (N_test / batch_size))
    
    print(f"Epoch: {epoch+1}/{epochs}, loss: {losses[-1]:.4f}, test_loss: {test_losses[-1]:.4f}, accuracy: {accuracies[-1]:.4f}")

# plot the loss and accuracy values
plt.figure(3)
plt.plot(losses, label="train_loss")
plt.plot(test_losses, label="test_loss")
plt.plot(accuracies, label="accuracy")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# show a batch of images from the test dataset with the predicted labels
fig, axs = plt.subplots(4, 4, figsize=(6, 6))
for i, ax in enumerate(axs.flat):
    images, labels = next(iter(test_dl))
    y_pred = model(images.to(device))
    ax.imshow(images[0].squeeze(), cmap="gray")
    ax.set_title(f"{y_pred.argmax(axis=1)[0]} --> {test_data.classes[y_pred.argmax(axis=1)[0]]}")
plt.tight_layout()
plt.show()

# Plot confusion matrix for the test dataset
#------------------------------------------------------------
from sklearn.metrics import confusion_matrix
import seaborn as sns

# get all predictions for the test dataset
y_pred = []
for batch, (X,y) in enumerate(test_dl):
    model2.eval()
    with torch.inference_mode():
        y_pred.append(model2(X.to(device)).argmax(axis=1).cpu().numpy())
y_pred = np.concatenate(y_pred)

# get all labels for the test dataset
y_true = []
for batch, (X,y) in enumerate(test_dl):
    y_true.append(y.numpy())
y_true = np.concatenate(y_true)

# plot the confusion matrix

plt.figure(4)
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

