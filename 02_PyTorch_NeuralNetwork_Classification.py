from calendar import c
from cgi import test
from cv2 import normalize
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sklearn 
from sklearn.datasets import make_circles, make_blobs
import pandas as pd
from sklearn.model_selection import train_test_split

# Tutorial 2: Neural Network Classification Model
# ------------------------------------------------
# Classification is the process of predicting the class of given data points. Classes are sometimes called as labels or categories.
# Classification predictive modeling is the task of approximating a mapping function (f) from input variables (X) to discrete output variables (y).
# For example, spam detection in email service providers can be identified as a classification problem. This is s binary classification since there are only 2 classes as spam and not spam.
# A classifier utilizes some training data to understand how given input variables relate to the class. In this case, known spam and non-spam emails have to be used as the training data.
# When the classifier is trained accurately, it can be used to detect an unknown email.
# Classification belongs to the category of supervised learning where the targets also provided with the input data.
# There are 2 types of classification:
# 1. Binary Classification - The classification with 2 classes. Example: Spam detection
# 2. Multi-class Classification - The classification with more than 2 classes. Example: Handwritten digit recognition
# ------------------------------------------------

# 0. Prepare Data
# ------------------------------------------------

N = 1000 # number of samples

X, y = make_circles(n_samples=N, noise=0.03, random_state=42)

# put data into pandas dataframe
df = pd.DataFrame({"X0":X[:,0], "X1":X[:,1], "label":y})
print(df.head(10))



# print first 5 samples of X and y
print(X[:5])
print(y[:5])

# plot data in 2 D with  class label as color
plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Data Distribution of dummy dataset')
plt.show()

# ------------------------------------------------

# Turn data into torch tensor
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

print(X.shape) # 1000 samples, 2 features 
print(y.shape) # 1000 samples, 1 label

# ------------------------------------------------

# Make test and train sets for model training and testing later
test_size = 0.2
N_test = int(test_size * N)
N_train = N - N_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# ------------------------------------------------
# 1. Build classification model with PyTorch
# ------------------------------------------------

# 1.1 Define model class

class CircleModelV0(nn.Module):
    def __init__(self, input_dim=2, output_dim=1):
        super(CircleModelV0, self).__init__()
        self.inner_dimension = 64
        self.linear1 = nn.Linear(input_dim, self.inner_dimension)
        self.linear3 = nn.Linear(self.inner_dimension, self.inner_dimension)
        self.linear2 = nn.Linear(self.inner_dimension, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.activation = nn.ReLU()
        
    def forward(self, x):
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear3(out)
        out = self.activation(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out

# 1.2 Instantiate model class
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
# set seeds to 42 
torch.manual_seed(42)
torch.cuda.manual_seed(42) # also set seed on GPU (if available)

myClassifier = CircleModelV0().to(device)

# 1.3 Instantiate loss class
criterion = nn.BCELoss() # Binary Cross Entropy Loss because we have a binary classification problem

# 1.4 Instantiate optimizer class
learning_rate = 1E-2
optimizer = torch.optim.Adam(myClassifier.parameters(), lr=learning_rate)

# Model surveillance
train_loss = []
test_loss = []
epoch_list = []




# 1.7 Send data to device
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

# 1.6 Train model
epochs = 1000
for epoch in range(epochs):
    myClassifier.train()
    # 1.8 Forward pass to get output
    y_pred = myClassifier(X_train)
    epoch_list.append(epoch+1)
    # 1.9 Calculate Loss
    loss = criterion(torch.squeeze(y_pred), y_train)
    train_loss.append(loss.item())
    optimizer.zero_grad()
    
    # 1.10 Backward pass to get gradient
    loss.backward()
    optimizer.step()
    
    myClassifier.eval() # deactivate dropout layers etc. for evaluation
    with torch.no_grad():
        y_test_pred = myClassifier(X_test)
        test_loss_elem = criterion(torch.squeeze(y_test_pred), y_test).item()
        test_loss.append(test_loss_elem)
    
    print('Epoch: {}, Loss: {}, Test Loss {}'.format(epoch+1, loss.item(), test_loss_elem))

# plot learning curve of training and test loss
plt.figure(2)
plt.plot(epoch_list, train_loss, label='train')
plt.plot(epoch_list, test_loss, label='test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 1.11 Make prediction with trained model
with torch.no_grad():
    y_pred = myClassifier(X_test)
    
y_pred_class = y_pred.round() # round off sigmoid output to obtain class label  < 0.5 = 0, > 0.5 = 1
y_pred_class = torch.squeeze(y_pred_class) # remove redundant dimension
# 1.12 Evaluate model

def evaluate_model_accuracy(y_pred_class, y_test):
    correct = torch.eq(y_pred_class, y_test).sum().item()
    acc = correct/len(y_pred_class) * 100	
    return acc

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test.cpu(), y_pred_class.cpu())
accuracy = accuracy*100 # type: ignore
print('Accuracy: {}'.format(accuracy))

acc = evaluate_model_accuracy(y_pred_class, y_test)
print('Accuracy: {}'.format(acc))

# 1.13 Plot classification result
plt.figure(3)
# add green circle for correct predictions
plt.scatter(X_test[y_pred_class==y_test, 0].cpu().numpy(), X_test[y_pred_class==y_test, 1].cpu().numpy(), c='green',s=60)
# add red circle for wrong predictions
plt.scatter(X_test[y_pred_class!=y_test, 0].cpu().numpy(), X_test[y_pred_class!=y_test, 1].cpu().numpy(), c='red',s=60)

plt.scatter(X_test[:, 0].cpu().numpy(), X_test[:, 1].cpu().numpy(), c=y_pred_class.cpu().numpy(), cmap='jet',s=20)

# add decision boundary to the plot
X0_min, X0_max = X_test[:, 0].min(), X_test[:, 0].max()
X1_min, X1_max = X_test[:, 1].min(), X_test[:, 1].max()
# transform to numpy array
X0_min, X0_max = X0_min.cpu().numpy(), X0_max.cpu().numpy()
X1_min, X1_max = X1_min.cpu().numpy(), X1_max.cpu().numpy()

xx, yy = np.meshgrid(np.linspace(X0_min, X0_max, 100), np.linspace(X1_min, X1_max, 100)) # create grid
grid = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).type(torch.float).to(device) # create grid tensor and reshape to (10000, 2)
y_grid_pred = myClassifier(grid) # predict grid points
#y_grid_pred_class = y_grid_pred.round() # round off sigmoid output to obtain class label  < 0.5 = 0, > 0.5 = 1
y_grid_pred_class = torch.squeeze(y_grid_pred) # remove redundant dimension

plt.contourf(xx, yy, y_grid_pred_class.cpu().reshape(xx.shape).detach().numpy(), cmap='jet', alpha=0.2)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Classification Result')
plt.show()

# Build activtaion functions with PyTorch
# ------------------------------------------------
# 1. Sigmoid
# ------------------------------------------------

def mySigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1/(1+torch.exp(-x))

x = torch.linspace(-10, 10, 100)
y = torch.sigmoid(x)
my_y = mySigmoid(x)
plt.figure(4)
plt.plot(x.numpy(), my_y.numpy(),color='blue')
plt.plot(x.numpy(), y.numpy(), '--', color='red')
plt.xlabel('Input Value') 
plt.ylabel('Output Value')
plt.title('Sigmoid')
plt.show()

# ------------------------------------------------
# 2. ReLU
# ------------------------------------------------

def myReLU(x: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.zeros_like(x), x)

x = torch.linspace(-10, 10, 100)
y = torch.relu(x)
my_y = myReLU(x)
plt.figure(5)
plt.plot(x.numpy(), my_y.numpy(),color='blue')
# plot dashed line for x and y data points
plt.plot(x.numpy(), y.numpy(), '--', color='red')

plt.title('ReLU')
plt.xlabel('Input Value') 
plt.ylabel('Output Value')
plt.show()


# ------------------------------------------------
# Continue with Multi Class Classification in PyTorch
# ------------------------------------------------

# 0. Prepare Data for Multi Class Classification
# ------------------------------------------------

N = 1000 # number of samples per class
NUM_Features = 2 # number of features
NUM_CLASSES = 4 # number of classes
X, y = make_blobs(n_samples=N, n_features=NUM_Features, centers=NUM_CLASSES, cluster_std=1.7, random_state=42) # type: ignore

# add test train split 
test_size = 0.2
N_test = int(test_size * N)
N_train = N - N_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42) # type: ignore

# plot data in 2 D with  class label as color
plt.figure(6)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Data Distribution of dummy dataset')
plt.show()

# Turn data into torch tensor
X_train = torch.from_numpy(X_train).type(torch.float).to(device)
y_train = torch.from_numpy(y_train).type(torch.long).to(device)
X_test = torch.from_numpy(X_test).type(torch.float).to(device)
y_test = torch.from_numpy(y_test).type(torch.long).to(device)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# 1. Build classification model with PyTorch for Multi Class Classification
# ------------------------------------------------

# 1.1 Define model class

class MultiClassModelV1(nn.Module):
    def __init__(self, input_dim=2, output_dim=4, hidden_units=64):
        super(MultiClassModelV1, self).__init__()
        self.inner_dimension = hidden_units
        self.linear1 = nn.Linear(input_dim, self.inner_dimension)
        self.linear3 = nn.Linear(self.inner_dimension, self.inner_dimension)
        self.linear2 = nn.Linear(self.inner_dimension, output_dim)
        self.softmax = nn.Softmax(dim=1)
        self.activation = nn.Tanh()    
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear3(out)
        out = self.activation(out)
        out = self.linear2(out)
        out = self.softmax(out)
        return out
    

# 1.2 Instantiate model class
myMultiClassClassifier = MultiClassModelV1().to(device)

# 1.3 Instantiate loss function

loss_fn = nn.CrossEntropyLoss()	# Cross Entropy Loss because we have a multi class classification problem

# 1.4 Instantiate optimizer class
learning_rate = 1E-2
optimizer = torch.optim.Adam(myMultiClassClassifier.parameters(), lr=learning_rate, weight_decay=1E-5)

# Model surveillance
train_loss = []
test_loss = []
epoch_list = []

# 1.5 Train model
epochs = 1000
for epoch in range(epochs):
    myMultiClassClassifier.train()
    # 1.6 Forward pass to get output
    y_pred = myMultiClassClassifier(X_train)
    epoch_list.append(epoch+1)
    # 1.7 Calculate Loss
    loss = loss_fn(y_pred, y_train)
    train_loss.append(loss.item())
    optimizer.zero_grad()
    
    # 1.8 Backward pass to get gradient
    loss.backward()
    optimizer.step()
    
    myMultiClassClassifier.eval() # deactivate dropout layers etc. for evaluation
    with torch.no_grad():
        y_test_pred = myMultiClassClassifier(X_test)
        test_loss_elem = loss_fn(y_test_pred, y_test).item()
        test_loss.append(test_loss_elem)
    
    print('Epoch: {}, Loss: {}, Test Loss {}'.format(epoch+1, loss.item(), test_loss_elem))

# plot learning curve of training and test loss
plt.figure(7)
plt.plot(epoch_list, train_loss, label='train')
plt.plot(epoch_list, test_loss, label='test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 1.9 Make prediction with trained model
with torch.no_grad():
    y_pred = myMultiClassClassifier(X_test) # type: ignore

print(y_pred)
y_pred_class = torch.argmax(y_pred, dim=1) # type: ignore

# 1.10 Evaluate model

accuracy = accuracy_score(y_test.cpu(), y_pred_class.cpu())
accuracy = accuracy*100 # type: ignore
print('Accuracy: {}'.format(accuracy))

# 1.11 Plot classification result
plt.figure(8)
# add green circle for correct predictions
plt.scatter(X_test[y_pred_class==y_test, 0].cpu().numpy(), X_test[y_pred_class==y_test, 1].cpu().numpy(), c='green',s=60)
# add red circle for wrong predictions 
plt.scatter(X_test[y_pred_class!=y_test, 0].cpu().numpy(), X_test[y_pred_class!=y_test, 1].cpu().numpy(), c='red',s=60)
# add decision boundary to the plot
X0_min, X0_max = X_test[:, 0].min(), X_test[:, 0].max()
X1_min, X1_max = X_test[:, 1].min(), X_test[:, 1].max()
# transform to numpy array
X0_min, X0_max = X0_min.cpu().numpy(), X0_max.cpu().numpy()
X1_min, X1_max = X1_min.cpu().numpy(), X1_max.cpu().numpy()
# create grid
xx, yy = np.meshgrid(np.linspace(X0_min, X0_max, 100), np.linspace(X1_min, X1_max, 100))
# create grid tensor and reshape to (10000, 2)
grid = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).type(torch.float).to(device)
# predict grid points
y_grid_pred = myMultiClassClassifier(grid)
# get class with highest probability
y_grid_pred_class = torch.argmax(y_grid_pred, dim=1)
# plot decision boundary
plt.contourf(xx, yy, y_grid_pred_class.cpu().reshape(xx.shape).detach().numpy(), cmap='jet', alpha=0.2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Classification Result')
plt.show()

# Adding a few more evaluation metrics to our model
# ------------------------------------------------
# 1.12 Evaluate model
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test.cpu(), y_pred_class.cpu()))
print(confusion_matrix(y_test.cpu(), y_pred_class.cpu()))

# 1.13 Plot confusion matrix
import seaborn as sns
plt.figure(9)
cm = confusion_matrix(y_test.cpu(), y_pred_class.cpu())
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

# Redo using torch metrics functions 
# ------------------------------------------------
from torchmetrics import Accuracy, Precision, Recall, F1Score

my_Accuracy = Accuracy(task='multiclass', average='macro', num_classes=NUM_CLASSES).to(device)
my_Precision = Precision(task='multiclass', average='macro', num_classes=NUM_CLASSES).to(device)
my_Recall = Recall(task='multiclass', average='macro', num_classes=NUM_CLASSES).to(device)
my_F1 = F1Score(task='multiclass', average='macro', num_classes=NUM_CLASSES).to(device)

with torch.no_grad():
    y_pred = myMultiClassClassifier(X_test) # type: ignore
    y_pred_class = torch.argmax(y_pred, dim=1) # type: ignore
    print('Accuracy: {}'.format(my_Accuracy(y_pred_class, y_test)))
    print('Precision: {}'.format(my_Precision(y_pred_class, y_test)))
    print('Recall: {}'.format(my_Recall(y_pred_class, y_test)))
    print('F1: {}'.format(my_F1(y_pred_class, y_test)))
    