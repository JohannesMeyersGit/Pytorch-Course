from matplotlib import axes
import torch 
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import os
from torchinfo import summary
from typing import Tuple, List, Dict
import matplotlib.image as mpimg
import random
import PIL
# In this tutorial, we will learn how to create a custom dataset class in PyTorch.

# ----------------------------------- Custom Dataset -----------------------------------

# Get the data from github repo

import requests
import zipfile
from pathlib import Path
if __name__ == '__main__':
    random.seed(42) # set the seed for reproducibility 

    DATA_PATH = Path("data/")
    image_path = DATA_PATH / "pizza_steak_shushi"

    # if the folder does not exist, create it
    if not image_path.exists():
        image_path.mkdir(parents=True, exist_ok=True)
    else:
        print("Folder already exists.")

    # Download the data from github repo
    with open(DATA_PATH / "pizza_steak_shushi.zip", "wb") as f:
        r = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        f.write(r.content)

    # Unzip the data out commented to dont do it again and again
    #with zipfile.ZipFile(DATA_PATH / "pizza_steak_shushi.zip", "r") as zip_ref:
    #    zip_ref.extractall(image_path)

    # walk trhough the data directory and list the number of files per folder
    for dirpath, dirnames, filenames in os.walk(image_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

    # Dataset is slightly unbalanced, but it is ok for this tutorial

    # Setup train and test data paths
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    # Get the class names
    class_names = os.listdir(train_dir)
    print(class_names)

    # Check for the number of images in each class
    for class_name in class_names:
        print(f"There are {len(os.listdir(train_dir / class_name))} {class_name} images.")
        print(f"There are {len(os.listdir(test_dir / class_name))} {class_name} images.")
        
    # Check an image
    def view_random_image(target_dir, target_class):
        # Set up the target directory
        target_folder = target_dir / target_class
        
        # Get a random image path
        random_image = random.sample(os.listdir(target_folder), 1)
        
        # Read in the image and plot it using matplotlib
        img = mpimg.imread(target_folder / random_image[0])
        plt.figure(1)
        plt.imshow(img)
        plt.title(target_class)
        plt.axis("off")
        plt.show()
        print(f"Image shape: {img.shape}") # show the shape of the image
        
        return img

    # View a random image from the training dataset
    img = view_random_image(train_dir, random.choice(class_names))

    # ----------------------------------- Preprocess the data -----------------------------------
    target_folder = train_dir / random.choice(class_names)
    random_image = random.sample(os.listdir(target_folder), 1)
    # load test image using PIL
    im = PIL.Image.open(target_folder/ random_image[0])

    # Define the transforms to apply to the data
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)), # resize the image to 64x64 pixels
        transforms.RandomHorizontalFlip(p=0.5), # randomly flip the image horizontally
        transforms.ToTensor() # convert the image to PyTorch tensor
    ])

    train_im = train_transform(im)
    print(f"Image shape: {train_im.shape}") # show the shape of the image
    print(f"Image dtype: {train_im.dtype}") # show the image data type after transformation

    # View the image after transformation
    plt.figure(1)
    plt.imshow(train_im.permute(1, 2, 0)) # permute the axes to plot the image correctly
    plt.title(random_image[0])
    plt.axis("off")
    plt.show()

    def plot_transformed_images(image, transform, n=3, seed=42):
        """
        Plots a random image n times with a given transformation.
        """
        random.seed(seed)
        fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(20, 10))
        for i in range(n):
            ax = axes[i]
            plt_im = transform(image)
            ax.imshow(plt_im.permute(1, 2, 0))
            ax.set_title(f"Transformed image {i + 1}")
            ax.axis("off")
        plt.show()

    plot_transformed_images(im, train_transform)

    # Load the data using ImageFolder from torchvision datasets
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    test_data = datasets.ImageFolder(test_dir, transform=train_transform)

    # Create a DataLoader
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=1, pin_memory=True)

    print(train_data)
    class_names = train_data.classes
    print(class_names)
    class_dict = train_data.class_to_idx
    print(class_dict)

    # check length of the train and test data

    print(len(train_data)) # number of images in the training dataset
    print(len(test_data)) # number of images in the test dataset

    # Get a batch of images and labels
    #images, labels = train_loader
    #print(images.shape, labels.shape) # print the shape of the images and labels in a batch  BATCH_SIZE x CHANNELS x HEIGHT x WIDTH

    im = train_data[0][0] # get the first image in the training dataset
    label = train_data[0][1] # get the label of the first image in the training dataset

    im2 = train_data[10][0] # get the 11th image in the training dataset
    label2 = train_data[10][1] # get the label of the 11th image in the training dataset

    # visualize the images and labels

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    axes[0].imshow(im.permute(1, 2, 0)) # change axes order to plot the image correctly
    axes[0].set_title(f"Label: {class_names[label]}")
    axes[0].axis("off")
    axes[1].imshow(im2.permute(1, 2, 0)) # change axes order to plot the image correctly
    axes[1].set_title(f"Label: {class_names[label2]}")
    axes[1].axis("off")
    plt.show()


    # ----------------------------------- Create a custom dataset -----------------------------------

    # Create a custom dataset class for the Food3 dataset 
    # 1. Want to be able to load images from file
    # 2. Want to be able to get class names from the Dataset 
    # 3. Want to be able to get classes as dictionary from the Dataset

    # Pros of creating a custom dataset class
    # 1. Can be used with PyTorch's DataLoader to load batches of images and apply transformations on the fly
    # 2. Can be used to create train and test splits of the data
    # 3. Can be extended to suit your needs (e.g. loading images from a database)


    # Setup path for target directory
    target_dir = train_dir 
    print(f"Target directory: {target_dir}")

    # get class names from the target directory
    class_names = os.listdir(target_dir)
    print(f"Class names: {class_names}")

    # get class names as dictionary from the target directory
    class_dict = {class_name: i for i, class_name in enumerate(class_names)}
    print(f"Class dictionary: {class_dict}")

    def find_classes(dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Returns the class names and class dictionary for a given directory.
        """
        classes = os.listdir(dir) # get all the class names from the directory alternative to os scandir
        if not classes:
            raise FileNotFoundError(f"Couldn't find any classes in directory '{dir}'. Check your directory and try again.")
        
        classes.sort()
        class_dict = {class_name: i for i, class_name in enumerate(classes)}
        return classes, class_dict

    print(find_classes(target_dir))

    # Create a custom dataset class for the Food3 dataset

    class Food3Dataset(Dataset): 
        """
        Custom dataset class for the Food3 dataset.
        """
        def __init__(self, dir: str, transform: transforms.Compose = transforms.Compose([transforms.ToTensor()])):
            """
            Args:
                dir: path to the target directory.
                transform: optional transform to apply to the images.
            """
            self.dir = dir
            self.transform = transform
            self.classes, self.class_dict = self.find_classes()
            self.images = self.find_images()
        
        def __len__(self) -> int:
            """
            Returns the total number of samples.
            """
            return len(self.images)
        
        def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Gets the image and label at a given index.
            """
            image_path, label = self.images[index]
            image = self.load_image(image_path)
            return image, label
        
        def find_classes(self) -> Tuple[List[str], Dict[str, int]]:
            """
            Returns the class names and class dictionary for a given directory.
            """
            classes = os.listdir(self.dir) # get all the class names from the directory
            if not classes:
                raise FileNotFoundError(f"Couldn't find any classes in directory '{self.dir}'. Check your directory and try again.")
            
            classes.sort()
            class_dict = {class_name: i for i, class_name in enumerate(classes)}
            return classes, class_dict
        
        def find_images(self) -> List[Tuple[str, int]]:
            """
            Returns a list of (image path, label) tuples for a given directory.
            """
            images = []
            for class_name in self.classes:
                class_dir = os.path.join(self.dir, class_name)
                for image_name in os.listdir(class_dir):
                    image_path = os.path.join(class_dir, image_name)
                    images.append((image_path, self.class_dict[class_name]))
            return images
        
        def load_image(self, image_path: str) -> torch.Tensor:
            """
            Loads an image from a file path and transforms it if required.
            """
            image = PIL.Image.open(image_path)
            if self.transform:
                image = self.transform(image)
            return image
        

    # Create a custom dataset class for the Food3 dataset
    food3_train_dataset = Food3Dataset(train_dir, transform=train_transform)
    food3_test_dataset = Food3Dataset(test_dir, transform=train_transform)

    print(len(food3_test_dataset))

    # Get a sample image and label from the dataset
    image, label = food3_test_dataset[0]
    print(image.shape, label)
    print(image.dtype)

    # Plot the image and label
    plt.figure(1)
    plt.imshow(image.permute(1, 2, 0))
    plt.title(f"Label: {food3_test_dataset.classes[label]}")
    plt.axis("off")
    plt.show()

    def display_random_images(dataset: Dataset, n=3) -> None:
        """
        Displays n random images and labels from a given dataset.
        Args:
            dataset: PyTorch dataset object.
            n: number of images to display.
        """
        
        fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(20, 10))
        
        for i in range(n):
            ax = axes[i]
            random_index = random.randint(0, len(dataset))
            image, label = dataset[random_index]
            ax.imshow(image.permute(1, 2, 0))
            ax.set_title(f"Label: {dataset.classes[label]}")
            ax.axis("off")  
        plt.show()
        
    display_random_images(food3_test_dataset)


    # Build a data loader for the Food3 dataset
    BATCH_SIZE = 8
    WORKERS = os.cpu_count()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    food3_train_loader = DataLoader(food3_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True)
    food3_test_loader = DataLoader(food3_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=True)

    # adding data augmentation to the custom dataset class 

    # Define the transforms to apply to the data

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224), # crop the image to 224x224 pixels
        transforms.TrivialAugmentWide(num_magnitude_bins=31), # apply the trivial augmentations
        transforms.ToTensor() # convert the image to a PyTorch tensor
    ])

    # Create a custom dataset class for the Food3 dataset with data augmentation

    food3_train_dataset_augmented = Food3Dataset(train_dir, transform=train_transform)
    food3_test_dataset_augmented = Food3Dataset(test_dir, transform=train_transform)

    # visualize the images and labels after data augmentation

    display_random_images(food3_train_dataset_augmented,n=10)


    # Model 0 TinyVGG for classification

    class TinyVGG(nn.Module):
        """
        TinyVGG model for classification.
        """
        def __init__(self, in_channels: int, num_classes: int) -> None:
            """
            Args:
                in_channels: number of input channels.
                num_classes: number of classes.
            """
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
            self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(64 * 28 * 28*4, 128)
            self.fc2 = nn.Linear(128, num_classes)
            self.softmax = nn.Softmax(dim=1)
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass of the model.
            """
            
            # Convolutional block 1
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = self.pool(x) # output: (32, 64, 64)
            x = self.conv3(x)
            x = F.relu(x)
            x = self.conv4(x)
            x = F.relu(x)
            x = self.pool(x)
            x = x.reshape(x.shape[0], -1) # flatten the output for each image
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            return self.softmax(x)
        
        

    # Create a TinyVGG model
    torch.manual_seed(42)

    in_channels = 3
    num_classes = len(food3_train_dataset.classes)

    tiny_vgg = TinyVGG(in_channels=in_channels, num_classes=num_classes).to(device)

    # Build data loaders for the TinyVGG model
    BATCH_SIZE = 8
    WORKERS = os.cpu_count()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create a custom tensor to test the TinyVGG model
    x = torch.randn(1, in_channels, 224, 224).to(device)
    print(x.shape)

    tiny_vgg(x).shape

    train_data_loader = DataLoader(food3_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True)
    test_data_loader = DataLoader(food3_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=True)

    # Get a summary of the TinyVGG model
    summary(tiny_vgg, input_size=(1, 3, 224, 224))

    # Define the loss function
    loss_fn = nn.CrossEntropyLoss()

    # Define the optimizer

    LR = 0.001

    optimizer = optim.Adam(tiny_vgg.parameters(), lr=LR)

    # Some lists to keep track of training outputs
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    # Define the training function

    def train(model: nn.Module, 
            train_loader: DataLoader, 
            test_loader: DataLoader, 
            loss_fn: nn.Module, 
            optimizer: optim.Optimizer, 
            epochs: int, 
            device: str) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Trains a PyTorch model and returns train/test losses and accuracies.
        Args:
            model: PyTorch model to train.
            train_loader: PyTorch DataLoader for the training set.
            test_loader: PyTorch DataLoader for the test set.
            loss_fn: PyTorch loss function.
            optimizer: PyTorch optimizer.
            epochs: number of epochs to train the model for.
            device: device to move the model and data to (e.g. "cuda").
        Returns:
            Tuple of train losses, train accuracies, test losses and test accuracies.
        """
        
        # Some lists to keep track of training outputs
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []
        
        for epoch in range(epochs):
            
            # Training
            model.train()
            
            # Keep track of some metrics
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                
                # Move the data to device
                data = data.to(device=device)
                target = target.to(device=device)
                
                # Get model's predictions
                output = model(data)
                
                # Calculate the loss
                loss = loss_fn(output, target)
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                
                # Update the model parameters
                optimizer.step()
                
                # Keep track of loss and accuracy
                running_loss += loss.item()
                _, predictions = torch.max(output, dim=1)
                correct_predictions += torch.sum(predictions == target).item()
                total_predictions += predictions.shape[0]
                
                # Print metrics every 100 batches
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Accuracy: {correct_predictions/total_predictions:.4f}")
            
            # Save train loss and accuracy for later
            
            train_losses.append(running_loss/len(train_loader))
            train_accuracies.append(correct_predictions/total_predictions)


            # Evaluation
            model.eval()
            
            # Keep track of some metrics
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(test_loader):

                    # Move the data to device
                    data = data.to(device=device)
                    target = target.to(device=device)

                    # Get model's predictions
                    output = model(data)

                    # Calculate the loss
                    loss = loss_fn(output, target)

                    # Keep track of loss and accuracy
                    running_loss += loss.item()
                    _, predictions = torch.max(output, dim=1)
                    correct_predictions += torch.sum(predictions == target).item()
                    total_predictions += predictions.shape[0]
            
            # Save test loss and accuracy for later
            test_losses.append(running_loss/len(test_loader))
            test_accuracies.append(correct_predictions/total_predictions)
            
            # Print metrics every epoch
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {train_losses[-1]:.4f} | Accuracy: {train_accuracies[-1]:.4f} | Test loss: {test_losses[-1]:.4f} | Test accuracy: {test_accuracies[-1]:.4f}")
            
        return train_losses, train_accuracies, test_losses, test_accuracies

    # Train the TinyVGG model

    EPOCHS = 5

    train_losses, train_accuracies, test_losses, test_accuracies = train(tiny_vgg, train_data_loader, test_data_loader, loss_fn, optimizer, EPOCHS, device)

    # Plot the train and test losses
    plt.figure(1)
    plt.plot(train_losses, label="Train loss")
    plt.plot(test_losses, label="Test loss")
    plt.title("Losses")
    plt.legend()
    plt.show()

    # Plot the train and test accuracies

    plt.figure(2)
    plt.plot(train_accuracies, label="Train accuracy")
    plt.plot(test_accuracies, label="Test accuracy")
    plt.title("Accuracies")
    plt.legend()
    plt.show()

    # Define the evaluation function for the TinyVGG model

    def evaluate(model: nn.Module, test_loader: DataLoader, device: str) -> Tuple[float, float]:
        """
        Evaluates a PyTorch model on a given dataset.
        Args:
            model: PyTorch model to evaluate.
            test_loader: PyTorch DataLoader for the test set.
            device: device to move the model and data to (e.g. "cuda").
        Returns:
            Tuple of test loss and test accuracy.
        """
        
        # Evaluation
        model.eval()

        # Keep track of some metrics
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):

                # Move the data to device
                data = data.to(device=device)
                target = target.to(device=device)

                # Get model's predictions
                output = model(data)

                # Calculate the loss
                loss = loss_fn(output, target)

                # Keep track of loss and accuracy
                running_loss += loss.item()
                _, predictions = torch.max(output, dim=1)
                correct_predictions += torch.sum(predictions == target).item()
                total_predictions += predictions.shape[0]

        # Print metrics
        print(f"Loss: {running_loss/len(test_loader):.4f} | Accuracy: {correct_predictions/total_predictions:.4f}")
        
        return running_loss/len(test_loader), correct_predictions/total_predictions

    # Evaluate the TinyVGG model

    test_loss, test_accuracy = evaluate(tiny_vgg, test_data_loader, device)

    # Draw a confusion matrix for the TinyVGG model

    def get_predictions(model: nn.Module, test_loader: DataLoader, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the predictions and targets from a PyTorch model.
        Args:
            model: PyTorch model to evaluate.
            test_loader: PyTorch DataLoader for the test set.
            device: device to move the model and data to (e.g. "cuda").
        Returns:
            Tuple of predictions and targets.
        """
        
        # Evaluation
        model.eval()

        predictions = []
        targets = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):

                # Move the data to device
                data = data.to(device=device)
                target = target.to(device=device)

                # Get model's predictions
                output = model(data)

                # Get predictions and targets
                _, batch_predictions = torch.max(output, dim=1)
                predictions.extend(batch_predictions)
                targets.extend(target)

        predictions = torch.as_tensor(predictions)
        targets = torch.as_tensor(targets)
        
        return predictions, targets

    # Get predictions and targets for the TinyVGG model
    predictions, targets = get_predictions(tiny_vgg, test_data_loader, device)

    # Plot the confusion matrix

    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(targets.cpu(), predictions.cpu())
    plt.figure(3)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.show()


