import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import zipfile
import tempfile
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# set random seed for consistent runs (if we want each run to be different, remove this part)
torch.manual_seed(42)

# main model:
# number of convolutional layers: 3
# conv1: 3x3 kernel
# max pooling: 2x2 kernel (better performance)
# conv2: 3x3 kernel
# conv3: 3x3 kernel
class MainModel(nn.Module):
    def __init__(self):
        super(MainModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        # modify kernel size here
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv_output_size = self._get_conv_output_size()
        self.fc1 = nn.Linear(self.conv_output_size, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.conv_output_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def _get_conv_output_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 3, 150, 150)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            return x.view(1, -1).size(1)
        
# variant 1:
# number of convolutional layers: 2
# conv1: 3x3 kernel
# max pooling: 2x2 kernel (better performance)
# conv2: 3x3 kernel
class Variant1(nn.Module):
    def __init__(self):
        super(Variant1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        # modify kernel size here
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv_output_size = self._get_conv_output_size()
        self.fc1 = nn.Linear(self.conv_output_size, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.conv_output_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def _get_conv_output_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 3, 150, 150)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            return x.view(1, -1).size(1)

# variant 2:
# number of convolutional layers: 3
# conv1: 5x5 kernel
# max pooling: 2x2 kernel (better performance)
# conv2: 5x5 kernel
# conv3: 5x5 kernel
class Variant2(nn.Module):
        def __init__(self):
            super(Variant2, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=1)
            # modify kernel size here
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=1)
            self.conv_output_size = self._get_conv_output_size()
            self.fc1 = nn.Linear(self.conv_output_size, 256)
            self.fc2 = nn.Linear(256, 4)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, self.conv_output_size)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

        def _get_conv_output_size(self):
            with torch.no_grad():
                x = torch.zeros(1, 3, 150, 150)
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = self.pool(F.relu(self.conv3(x)))
                return x.view(1, -1).size(1)



# early stopping class to monitor validation loss and stop training if it doesn't improve
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


# unzip files into a temporary directory
def unzip_files(data_dir, temp_dir):
    folders = ['angry', 'focused', 'happy', 'neutral']
    for folder in folders:
        zip_path = os.path.join(data_dir, f"{folder}.zip")
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.join(temp_dir, folder))
        else:
            print(f"File not found: {zip_path}")


# load data
def load_data(temp_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(temp_dir, transform=transform)
    return dataset


# train and validate model
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10):
    early_stopping = EarlyStopping(patience=3, min_delta=0.001)
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # initialize validation loss
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print(
            f"Epoch {epoch + 1}, Training Loss: {running_loss / len(train_loader)}, Validation Loss: {val_loss / len(val_loader)}")

        # check if the current validation loss is better than the best validation loss
        # update the best validation loss, save state of best model to file
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, f"best_model_{model.__class__.__name__}.pth")

        # check if early stopping condition is met
        early_stopping(val_loss / len(val_loader))
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # return model
    return model

# #################################################################################################
# MAIN METHOD
# #################################################################################################

if __name__ == "__main__":
    data_dir = input("Enter the directory path where your zip files are located: ")
    batch_size = 32
    num_epochs = 10

    with tempfile.TemporaryDirectory() as temp_dir:
        unzip_files(data_dir, temp_dir)
        dataset = load_data(temp_dir)

        # split dataset into training 70%, validation 15%, and testing 15%
        train_data, test_data = train_test_split(dataset, test_size=0.3, random_state=42)
        val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

        # create data loaders for training, validation, and testing sets
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        # initialize and train models
        main_model = MainModel()
        variant1 = Variant1()
        variant2 = Variant2()

        # define the loss function (criterion)
        criterion = nn.CrossEntropyLoss()

        # define the optimizer for each model
        optimizer_main = torch.optim.Adam(main_model.parameters(), lr=0.001)
        optimizer_variant1 = torch.optim.Adam(variant1.parameters(), lr=0.001)
        optimizer_variant2 = torch.optim.Adam(variant2.parameters(), lr=0.001)

        # train each model with early stopping
        print("Model: Main Model")
        main_model = train_model(main_model, criterion, optimizer_main, train_loader, val_loader, num_epochs)
        print("Model: Variant 1")
        variant1 = train_model(variant1, criterion, optimizer_variant1, train_loader, val_loader, num_epochs)
        print("Model: Variant 2")
        variant2 = train_model(variant2, criterion, optimizer_variant2, train_loader, val_loader, num_epochs)
        print("\n")
        print("Finished training all models!")
