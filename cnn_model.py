import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import zipfile
import tempfile
from PIL import Image

# main Model
class MainModel(nn.Module):
    def __init__(self):
        super(MainModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=0, padding=0)
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

# variant 1
class Variant1(nn.Module):
        def __init__(self):
            super(Variant1, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
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

# variant 2
class Variant2(nn.Module):
    def __init__(self):
        super(Variant2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
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
    folders = ['angry', 'focused', 'neutral', 'happy']
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
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(temp_dir, transform=transform)
    return dataset


# train and validate model
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10):
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
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

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print(
            f"Epoch {epoch + 1}, Training Loss: {running_loss / len(train_loader)}, Validation Loss: {val_loss / len(val_loader)}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

        early_stopping(val_loss / len(val_loader))
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # save the best performing model
    torch.save(best_model_state, 'best_model.pth')

if __name__ == "__main__":
    data_dir = input("Enter the directory path where your zip files are located: ")
    batch_size = 32
    num_epochs = 10

    with tempfile.TemporaryDirectory() as temp_dir:
        unzip_files(data_dir, temp_dir)
        dataset = load_data(temp_dir)

        # split dataset into training and validation sets: 80% training, 20% validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # create data loaders for training and validation sets
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # initialize and train your models
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
        train_model(main_model, criterion, optimizer_main, train_loader, val_loader, num_epochs)
        print("Model: Variant 1")
        train_model(variant1, criterion, optimizer_variant1, train_loader, val_loader, num_epochs)
        print("Model: Variant 2")
        train_model(variant2, criterion, optimizer_variant2, train_loader, val_loader, num_epochs)