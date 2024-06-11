import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split, Subset
import zipfile
import tempfile
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# main model:
# number of convolutional layers: 2
# conv1: 3x3 kernel
# max pooling: 2x2 kernel (better performance)
# conv2: 3x3 kernel
class MainModel(nn.Module):
    def __init__(self):
        super(MainModel, self).__init__()
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

# variant 1:
# number of convolutional layers: 3
# conv1: 5x5 kernel
# max pooling: 2x2 kernel (better performance)
# conv2: 5x5 kernel
# conv3: 5x5 kernel
class Variant1(nn.Module):
        def __init__(self):
            super(Variant1, self).__init__()
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

# variant 2:
# number of convolutional layers: 2
# conv1: 7x7 kernel
# max pooling: 3x3 kernel (better performance)
# conv2: 7x7 kernel
class Variant2(nn.Module):
    def __init__(self):
        super(Variant2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=1)
        # modify kernel size here
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=1)
        self.conv_output_size = self._get_conv_output_size()
        self.fc1 = nn.Linear(self.conv_output_size, 256)
        self.fc2 = nn.Linear(256, 4)

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
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(temp_dir, transform=transform)
    return dataset


# train and validate model
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10):
    early_stopping = EarlyStopping(patience=3, min_delta=0.001)
    best_val_loss = float('inf')
    best_perf_model_state = None

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
            best_perf_model_state = model.state_dict()
            # save the best model state
            torch.save(best_perf_model_state, 'best_performing_model.pth')

        early_stopping(val_loss / len(val_loader))
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Load the best model state before returning
    model.load_state_dict(best_perf_model_state)

    # Return the best model state
    return model

 # Function to create confusion matrix
def create_confusion_matrix(y_true, y_pred):
    cm = np.zeros((4, 4), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true][pred] += 1
    return cm

# Visualize confusion matrix as a heatmap
def visualize_confusion_matrix(cm, name):
    plt.imshow(cm, cmap='Oranges', interpolation='nearest')

    # Add numbers in cell
    for i in range(4):
        for j in range(4):
            plt.text(j, i, str(cm[i][j]), ha='center', va='center', color='black')
            
    plt.colorbar()
    plt.xticks(range(4), ['angry', 'focused', 'happy', 'neutral'])
    plt.yticks(range(4), ['angry', 'focused', 'happy', 'neutral'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix for ' + f"{name}")
    plt.show()


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
        train_model(main_model, criterion, optimizer_main, train_loader, val_loader, num_epochs)
        print("Model: Variant 1")
        train_model(variant1, criterion, optimizer_variant1, train_loader, val_loader, num_epochs)
        print("Model: Variant 2")
        train_model(variant2, criterion, optimizer_variant2, train_loader, val_loader, num_epochs)
        print("\n")
        print("\n")

        # evaluate the models
        models = {
            'Main Model': main_model,
            'Variant 1': variant1,
            'Variant 2': variant2
        }

        results = {}
        
        for name, model in models.items():
            model.eval()
            y_true = []
            y_pred = []

            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    y_true.extend(labels.numpy())
                    y_pred.extend(predicted.numpy())

            # calculate metrics: macro and micro
            accuracy = accuracy_score(y_true, y_pred)
            precision_macro = precision_score(y_true, y_pred, average='macro')
            recall_macro = recall_score(y_true, y_pred, average='macro')
            f1_macro = f1_score(y_true, y_pred, average='macro')
            precision_micro = precision_score(y_true, y_pred, average='micro')
            recall_micro = recall_score(y_true, y_pred, average='micro')
            f1_micro = f1_score(y_true, y_pred, average='micro')

            # Create  confusion matrix
            cm = create_confusion_matrix(y_true, y_pred)
            visualize_confusion_matrix(cm, name)
            

            results[name] = {
                'accuracy': accuracy,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_macro': f1_macro,
                'precision_micro': precision_micro,
                'recall_micro': recall_micro,
                'f1_micro': f1_micro,
                'confusion_matrix': cm
            }

        # print results in terminal
        for name, metrics in results.items():
            print(f"Model: {name}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Macro-Precision: {metrics['precision_macro']:.4f}")
            print(f"Macro-Recall: {metrics['recall_macro']:.4f}")
            print(f"Macro-F1-Score: {metrics['f1_macro']:.4f}")
            print(f"Micro-Precision: {metrics['precision_micro']:.4f}")
            print(f"Micro-Recall: {metrics['recall_micro']:.4f}")
            print(f"Micro-F1-Score: {metrics['f1_micro']:.4f}")
            print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
            print()
        
        # Results from different models
        main_model_info = results.get("Main Model",{})
        mm_accuracy = round(main_model_info.get("accuracy"), 4)
        mm_precision_macro = round(main_model_info.get("precision_macro"), 4)
        mm_recall_macro = round(main_model_info.get("recall_macro"), 4)
        mm_f1_macro = round(main_model_info.get("f1_macro"), 4)
        mm_precision_micro = round(main_model_info.get("precision_micro"), 4)
        mm_recall_micro = round(main_model_info.get("recall_micro"), 4)
        mm_f1_micro = round(main_model_info.get("f1_micro"), 4)

        variant1_info = results.get('Variant 1', {})
        v1_accuracy = round(variant1_info.get("accuracy"), 4)
        v1_precision_macro = round(variant1_info.get("precision_macro"), 4)
        v1_recall_macro = round(variant1_info.get("recall_macro"), 4)
        v1_f1_macro = round(variant1_info.get("f1_macro"), 4)
        v1_precision_micro = round(variant1_info.get("precision_micro"), 4)
        v1_recall_micro = round(variant1_info.get("recall_micro"), 4)
        v1_f1_micro = round(variant1_info.get("f1_micro"), 4)

        variant2_info = results.get('Variant 2', {})
        v2_accuracy = round(variant2_info.get("accuracy"), 4)
        v2_precision_macro = round(variant2_info.get("precision_macro"), 4)
        v2_recall_macro = round(variant2_info.get("recall_macro"), 4)
        v2_f1_macro = round(variant2_info.get("f1_macro"), 4)
        v2_precision_micro = round(variant2_info.get("precision_micro"), 4)
        v2_recall_micro = round(variant2_info.get("recall_micro"), 4)
        v2_f1_micro = round(variant2_info.get("f1_micro"), 4)

        # Initialize data that will go in table
        data = [
            ['Model', 'Macro P', 'Macro R', 'Macro F', 'Micro P', 'Micro R', 'Micro F', 'Accuracy'],
            ["Main Model", mm_precision_macro, mm_recall_macro, mm_f1_macro, mm_precision_micro, mm_recall_micro, mm_f1_micro, mm_accuracy],
            ["Variation 1", v1_precision_macro, v1_recall_macro, v1_f1_macro, v1_precision_micro, v1_recall_micro, v1_f1_micro, v1_accuracy],
            ["Variation 2", v2_precision_macro, v2_recall_macro, v2_f1_macro, v2_precision_micro, v2_recall_micro, v2_f1_micro, v2_accuracy],
        ]

        # Create table
        fig, ax = plt.subplots()
        ax.axis('off')  # Hide axes
        ax.table(cellText=data, loc='center')

        plt.show()

# to use the trained model (after running this code and generating the file):
# model.load_state_dict(torch.load('best_performing_model.pth'))
# for this part: You also must have a separate Python program that can load and run the saved model,
# both on a complete dataset and an individual image (evaluation/application mode).
