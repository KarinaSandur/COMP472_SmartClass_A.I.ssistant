import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import zipfile
import tempfile

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

# Early stopping class
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

# Unzip files
def unzip_files(data_dir, temp_dir):
    folders = ['angry', 'focused', 'happy', 'neutral']
    for folder in folders:
        zip_path = os.path.join(data_dir, f"{folder}.zip")
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.join(temp_dir, folder))
        else:
            print(f"File not found: {zip_path}")

# Load data
def load_data(temp_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(temp_dir, transform=transform)
    return dataset

# Train and validate model
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

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1}, Training Loss: {running_loss / len(train_loader)}, Validation Loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(best_model_state)
    return model, best_val_loss

# Create confusion matrix
def create_confusion_matrix(y_true, y_pred):
    cm = np.zeros((4, 4), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true][pred] += 1
    return cm

# Visualize confusion matrix as a heatmap
def visualize_confusion_matrix(cm, name):
    plt.imshow(cm, cmap='Oranges', interpolation='nearest')
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

# k-fold cross-validation
def k_fold_cross_validation(model_class, dataset, k=10, num_epochs=10, batch_size=32):
    # initialize KFold with the number of splits (10), shuffle, and a random seed (42) for reproducibility
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []

    best_val_loss = float('inf')
    best_model_state = None

    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
        print(f"Fold {fold + 1}")

        # create subsets for training + testing based on the current folds indices
        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)

        # split the training subset into a training and validation set
        train_data, val_data = train_test_split(train_subset, test_size=0.15, random_state=42)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

        # initialize model, loss criterion, and optimizer inside the loop
        model = model_class()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model, val_loss = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

        model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.numpy())
                y_pred.extend(predicted.numpy())

        # calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro')
        recall_macro = recall_score(y_true, y_pred, average='macro')
        f1_macro = f1_score(y_true, y_pred, average='macro')

        precision_micro = precision_score(y_true, y_pred, average='micro')
        recall_micro = recall_score(y_true, y_pred, average='micro')
        f1_micro = f1_score(y_true, y_pred, average='micro')

        cm = create_confusion_matrix(y_true, y_pred)
        visualize_confusion_matrix(cm, f"{model_class.__name__} - Fold {fold + 1}")

        # append metrics to the fold results
        fold_results.append((accuracy, precision_macro, recall_macro, f1_macro, precision_micro, recall_micro, f1_micro))

    # save the best model after all folds are completed
    torch.save(best_model_state, 'k_fold_best_model.pth')

    return fold_results


# evaluate main model using 10-fold cross-validation
if __name__ == "__main__":
    data_dir = input("Enter the directory path where your zip files are located: ")
    batch_size = 32
    num_epochs = 10

    with tempfile.TemporaryDirectory() as temp_dir:
        unzip_files(data_dir, temp_dir)
        dataset = load_data(temp_dir)

        print("Evaluating Main Model with 10-fold cross-validation")

        # perform k-fold cross-validation and get the results
        fold_results = k_fold_cross_validation(MainModel, dataset, k=10, num_epochs=num_epochs, batch_size=batch_size)

        # print the results
        print("\nFold | Macro Precision | Macro Recall | Macro F1 | Micro Precision | Micro Recall | Micro F1 | Accuracy")
        print("-" * 90)
        for i, (acc, prec_macro, rec_macro, f1_macro, prec_micro, rec_micro, f1_micro) in enumerate(fold_results):
            print(f"{i + 1:<5}| {prec_macro:.4f}          | {rec_macro:.4f}        | {f1_macro:.4f}   | {prec_micro:.4f}           | {rec_micro:.4f}         | {f1_micro:.4f}   | {acc:.4f}")

        # Calculate and print the average metrics
        avg_metrics = np.mean(fold_results, axis=0)
        print("\nAverage Metrics")
        print(f"Macro Precision: {avg_metrics[1]:.4f}")
        print(f"Macro Recall: {avg_metrics[2]:.4f}")
        print(f"Macro F1: {avg_metrics[3]:.4f}")
        print(f"Micro Precision: {avg_metrics[4]:.4f}")
        print(f"Micro Recall: {avg_metrics[5]:.4f}")
        print(f"Micro F1: {avg_metrics[6]:.4f}")
        print(f"Accuracy: {avg_metrics[0]:.4f}")
