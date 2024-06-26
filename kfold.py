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

        # split the training subset into training and validation sets
        # further divide the training set into a training subset and a validation subset
        # train_data contains 85% of samples from train_subset using train_indices
        # val_data contains 15% samples from train_subset using val_indices
        train_indices, val_indices = train_test_split(list(range(len(train_subset))), test_size=0.15, random_state=42)
        train_data = Subset(train_subset, train_indices)
        val_data = Subset(train_subset, val_indices)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

        # initialize new model within each fold, loss criterion, and optimizer inside the loop
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

    results = {}

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
            fold = i + 1
            results[fold] = {
                'precision_macro': prec_macro,
                'recall_macro': rec_macro,
                'f1_macro': f1_macro,
                'precision_micro': prec_micro,
                'recall_micro': rec_micro,
                'f1_micro': f1_micro,
                'accuracy': acc,
            }

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
    
    avg_macro_prec = f"{avg_metrics[1]:.4f}"
    avg_macro_rec = f"{avg_metrics[2]:.4f}"
    avg_macro_f1 = f"{avg_metrics[3]:.4f}"
    avg_micro_prec = f"{avg_metrics[4]:.4f}"
    avg_micro_rec = f"{avg_metrics[5]:.4f}"
    avg_micro_f1 = f"{avg_metrics[6]:.4f}"
    avg_acc = f"{avg_metrics[0]:.4f}"
    

    # Fold 1 Metrics
    fold1_info = results.get(1, {})
    fold1_prec_macro = round(fold1_info.get("precision_macro"), 4)
    fold1_recall_macro = round(fold1_info.get("recall_macro"), 4)
    fold1_f1_macro = round(fold1_info.get("f1_macro"), 4)
    fold1_prec_micro = round(fold1_info.get("precision_micro"), 4)
    fold1_recall_micro = round(fold1_info.get("recall_macro"), 4)
    fold1_f1_micro = round(fold1_info.get("f1_micro"), 4)
    fold1_accuracy = round(fold1_info.get("accuracy"), 4)

    # Fold 2 Metrics
    fold2_info = results.get(2, {})
    fold2_prec_macro = round(fold2_info.get("precision_macro"), 4)
    fold2_recall_macro = round(fold2_info.get("recall_macro"), 4)
    fold2_f1_macro = round(fold2_info.get("f1_macro"), 4)
    fold2_prec_micro = round(fold2_info.get("precision_micro"), 4)
    fold2_recall_micro = round(fold2_info.get("recall_macro"), 4)
    fold2_f1_micro = round(fold2_info.get("f1_micro"), 4)
    fold2_accuracy = round(fold2_info.get("accuracy"), 4)

    # Fold 3 Metrics
    fold3_info = results.get(3, {})
    fold3_prec_macro = round(fold3_info.get("precision_macro"), 4)
    fold3_recall_macro = round(fold3_info.get("recall_macro"), 4)
    fold3_f1_macro = round(fold3_info.get("f1_macro"), 4)
    fold3_prec_micro = round(fold3_info.get("precision_micro"), 4)
    fold3_recall_micro = round(fold3_info.get("recall_macro"), 4)
    fold3_f1_micro = round(fold3_info.get("f1_micro"), 4)
    fold3_accuracy = round(fold3_info.get("accuracy"), 4)

    # Fold 4 Metrics
    fold4_info = results.get(4, {})
    fold4_prec_macro = round(fold4_info.get("precision_macro"), 4)
    fold4_recall_macro = round(fold4_info.get("recall_macro"), 4)
    fold4_f1_macro = round(fold4_info.get("f1_macro"), 4)
    fold4_prec_micro = round(fold4_info.get("precision_micro"), 4)
    fold4_recall_micro = round(fold4_info.get("recall_macro"), 4)
    fold4_f1_micro = round(fold4_info.get("f1_micro"), 4)
    fold4_accuracy = round(fold4_info.get("accuracy"), 4)

    # Fold 5 Metrics
    fold5_info = results.get(5, {})
    fold5_prec_macro = round(fold5_info.get("precision_macro"), 4)
    fold5_recall_macro = round(fold5_info.get("recall_macro"), 4)
    fold5_f1_macro = round(fold5_info.get("f1_macro"), 4)
    fold5_prec_micro = round(fold5_info.get("precision_micro"), 4)
    fold5_recall_micro = round(fold5_info.get("recall_macro"), 4)
    fold5_f1_micro = round(fold5_info.get("f1_micro"), 4)
    fold5_accuracy = round(fold5_info.get("accuracy"), 4)

    # Fold 6 Metrics
    fold6_info = results.get(6, {})
    fold6_prec_macro = round(fold6_info.get("precision_macro"), 4)
    fold6_recall_macro = round(fold6_info.get("recall_macro"), 4)
    fold6_f1_macro = round(fold6_info.get("f1_macro"), 4)
    fold6_prec_micro = round(fold6_info.get("precision_micro"), 4)
    fold6_recall_micro = round(fold6_info.get("recall_macro"), 4)
    fold6_f1_micro = round(fold6_info.get("f1_micro"), 4)
    fold6_accuracy = round(fold6_info.get("accuracy"), 4)

    # Fold 7 Metrics
    fold7_info = results.get(7, {})
    fold7_prec_macro = round(fold7_info.get("precision_macro"), 4)
    fold7_recall_macro = round(fold7_info.get("recall_macro"), 4)
    fold7_f1_macro = round(fold7_info.get("f1_macro"), 4)
    fold7_prec_micro = round(fold7_info.get("precision_micro"), 4)
    fold7_recall_micro = round(fold7_info.get("recall_macro"), 4)
    fold7_f1_micro = round(fold7_info.get("f1_micro"), 4)
    fold7_accuracy = round(fold7_info.get("accuracy"), 4)

    # Fold 8 Metrics
    fold8_info = results.get(8, {})
    fold8_prec_macro = round(fold8_info.get("precision_macro"), 4)
    fold8_recall_macro = round(fold8_info.get("recall_macro"), 4)
    fold8_f1_macro = round(fold8_info.get("f1_macro"), 4)
    fold8_prec_micro = round(fold8_info.get("precision_micro"), 4)
    fold8_recall_micro = round(fold8_info.get("recall_macro"), 4)
    fold8_f1_micro = round(fold8_info.get("f1_micro"), 4)
    fold8_accuracy = round(fold8_info.get("accuracy"), 4)

    # Fold 9 Metrics
    fold9_info = results.get(9, {})
    fold9_prec_macro = round(fold9_info.get("precision_macro"), 4)
    fold9_recall_macro = round(fold9_info.get("recall_macro"), 4)
    fold9_f1_macro = round(fold9_info.get("f1_macro"), 4)
    fold9_prec_micro = round(fold9_info.get("precision_micro"), 4)
    fold9_recall_micro = round(fold9_info.get("recall_macro"), 4)
    fold9_f1_micro = round(fold9_info.get("f1_micro"), 4)
    fold9_accuracy = round(fold9_info.get("accuracy"), 4)

    # Fold 10 Metrics
    fold10_info = results.get(10, {})
    fold10_prec_macro = round(fold10_info.get("precision_macro"), 4)
    fold10_recall_macro = round(fold10_info.get("recall_macro"), 4)
    fold10_f1_macro = round(fold10_info.get("f1_macro"), 4)
    fold10_prec_micro = round(fold10_info.get("precision_micro"), 4)
    fold10_recall_micro = round(fold10_info.get("recall_macro"), 4)
    fold10_f1_micro = round(fold10_info.get("f1_micro"), 4)
    fold10_accuracy = round(fold10_info.get("accuracy"), 4)


    # Initialize data that will go in table
    data = [
        ['Fold', 'Macro Precision', 'Macro Recall', 'Macro F1', 'Micro Precision', 'Micro Recall', 'Micro F1', 'Accuracy'],
        ["1", fold1_prec_macro, fold1_recall_macro, fold1_f1_macro, fold1_prec_micro, fold1_recall_micro, fold1_f1_micro, fold1_accuracy],
        ["2", fold2_prec_macro, fold2_recall_macro, fold2_f1_macro, fold2_prec_micro, fold2_recall_micro, fold2_f1_micro, fold2_accuracy],
        ["3", fold3_prec_macro, fold3_recall_macro, fold3_f1_macro, fold3_prec_micro, fold3_recall_micro, fold3_f1_micro, fold3_accuracy],
        ["4", fold4_prec_macro, fold4_recall_macro, fold4_f1_macro, fold4_prec_micro, fold4_recall_micro, fold4_f1_micro, fold4_accuracy],
        ["5", fold5_prec_macro, fold5_recall_macro, fold5_f1_macro, fold5_prec_micro, fold5_recall_micro, fold5_f1_micro, fold5_accuracy],
        ["6", fold6_prec_macro, fold6_recall_macro, fold6_f1_macro, fold6_prec_micro, fold6_recall_micro, fold6_f1_micro, fold6_accuracy],
        ["7", fold7_prec_macro, fold7_recall_macro, fold7_f1_macro, fold7_prec_micro, fold7_recall_micro, fold7_f1_micro, fold7_accuracy],
        ["8", fold8_prec_macro, fold8_recall_macro, fold8_f1_macro, fold8_prec_micro, fold8_recall_micro, fold8_f1_micro, fold8_accuracy],
        ["9", fold9_prec_macro, fold9_recall_macro, fold9_f1_macro, fold9_prec_micro, fold9_recall_micro, fold9_f1_micro, fold9_accuracy],
        ["10", fold10_prec_macro, fold10_recall_macro, fold10_f1_macro, fold10_prec_micro, fold10_recall_micro, fold10_f1_micro, fold10_accuracy],
        ["Average", avg_macro_prec, avg_macro_rec, avg_macro_f1, avg_micro_prec, avg_micro_rec, avg_micro_f1, avg_acc]
        
    ]

    

    # Create table
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.axis('off')
    ax.table(cellText=data, loc='center')

    # Display table
    plt.show()