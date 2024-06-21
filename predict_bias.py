import os
import zipfile
import cnn_model
import tempfile
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

from evaluateModels import create_confusion_matrix, visualize_confusion_matrix


# def unzip_files(data_dir, temp_dir):
#     folders = ['angry', 'focused', 'happy', 'neutral']
#     for folder in folders:
#         zip_path = os.path.join(data_dir, f"{folder}.zip")
#         if os.path.exists(zip_path):
#             with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#                 zip_ref.extractall(os.path.join(temp_dir, folder))
#         else:
#             print(f"File not found: {zip_path}")
torch.manual_seed(42)

def count_files_in_folders(directory):
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        if os.path.isdir(folder_path):
            num_files = 0
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if zipfile.is_zipfile(file_path):
                        with zipfile.ZipFile(file_path, 'r') as zip_ref:
                            num_files += len(zip_ref.namelist())
                    else:
                        num_files += 1
            print(f"Folder '{folder}' contains {num_files} files.")


if __name__ == "__main__":
    data_dir = input("your_directory_path_here: ")
    batch_size = 32

    # count_files_in_folders(data_dir)

    # initialize model
    mainModel = cnn_model.MainModel()

    # load model
    mainModel.load_state_dict(torch.load("best_model_MainModel.pth"))

    results = {}
        
    with tempfile.TemporaryDirectory() as temp_dir:
        for folder in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder)
            if os.path.isdir(folder_path):
                cnn_model.unzip_files(folder_path, temp_dir)
                print(f"Unzipped files from {folder_path} to {temp_dir}")
                dataset = cnn_model.load_data(temp_dir)

                # split dataset into training 70%, validation 15%, and testing 15%
                train_data, test_data = train_test_split(dataset, test_size=0.3, random_state=42)
                val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)
                
                # create data loaders for training, validation, and testing sets
                train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
                test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

                # Evaluate the models
                models = {
                    'Main Model': mainModel
                    # 'Variant 1': var1Model,
                    # 'Variant 2': var2Model
                }

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

                    # Create confusion matrix and display it
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