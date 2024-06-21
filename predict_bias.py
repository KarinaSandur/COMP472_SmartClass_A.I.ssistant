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

    # Obtain folder paths where data is located from user
    clean_data_age = input("Enter the path where you Clean Data Age folder is located: ")
    clean_data_gender = input("Enter the path where you Clean Data Gender folder is located: ")
    
    # Initialize batch size
    batch_size = 32

    # Input paths where data is located
    paths = {
        clean_data_age,
        clean_data_gender,
    }

    # initialize model
    mainModel = cnn_model.MainModel()

    # load model
    mainModel.load_state_dict(torch.load("best_model_MainModel.pth"))

    # Declare results
    results = {}

    # Obtains metrics for every folder in every path given as input
    for data_dir in paths:  
        with tempfile.TemporaryDirectory() as temp_dir:
            for folder in os.listdir(data_dir):
                folder_path = os.path.join(data_dir, folder)
                if os.path.isdir(folder_path):
                    cnn_model.unzip_files(folder_path, temp_dir)
                    print(f"Unzipped files from {folder_path} to {temp_dir}")
                    dataset = cnn_model.load_data(temp_dir)

                    # Split dataset into training 70%, validation 15%, and testing 15%
                    train_data, test_data = train_test_split(dataset, test_size=0.3, random_state=42)
                    val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)
                    
                    # Create data loaders for training, validation, and testing sets
                    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
                    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

                    # Evaluate using main model
                    mainModel.eval()
                    y_true = []
                    y_pred = []

                    with torch.no_grad():
                        for inputs, labels in test_loader:
                            outputs = mainModel(inputs)
                            _, predicted = torch.max(outputs, 1)
                            y_true.extend(labels.numpy())
                            y_pred.extend(predicted.numpy())

                    # Calculate macro metrics
                    accuracy = accuracy_score(y_true, y_pred)
                    precision_macro = precision_score(y_true, y_pred, average='macro')
                    recall_macro = recall_score(y_true, y_pred, average='macro')
                    f1_macro = f1_score(y_true, y_pred, average='macro')

                    # Save metrics in results
                    results[folder] = {
                        'accuracy': accuracy,
                        'precision_macro': precision_macro,
                        'recall_macro': recall_macro,
                        'f1_macro': f1_macro,
                    }

    print(results)
    print("\n")
    print("\n")
    

    # Results from different age groups
    young_info = results.get("young",{})
    young_accuracy = round(young_info.get("accuracy"), 4)
    young_precision_macro = round(young_info.get("precision_macro"), 4)
    young_recall_macro = round(young_info.get("recall_macro"), 4)
    young_f1_macro = round(young_info.get("f1_macro"), 4)

    middle_aged_info = results.get("middle-aged",{})
    middle_aged_accuracy = round(middle_aged_info.get("accuracy"), 4)
    middle_aged_precision_macro = round(middle_aged_info.get("precision_macro"), 4)
    middle_aged_recall_macro = round(middle_aged_info.get("recall_macro"), 4)
    middle_aged_f1_macro = round(middle_aged_info.get("f1_macro"), 4)

    senior_info = results.get("senior",{})
    senior_accuracy = round(senior_info.get("accuracy"), 4)
    senior_precision_macro = round(senior_info.get("precision_macro"), 4)
    senior_recall_macro = round(senior_info.get("recall_macro"), 4)
    senior_f1_macro = round(senior_info.get("f1_macro"), 4)

    # Results from different genders
    men_info = results.get("men",{})
    men_accuracy = round(men_info.get("accuracy"), 4)
    men_precision_macro = round(men_info.get("precision_macro"), 4)
    men_recall_macro = round(men_info.get("recall_macro"), 4)
    men_f1_macro = round(men_info.get("f1_macro"), 4)

    women_info = results.get("women",{})
    women_accuracy = round(women_info.get("accuracy"), 4)
    women_precision_macro = round(women_info.get("precision_macro"), 4)
    women_recall_macro = round(women_info.get("recall_macro"), 4)
    women_f1_macro = round(women_info.get("f1_macro"), 4)


    # Calculating Averages for Age Group Metrics
    age_accuracy_avg = round(((young_accuracy + middle_aged_accuracy + senior_accuracy)/3), 4)
    age_precision_avg = round(((young_precision_macro + middle_aged_precision_macro + senior_precision_macro)/3), 4)
    age_recall_avg = round(((young_recall_macro + middle_aged_recall_macro + senior_recall_macro)/3), 4)
    age_f1_avg = round(((young_f1_macro + middle_aged_f1_macro + senior_f1_macro)/3), 4)

    # Calculating Averages for Gender Metrics
    gender_accuracy_avg = round(((men_accuracy + women_accuracy)/2), 4)
    gender_precision_avg = round(((men_precision_macro + women_precision_macro)/2), 4)
    gender_recall_avg = round(((men_recall_macro + women_recall_macro)/2), 4)
    gender_f1_avg = round(((men_f1_macro + women_f1_macro)/2), 4)

    # Initialize data that will go in table
    data = [
        ['Group', '# Images', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
        ["Young", "10", young_accuracy, young_precision_macro, young_recall_macro, young_f1_macro],
        ["Middle-Aged", "10", middle_aged_accuracy, middle_aged_precision_macro, middle_aged_recall_macro, middle_aged_f1_macro],
        ["Senior", "10", senior_accuracy, senior_precision_macro, senior_recall_macro, senior_f1_macro],
        ["Total/Average", "30", age_accuracy_avg, age_precision_avg, age_recall_avg, age_f1_avg],
        ["Male", "10", men_accuracy, men_precision_macro, men_recall_macro, men_f1_macro],
        ["Female", "10", women_accuracy, women_precision_macro, women_recall_macro, women_f1_macro],
        ["Total/Average", "30", gender_accuracy_avg, gender_precision_avg, gender_recall_avg, gender_f1_avg],
    ]

    # Create table
    fig, ax = plt.subplots()
    ax.axis('off')  # Hide axes
    ax.table(cellText=data, loc='center')

    # Display table
    plt.show()
