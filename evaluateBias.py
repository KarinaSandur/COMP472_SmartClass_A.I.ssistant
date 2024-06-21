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


def count_files(directory):
    num_files = 0
    for root, dirs, files in os.walk(directory):
        num_files += len(files)
    return num_files


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
    count_images = {}

    # Obtains metrics for every folder in every path given as input
    for data_dir in paths:  
        with tempfile.TemporaryDirectory() as temp_dir:
            for folder in os.listdir(data_dir):
                folder_path = os.path.join(data_dir, folder)
                if os.path.isdir(folder_path):

                    # Unzip Files
                    cnn_model.unzip_files(folder_path, temp_dir)
                    print(f"Unzipped files from {folder_path} to {temp_dir}")

                    # Get number of images in file
                    num_of_images = count_files(temp_dir)

                    # Save number of images in file
                    count_images[folder] = {
                        'Number of Images': num_of_images
                    }

                    # Load dataset
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

                    # Save macro metrics in results
                    results[folder] = {
                        'accuracy': accuracy,
                        'precision_macro': precision_macro,
                        'recall_macro': recall_macro,
                        'f1_macro': f1_macro,
                    }

    print("Generating table of metrics....")

    # Number of Images for each age group
    num_of_young = (count_images.get('young')).get("Number of Images")
    num_of_middle_aged = (count_images.get('middle-aged')).get("Number of Images")
    num_of_senior = (count_images.get('senior')).get("Number of Images")
    total_age = num_of_young + num_of_middle_aged + num_of_senior

    # Number of Images for each gender
    num_of_men = (count_images.get('men')).get("Number of Images")
    num_of_women = (count_images.get('women')).get("Number of Images")
    total_gender = num_of_men + num_of_women

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

    # Overall System Calculations
    total_images = total_gender + total_age
    avg_overall_accuracy = round(((young_accuracy + middle_aged_accuracy + senior_accuracy + men_accuracy + women_accuracy)/5),4)
    avg_overall_precision = round(((young_precision_macro + middle_aged_precision_macro + senior_precision_macro + men_precision_macro + women_precision_macro)/5),4)
    avg_overall_recall = round(((young_recall_macro + middle_aged_recall_macro + senior_recall_macro + men_recall_macro + women_recall_macro)/5),4)
    avg_overall_f1 = round(((young_f1_macro + middle_aged_f1_macro + senior_f1_macro + men_f1_macro + women_f1_macro)/5),4)

    # Initialize data that will go in table
    data = [
        ['Attribute', 'Group', '# Images', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
        ['Age',"Young", num_of_young, young_accuracy, young_precision_macro, young_recall_macro, young_f1_macro],
        ['Age', "Middle-Aged", num_of_middle_aged, middle_aged_accuracy, middle_aged_precision_macro, middle_aged_recall_macro, middle_aged_f1_macro],
        ['Age',"Senior", num_of_senior, senior_accuracy, senior_precision_macro, senior_recall_macro, senior_f1_macro],
        ['Age',"Total/Average", total_age, age_accuracy_avg, age_precision_avg, age_recall_avg, age_f1_avg],
        ['Gender',"Male", num_of_men, men_accuracy, men_precision_macro, men_recall_macro, men_f1_macro],
        ['Gender',"Female", num_of_women, women_accuracy, women_precision_macro, women_recall_macro, women_f1_macro],
        ['Gender',"Total/Average", total_gender, gender_accuracy_avg, gender_precision_avg, gender_recall_avg, gender_f1_avg],
        ['Overall System',"Total/Average", total_images, avg_overall_accuracy, avg_overall_precision, avg_overall_recall, avg_overall_f1],
        
    ]

    # Create table
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    ax.table(cellText=data, loc='center')

    # Display table
    plt.show()

    print("Done!")
