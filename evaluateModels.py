from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cnn_model 
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import zipfile
import tempfile

# set random seed for consistent runs (if we want each run to be different, remove this part)
torch.manual_seed(42)

# Initialize models
mainModel = cnn_model.MainModel()
var1Model = cnn_model.Variant1()
var2Model = cnn_model.Variant2()

# Load the models
mainModel.load_state_dict(torch.load("best_model_MainModel.pth"))
var1Model.load_state_dict(torch.load("best_model_Variant1.pth"))
var2Model.load_state_dict(torch.load("best_model_Variant2.pth"))


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

if __name__ == "__main__":
    data_dir = input("Enter the directory path where your zip files are located: ")
    batch_size = 32

    results = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        cnn_model.unzip_files(data_dir, temp_dir)
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
            'Main Model': mainModel,
            'Variant 1': var1Model,
            'Variant 2': var2Model
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

        # Display table
        plt.show()

        # Print out message that figure display is complete
        print("Finished displaying all figures!")

        # determine the best model based on overall performance across all metrics
        # the sum of all metrics except the confusion matrix is calculated for each model,
        # the model with the highest sum of these metrics is considered the best model
        best_model_name = max(results, key=lambda x: sum(results[x][metric] for metric in results[x] if metric != 'confusion_matrix'))
        best_model = models[best_model_name]
        best_model_metrics = results[best_model_name]

        # Print out which is the best model
        print("The best model is: " + best_model_name)

        # save the best performing model out of the three (main model, variant 1, variant 2)
        torch.save(best_model.state_dict(), 'best_performing_model.pth')

        # Print out confirmation message
        print("Finished evaluating all models and saving the best one!")