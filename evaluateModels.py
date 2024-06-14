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

# Initialize models
mainModel = cnn_model.MainModel()
var1Model = cnn_model.Variant1()
var2Model = cnn_model.Variant2()

# Load the models
mainModel.load_state_dict(torch.load("best_model_MainModel.pth"))
var1Model.load_state_dict(torch.load("best_model_Variant1.pth"))
var2Model.load_state_dict(torch.load("best_model_Variant2.pth"))

# Set the models to evaluation mode
# mainModel.eval()
# var1Model.eval()
# var2Model.eval()

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

# evaluate the models
models = {
    'Main Model': mainModel,
    'Variant 1': var1Model,
    'Variant 2': var2Model
}

# Evaluate them
for name, model in models.items():
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in cnn_model.test_loader:
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


# Generate confusion matrices for each and table summarizing metrics