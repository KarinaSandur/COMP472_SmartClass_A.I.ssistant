import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cnn_model 
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

# evaluate the models
models = {
    'Main Model': mainModel,
    'Variant 1': var1Model,
    'Variant 2': var2Model
}

# Evaluate them



# Generate confusion matrices for each and table summarizing metrics