import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cnn_model 

# Initialize model
model = cnn_model.Variant2()

# Load the model
model.load_state_dict(torch.load("best_performing_model.pth"))

# Set the model to evaluation mode
model.eval()

