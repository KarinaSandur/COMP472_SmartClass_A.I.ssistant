import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cnn_model 
import os

# Initialize model
model = cnn_model.MainModel()

# Load the model
model.load_state_dict(torch.load("best_model_MainModel.pth"))

# Set the model to evaluation mode
model.eval()

# Get image path
image_path = input("Enter path to single image or path to directory containing images: ")

# ############################################################
# Prediction for single image
# ############################################################

if image_path.endswith(".jpg"):
    image = Image.open(image_path)

    # Transform image to correct size
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor()
    ])

    # Apply transformation to image
    transformed_image = transform(image).unsqueeze(0)

    # Get class prediction from model
    with torch.no_grad():
        output = model(transformed_image)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities)

    # Print class prediction
    if (predicted_class.item() == 0):
        print(f"Predicted class: angry")
    if (predicted_class.item() == 1):
        print(f"Predicted class: focused")
    if (predicted_class.item() == 2):
        print(f"Predicted class: happy")
    if (predicted_class.item() == 3):
        print(f"Predicted class: neutral")

# ############################################################
# Prediction for directory of images
# ############################################################

else:
    for file in os.listdir(image_path):
        if file.endswith(".jpg"):
            image_file = Image.open(os.path.join(image_path, file))

            # Transform image to correct size
            transform = transforms.Compose([
                transforms.Resize((150, 150)),
                transforms.ToTensor()
            ])

            # Apply transformation to image
            transformed_image = transform(image_file).unsqueeze(0)

            # Get class prediction from model
            with torch.no_grad():
                output = model(transformed_image)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities)

            # Print class prediction
            if predicted_class.item() == 0:
                print(f"Predicted class: angry")
            elif predicted_class.item() == 1:
                print(f"Predicted class: focused")
            elif predicted_class.item() == 2:
                print(f"Predicted class: happy")
            elif predicted_class.item() == 3:
                print(f"Predicted class: neutral")
            