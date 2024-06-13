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

# Get image path
image_path = input("Enter image path: ")
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
    print(f"Predicted class: happy")

