import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import zipfile

# unzipping function, unzip file to the specified path
def unzip_data(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

# load the images from a folder and convert them to numpy arrays
def load_images(folder_path):
    images = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('jpg')):
                img_path = os.path.join(root, file)
                image = Image.open(img_path).convert('RGB')
                images.append(np.array(image))
    return images

# calculate pixel intensity distributions
def calculate_pixel_intensity(images):
    all_pixels = [img.flatten() for img in images]
    return np.array(all_pixels)

if __name__ == "__main__":
    data_dir = input("Enter the directory path where your zip files are located: ")
    folders = ['angry', 'focused', 'neutral', 'happy']

    # unzip the data
    for folder in folders:
        unzip_data(os.path.join(data_dir, f"{folder}.zip"), os.path.join(data_dir, folder))

    pixel_intensity = {}
    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        images = load_images(folder_path)
        pixel_intensity[folder] = calculate_pixel_intensity(images)

    # plot the pixel intensity distributions for each class
    for folder in folders:
        pixels = pixel_intensity[folder]
        plt.figure(figsize=(10, 6))
        plt.hist(pixels[:, 0], bins=256, alpha=0.5, label='Red', color='red')
        plt.hist(pixels[:, 1], bins=256, alpha=0.5, label='Green', color='green')
        plt.hist(pixels[:, 2], bins=256, alpha=0.5, label='Blue', color='blue')
        plt.title(f'Pixel Intensity Distribution for class {folder.capitalize()}')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()
