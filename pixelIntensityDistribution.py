import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import zipfile
import io

# load images from a zip file and convert them to numpy arrays
def load_images_from_zip(zip_path):
    images = []
    # open zip file in read mode and iterate over each file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if file.endswith(('jpg')):
                # open image file in the zip without extracting it
                with zip_ref.open(file) as img_file:
                    # read image file as bytes and open
                    image = Image.open(io.BytesIO(img_file.read())).convert('RGB')
                    images.append(np.array(image))
    return images

# calculate pixel intensity distributions
def calculate_pixel_intensity(images):
    all_pixels = [img.flatten() for img in images]
    return np.array(all_pixels)

if __name__ == "__main__":
    data_dir = input("Enter the directory path where your zip files are located: ")
    folders = ['angry', 'focused', 'neutral', 'happy']

    pixel_intensity = {}
    for folder in folders:
        # path to the zip file, load images from the zip file and calculate pixel intensity
        zip_path = os.path.join(data_dir, f"{folder}.zip")
        images = load_images_from_zip(zip_path)
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