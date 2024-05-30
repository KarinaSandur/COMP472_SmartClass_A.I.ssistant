import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import zipfile
import tempfile
import shutil

# unzipping function, unzip file to the specified path
def unzip_data(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

# get image paths from a folder
def get_image_paths(folder_path):
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('jpg')):
                img_path = os.path.join(root, file)
                image_paths.append(img_path)
    return image_paths

# display sample images with histograms, 5 x 3 with a total of 15 images per class, randomly generated
def display_sample_images_with_histograms(image_paths, class_name, num_images=15):
    # select a random sample of img paths and create a figure with a grid of subplots
    sample_image_paths = random.sample(image_paths, min(num_images, len(image_paths)))
    fig, axes = plt.subplots(5, 6, figsize=(20, 15))

    # iterate over the sample image paths, open the img and convert it to RGB array
    for i, img_path in enumerate(sample_image_paths):
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)

        # calculate the row and column for the current image subplot
        row, col = divmod(i, 3)
        # display the image on the left
        axes[row, col * 2].imshow(img)
        axes[row, col * 2].axis('off')

        # display histograms on the right
        ax_hist = axes[row, col * 2 + 1]
        ax_hist.hist(img_array[:, :, 0].flatten(), bins=256, alpha=0.5, label='Red', color='red')
        ax_hist.hist(img_array[:, :, 1].flatten(), bins=256, alpha=0.5, label='Green', color='green')
        ax_hist.hist(img_array[:, :, 2].flatten(), bins=256, alpha=0.5, label='Blue', color='blue')
        ax_hist.legend()

    plt.suptitle(f'Sample Images with Histograms from class {class_name.capitalize()}')
    plt.show()

if __name__ == "__main__":
    data_dir = input("Enter the directory path where your zip files are located: ")
    folders = ['angry', 'focused', 'neutral', 'happy']

    # create a temporary directory for working with the data
    with tempfile.TemporaryDirectory() as temp_dir:
        # copy zip files to the temporary directory
        for folder in folders:
            original_zip_path = os.path.join(data_dir, f"{folder}.zip")
            temp_zip_path = os.path.join(temp_dir, f"{folder}.zip")
            if os.path.exists(original_zip_path):
                shutil.copy2(original_zip_path, temp_zip_path)
            else:
                print(f"File not found: {original_zip_path}")
                continue

        # unzip the data in the temporary directory
        for folder in folders:
            unzip_data(os.path.join(temp_dir, f"{folder}.zip"), os.path.join(temp_dir, folder))

        # get image paths and display sample images with histograms
        for folder in folders:
            folder_path = os.path.join(temp_dir, folder)
            image_paths = get_image_paths(folder_path)
            display_sample_images_with_histograms(image_paths, folder)