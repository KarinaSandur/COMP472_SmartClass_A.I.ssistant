from PIL import Image
import os

# Function that iterates through files in folder and resizes them
def resizeImages(folder, width, height):
    for file in os.listdir(folder):
        
        # Resizes image
        image = Image.open(os.path.join(folder, file))
        new_image = image.resize((width, height))
        new_image.save(os.path.join(folder, file))

        # Prints Success Statement
        print(f"Resized {file}")

    print("Finished resizing!")


if __name__ == "__main__":
    input_folder = input("Enter the folder path: ")
    resizeImages(input_folder, 150, 150)