from PIL import Image
import os

# Function that iterates through files in folder and converts all PNG files to JPEG
def png_to_jpeg_converter(folder):
    for file in os.listdir(folder):
        if file.endswith(".png"):
            png_image = Image.open(os.path.join(folder, file))
            new_file = os.path.splitext(file)[0] + ".jpg"
            rgb_image = png_image.convert('RGB') # Converts to RGB format which is required for JPEG

            # Saves new image with .jpg extension
            rgb_image.save(os.path.join(folder, new_file))

            # Deletes PNG file that is meant to be replaced
            os.remove(os.path.join(folder, file))

            # Prints success statement
            print(f"Replaced {file} with {new_file}")

    print("Finished converting PNG images to JPEG!")


if __name__ == "__main__":
    input_folder = input("Enter the folder path: ")
    png_to_jpeg_converter(input_folder)