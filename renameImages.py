import os

def renameImages(folder, name):
    for count, filename in enumerate(os.listdir(folder)):

        # New Image Name
        new_filename = f"{name}{count}.jpg" 

        # Saving Image with New Name
        os.rename(os.path.join(folder, filename), os.path.join(folder, new_filename))

    print("Finished renaming all images!")

if __name__ == "__main__":
    input_folder = input("Enter the folder path: ")
    input_name = input("Enter name or letter that will go before count: ")
    renameImages(input_folder, input_name)