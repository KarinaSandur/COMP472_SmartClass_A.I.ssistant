import logging
import os
from shutil import unpack_archive
import tempfile
import zipfile
from pathlib import Path
import cnn_model 


def unzip_files(data_dir, temp_dir):
    folders = ['angry', 'focused', 'happy', 'neutral']
    for folder in folders:
        zip_path = os.path.join(data_dir, f"{folder}.zip")
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.join(temp_dir, folder))
        else:
            print(f"File not found: {zip_path}")


def count_files_in_folders(directory):
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        if os.path.isdir(folder_path):
            num_files = 0
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if zipfile.is_zipfile(file_path):
                        with zipfile.ZipFile(file_path, 'r') as zip_ref:
                            num_files += len(zip_ref.namelist())
                    else:
                        num_files += 1
            print(f"Folder '{folder}' contains {num_files} files.")


if __name__ == "__main__":
    data_dir = input("your_directory_path_here: ")

    count_files_in_folders(data_dir)
        
    with tempfile.TemporaryDirectory() as temp_dir:
        for folder in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder)
            if os.path.isdir(folder_path):
                unzip_files(folder_path, temp_dir)
                print(f"Unzipped files from {folder_path} to {temp_dir}")