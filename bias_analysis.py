import os
import zipfile
import pandas as pd

# Function to extract the zip file
def extract_zip(zip_file_path, extract_to):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Function to load datasets
def load_datasets(extracted_folder):
    gender_data_path = os.path.join(extracted_folder, 'Clean_Data_Gender.csv')
    age_data_path = os.path.join(extracted_folder, 'Clean_Data_Age.csv')

    if not os.path.exists(gender_data_path):
        print(f"Gender data file not found at: {gender_data_path}")
        return None, None

    if not os.path.exists(age_data_path):
        print(f"Age data file not found at: {age_data_path}")
        return None, None

    data_gender = pd.read_csv(gender_data_path)
    data_age = pd.read_csv(age_data_path)

    return data_gender, data_age
