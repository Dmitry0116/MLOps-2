import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(root, f'preprocessed_{file}')

                preprocess_data(input_file_path, output_file_path)

def preprocess_data(input_file, output_file):

    data = pd.read_csv(input_file)

    numerical_features = data.select_dtypes(include=['float64', 'int64']).columns

    scaler = StandardScaler()

    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    data.to_csv(output_file, index=False)

def main():

    current_directory = os.getcwd()

    train_folder_path = os.path.join(current_directory, 'train')
    test_folder_path = os.path.join(current_directory, 'test')
    
    preprocess_folder(train_folder_path)
    preprocess_folder(test_folder_path)

if __name__ == "__main__":
    main()