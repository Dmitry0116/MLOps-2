import os
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

def train_arima_model(data_file, order=(1, 0, 1)):

    data = pd.read_csv(data_file)

    print("Наименования столбцов:", data.columns)
    
    # индекс 1 - столбец, содержащий временной ряд
    time_series = data[data.columns[1]]  

    # Создание и обучение модели 
    model = ARIMA(time_series, order=order)
    trained_model = model.fit()

    return trained_model

def process_files_in_train_folder():
    
    current_directory = os.getcwd()

    train_folder_path = os.path.join(current_directory, 'train')

    train_files = [file for file in os.listdir(train_folder_path) if file.startswith('preprocessed_')]

    if not train_files:
        print("В папке "train" не найдено предварительно обработанных файлов")
        return

    for train_file in train_files:
        
        train_data_file = os.path.join(train_folder_path, train_file)

        arima_model = train_arima_model(train_data_file)

def main():
    
    process_files_in_train_folder()

if __name__ == "__main__":
    main()