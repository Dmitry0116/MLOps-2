import os
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from model_preparation import train_arima_model

def test_arima_model(model, test_data_path):
   
    test_data = pd.read_csv(test_data_path)


    print("Наименования столбцов:", test_data.columns)

    #индекс 1 - столбец, содержащий временной ряд
    test_series = test_data[test_data.columns[1]]  

    prognostication = model.predict(start=0, end=len(test_series)-1)

    mse = mean_squared_error(test_series, prognostication)
    mae = mean_absolute_error(test_series, prognostication)

    print(f"Среднеквадратичная ошибка (MSE): {mse}")
    print(f"Средняя абсолютная погрешность (MAE): {mae}")

def process_files_in_train_folder():

    current_directory = os.getcwd()

    train_folder_path = os.path.join(current_directory, 'train')

    train_files = [file for file in os.listdir(train_folder_path) if file.startswith('preprocessed_')]

    if not train_files:
        print("В папке "train" не найдено предварительно обработанных файлов.")
        return

    for train_file in train_files:
    
        train_data_file = os.path.join(train_folder_path, train_file)

        arima_model = train_arima_model(train_data_file)

        test_arima_model(arima_model, train_data_file.replace('train', 'test'))

def main():

    process_files_in_train_folder()

if __name__ == "__main__":
    main()