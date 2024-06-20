import os
import numpy as np
import pandas as pd

def generate_time_series(length=365, anomalies=False, noise=False, seasonality=True, cycles=True):
    time = pd.date_range(start='2002-01-01', periods=length, freq='D')

    meat_price = np.random.normal(loc=10, scale=2, size=length)
    milk_price = np.random.normal(loc=5, scale=1, size=length)
    apple_price = np.random.normal(loc=2, scale=0.5, size=length)

    if seasonality:
        time_of_year = np.sin(2 * np.pi * np.arange(length) / 365)
        meat_price = meat_price + 2 * time_of_year
        milk_price = milk_price + 1 * time_of_year
        apple_price = apple_price + 0.5 * time_of_year

    if cycles:
        time_cycle = np.sin(2 * np.pi * np.arange(length) / 30)  # Пример цикла продолжительностью в 30 дней
        meat_price = meat_price + 1 * time_cycle
        milk_price = milk_price + 0.5 * time_cycle
        apple_price = apple_price + 0.2 * time_cycle

    if anomalies:
 
        anomaly_indices = np.random.choice(length, size=int(0.05 * length), replace=False)
        meat_price[anomaly_indices] = meat_price[anomaly_indices] + np.random.normal(loc=5, scale=2, size=len(anomaly_indices))
        milk_price[anomaly_indices] = milk_price[anomaly_indices] + np.random.normal(loc=2, scale=1, size=len(anomaly_indices))
        apple_price[anomaly_indices] = apple_price[anomaly_indices] + np.random.normal(loc=1, scale=0.5, size=len(anomaly_indices))

    if noise:

        meat_price = meat_price + np.random.normal(loc=0, scale=0.5, size=length)
        milk_price = milk_price + np.random.normal(loc=0, scale=0.2, size=length)
        apple_price = apple_price + np.random.normal(loc=0, scale=0.1, size=length)

    return pd.DataFrame({'Date': time, 'Meat_Price': meat_price, 'Milk_Price': milk_price, 'Apple_Price': apple_price})

def save_dataset(dataset, folder, split_ratio=0.7):
    if not os.path.exists(folder):
        os.makedirs(folder)

    split_index = int(len(dataset) * split_ratio)
    train_data = dataset.iloc[:split_index]
    test_data = dataset.iloc[split_index:]

    projects = ['Meat', 'Milk', 'Apple']

    train_folder = os.path.join(folder, 'train')
    test_folder = os.path.join(folder, 'test')

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)

    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    for project in projects:
        project_train_data = train_data[['Date', f'{project}_Price']]
        project_test_data = test_data[['Date', f'{project}_Price']]

        project_train_data.to_csv(os.path.join(train_folder, f'{project.lower()}_train_data.csv'), index=False)
        project_test_data.to_csv(os.path.join(test_folder, f'{project.lower()}_test_data.csv'), index=False)

def main():
    
    generated_data = generate_time_series(length=3650, anomalies=True, noise=True, seasonality=True, cycles=True)
    save_dataset(generated_data, '.', split_ratio=0.7)

if __name__ == "__main__":
    main()