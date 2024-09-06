import subprocess

# Define the Kaggle dataset URL
kaggle_url = "harrywang/wine-dataset-for-clustering"

# Define the filename you want to download
file_name = "wine-clustering.csv"

# Use Kaggle API to download the file
subprocess.run(['kaggle', 'datasets', 'download', '-d', kaggle_url, '-f', file_name])

import pandas as pd

# Load the dataset
data = pd.read_csv('wine-clustering.csv')

# Display basic information about the dataset
data.info()
data.describe()

from sklearn.preprocessing import StandardScaler

# Standardize the features
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
data_scaled.info()
data_scaled.describe()
data_scaled.head()