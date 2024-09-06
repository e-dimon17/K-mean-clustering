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

from sklearn.cluster import KMeans

# Define the K-Means model
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)

# Fit the model to the data
kmeans.fit(data_scaled)

# Get the cluster labels
data['Cluster'] = kmeans.labels_
print(data)

import seaborn as sns
import matplotlib.pyplot as plt

# Pairplot to visualize clusters
sns.pairplot(data, hue='Cluster', diag_kind='kde')
plt.show()

import joblib

# Save the K-Means model
joblib.dump(kmeans, 'kmeans_wine_model.pkl')