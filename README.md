Here’s a README file tailored for your K-mean clustering project based on the provided `main.py` and `run_model.py` files.

```markdown
# K-Mean Clustering

This project demonstrates the use of the K-Means clustering algorithm on the Wine dataset. It includes model training, evaluation, and deployment using FastAPI.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [How to Run](#how-to-run)
- [Model Training](#model-training)
- [API Usage](#api-usage)

## Introduction

The project performs the following tasks:
- Downloads the Wine dataset for clustering from Kaggle.
- Preprocesses the data by standardizing the features.
- Applies the K-Means clustering algorithm to the standardized data.
- Saves the trained model for later use.
- Serves the model through a FastAPI application to predict clusters for new data.

## Requirements

Ensure you have the following dependencies installed:

```bash
fastapi
uvicorn
joblib
pandas
scikit-learn
seaborn
matplotlib
kaggle
pyngrok
nest_asyncio
```

Install the required packages:

```bash
pip install fastapi uvicorn joblib pandas scikit-learn seaborn matplotlib kaggle pyngrok nest_asyncio
```

## Dataset

The dataset used in this project is the Wine dataset for clustering, which you can download from Kaggle. Make sure to set up your Kaggle API credentials correctly to access the dataset.

## How to Run

1. **Download the dataset**:
   - The dataset will be automatically downloaded when you run `main.py`.

2. **Train the model**:
   - Run `main.py` to perform data preprocessing and model training.

   ```bash
   python main.py
   ```

   The trained K-Means model will be saved as `kmeans_wine_model.pkl`.

3. **Start the FastAPI server**:
   - Run `run_model.py` to start the FastAPI server.

   ```bash
   python run_model.py
   ```

   The server will run and provide a public URL for accessing the API.

4. **Test the API**:
   You can use tools like Postman or `curl` to send POST requests to the API.

   Example `curl` request:

   ```bash
   curl -X POST "http://<public_url>/predict" -H "Content-Type: application/json" -d '{
     "Alcohol": 13.0,
     "Malic_Acid": 2.0,
     "Ash": 2.5,
     "Ash_Alcanity": 19.0,
     "Magnesium": 100.0,
     "Total_Phenols": 2.5,
     "Flavanoids": 1.0,
     "Nonflavanoid_Phenols": 0.2,
     "Proanthocyanins": 1.5,
     "Color_Intensity": 5.0,
     "Hue": 0.5,
     "OD280": 3.0,
     "Proline": 2.0
   }'
   ```

   The response will return the predicted cluster for the given data.

## Model Training

The `main.py` script performs the following:

1. Downloads the Wine dataset from Kaggle.
2. Loads the dataset and displays basic information.
3. Standardizes the features using `StandardScaler`.
4. Applies the K-Means clustering algorithm with 3 clusters.
5. Saves the trained model to a file (`kmeans_wine_model.pkl`).

## API Usage

The `run_model.py` script uses FastAPI to serve predictions based on the trained model. You can send data in the following format:

```json
{
  "Alcohol": 13.0,
  "Malic_Acid": 2.0,
  "Ash": 2.5,
  "Ash_Alcanity": 19.0,
  "Magnesium": 100.0,
  "Total_Phenols": 2.5,
  "Flavanoids": 1.0,
  "Nonflavanoid_Phenols": 0.2,
  "Proanthocyanins": 1.5,
  "Color_Intensity": 5.0,
  "Hue": 0.5,
  "OD280": 3.0,
  "Proline": 2.0
}
```

The response will provide the predicted cluster for the given data.
```

Feel free to adjust any sections or details as needed!