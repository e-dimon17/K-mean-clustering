from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
model = joblib.load('kmeans_wine_model.pkl')

# Define the input data model
class WineData(BaseModel):
    Alcohol: float
    Malic_Acid: float
    Ash: float
    Ash_Alcanity: float
    Magnesium: float
    Total_Phenols: float
    Flavanoids: float
    Nonflavanoid_Phenols: float
    Proanthocyanins: float
    Color_Intensity: float
    Hue: float
    OD280: float
    Proline: float

@app.post('/predict')
def predict_wine_cluster(data: WineData):
    # Convert input data to DataFrame
    input_data = pd.DataFrame([data.dict().values()], columns=data.dict().keys())

    # Reorder input data to match the features used during training
    training_features = ['Alcohol', 'Malic_Acid', 'Ash', 'Ash_Alcanity', 'Magnesium',
                         'Total_Phenols', 'Flavanoids', 'Nonflavanoid_Phenols',
                         'Proanthocyanins', 'Color_Intensity', 'Hue', 'OD280', 'Proline']

    input_data = input_data[training_features]

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Ensure that the scaled data retains the column names (feature names)
    input_data_scaled_df = pd.DataFrame(input_data_scaled, columns=training_features)

    # Make prediction using the DataFrame with valid feature names
    prediction = model.predict(input_data_scaled_df)
    print(f"Prediction: {int(prediction[0])}")
    return {'prediction': int(prediction[0])}

import uvicorn
import nest_asyncio
from pyngrok import ngrok

# Apply the nest_asyncio patch
nest_asyncio.apply()

public_url = ngrok.connect(9002, "http")
print('Public URL:', public_url)

uvicorn.run(app, host='0.0.0.0', port=9002)