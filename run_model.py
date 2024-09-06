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