from fastapi import FastAPI
import pandas as pd
import mlflow.sklearn
import numpy as np
import os
import sys
# Add the src directory to the path for module imports
sys.path.append(os.path.abspath("../src"))
from src.api.pydantic_models import PredictionRequest, PredictionResponse

from src.data_processing import (
    extract_date_features,
    build_aggregate_features,
    clean_data
)

app = FastAPI()

# Load model from MLflow registry
model_uri = "models:/CustomerSegmentationModel/Production"
model = mlflow.sklearn.load_model(model_uri)


@app.get("/")
def read_root():
    return {"message": "Customer Segmentation API is running 🚀"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # Convert request to dataframe
    input_data = pd.DataFrame([request.dict()])

    # Apply feature engineering
    input_data = extract_date_features(input_data)
    input_data = build_aggregate_features(input_data)
    input_data = clean_data(input_data)

    # Prediction
    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data).max()

    return PredictionResponse(cluster=int(pred), probability=float(proba))
