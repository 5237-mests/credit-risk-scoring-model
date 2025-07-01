from fastapi import FastAPI
from src.api.pydantic_models import PredictionRequest, PredictionResponse
import pandas as pd
import joblib
# import mlflow
# import mlflow.sklearn

from src.data_processing import (
    extract_date_features,
    build_aggregate_features,
    clean_data
)

# Use the mlruns folder inside Docker (mounted from local)
# mlflow.set_tracking_uri("file:/app/mlruns")

app = FastAPI(
    title="Credit Risk Scoring API",
    description="API for predicting customer risk clusters",
    version="1.0.0"
)

# ✅ Load the trained model from mlruns folder
# model = mlflow.sklearn.load_model("models:/credit-risk-randomforest-model/1")

# ✅ Load the trained model
model = joblib.load("./models/rfm_cluster_model.pkl")


@app.get("/")
def read_root():
    return {"message": "Credit Risk Scoring API is up and running 🚀"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # Convert request to dataframe
    input_data = pd.DataFrame([request.dict()])

    # ✅ Apply feature engineering steps (if applicable)
    input_data = extract_date_features(input_data)
    input_data = build_aggregate_features(input_data)
    input_data = clean_data(input_data)

    # ✅ Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data).max()

    return PredictionResponse(cluster=int(prediction), probability=float(probability))
