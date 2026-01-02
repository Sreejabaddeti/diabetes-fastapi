import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 1. Initialize the FastAPI app
app = FastAPI(
    title="Diabetes Prediction API",
    description="A machine learning API to predict diabetes risk using patient health data.",
    version="1.0.0"
)

# 2. Define the Input Data Schema
# These names must match the columns used during model training
class PatientData(BaseModel):
    gender: int               # 0: Female, 1: Male, 2: Other
    age: float
    hypertension: int         # 0: No, 1: Yes
    heart_disease: int        # 0: No, 1: Yes
    smoking_history: int      # Encoded value (0-5)
    bmi: float
    HbA1c_level: float
    blood_glucose_level: int

# 3. Load the Model (using joblib for better compatibility)
MODEL_PATH = "diabetes_model.pkl"

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        model = None
        print(f"Error loading model: {e}")
else:
    model = None
    print(f"Warning: {MODEL_PATH} not found in the current directory.")

# --- Endpoints ---

@app.get("/")
def home():
    """Root endpoint to check API status."""
    return {
        "status": "Online",
        "message": "Welcome to the Diabetes Prediction API!",
        "documentation": "/docs"
    }

@app.post("/predict")
def predict(data: PatientData):
    """Prediction endpoint that takes patient metrics and returns the result."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model file is missing or corrupted on the server.")
    
    try:
        # Convert Pydantic model to a 2D numpy array for the ML model
        input_features = np.array([[
            data.gender, 
            data.age, 
            data.hypertension, 
            data.heart_disease,
            data.smoking_history, 
            data.bmi, 
            data.HbA1c_level, 
            data.blood_glucose_level
        ]])
        
        # Make prediction
        prediction = model.predict(input_features)
        
        # Get probability if the model supports it
        result = "Positive" if int(prediction[0]) == 1 else "Negative"
        
        return {
            "prediction": result,
            "status_code": 200
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Note: In production (Render/Railway), the start command handles the server execution.