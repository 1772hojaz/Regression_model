import pickle as pk
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder

# Load the trained regression model
with open('model.pkl', 'rb') as file:
    model = pk.load(file)

# Initialize FastAPI app
app = FastAPI()

# Define the input schema for prediction requests
class PredictionRequest(BaseModel):
    Item_MRP: float
    Outlet_Size: str
    Outlet_Location_Type: str
    Outlet_Type: str
    New_Item_Type: str
    Outlet_Years: int

# Root endpoint to provide API information
@app.get('/')
def root():
    return {
        "message": "Welcome to the Regression Model API!",
        "endpoints": {
            "predict": "POST /predict - Provide features to get a prediction"
        }
    }

# Prediction endpoint
@app.post('/predict')
def predict(request: PredictionRequest):
    try:
        # Initialize label encoders (refit them here)
        label_encoders = {
            'Outlet_Type': LabelEncoder(),
            'Outlet_Size': LabelEncoder(),
            'Outlet_Location_Type': LabelEncoder(),
            'New_Item_Type': LabelEncoder(),
        }

        # Fit the encoders using known categories (this is just an example, replace with your known categories)
        label_encoders['Outlet_Type'].fit(["Supermarket Type1", "Supermarket Type2", "Supermarket Type3"])
        label_encoders['Outlet_Size'].fit(["small", "medium", "high"])
        label_encoders['Outlet_Location_Type'].fit(["Tier1", "Tier2", "Tier3"])
        label_encoders['New_Item_Type'].fit(["Food", "Drinks", "Non-Consumable"])

        # Apply label encoding to categorical features
        encoded_features = [
            label_encoders['Outlet_Type'].transform([request.Outlet_Type])[0],
            label_encoders['Outlet_Size'].transform([request.Outlet_Size])[0],
            label_encoders['Outlet_Location_Type'].transform([request.Outlet_Location_Type])[0],
            label_encoders['New_Item_Type'].transform([request.New_Item_Type])[0],
            request.Item_MRP,
            request.Outlet_Years
        ]

        # Convert the encoded features to a NumPy array
        input_features = np.array(encoded_features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_features)

        # Return the prediction as a response
        return {"prediction": prediction[0]}  # Assuming regression model outputs a single value

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

