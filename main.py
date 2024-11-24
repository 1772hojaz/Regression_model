import pickle as pk
import numpy as np
import pandas as pd
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
    Item_MRP: str  # Item_MRP as a string, to be converted to float
    Outlet_Size: str
    Outlet_Location_Type: str
    Outlet_Type: str
    New_Item_Type: str  # Replaced Item_Type with New_Item_Type
    Outlet_Years: int  # Outlet_Years as an integer

# Initialize the label encoder
le = LabelEncoder()

# Dummy categories for LabelEncoder transformation
# Ensure these match your actual categories in the trained model
categories = {
    'Outlet_Size': ['Small', 'Medium', 'Large'],
    'Outlet_Location_Type': ['Tier1', 'Tier2'],
    'Outlet_Type': ['Supermarket', 'Grocery'],
    'New_Item_Type': ['Fresh', 'NonFresh']  # Replaced Item_Type with New_Item_Type
}

# Fit the label encoder for each categorical feature (these need to match the training data)
for col, values in categories.items():
    le.fit(values)

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
        # Convert Item_MRP from string to float
        Item_MRP = float(request.Item_MRP)

        # Apply Label Encoding for categorical variables
        Outlet_Size = le.transform([request.Outlet_Size])[0]
        Outlet_Location_Type = le.transform([request.Outlet_Location_Type])[0]
        Outlet_Type = le.transform([request.Outlet_Type])[0]
        New_Item_Type = le.transform([request.New_Item_Type])[0]
        
        # Get the Outlet_Years as an integer
        Outlet_Years = request.Outlet_Years

        # Prepare the features for One-Hot Encoding (assuming these are categorical variables for one-hot encoding)
        # Create a DataFrame with the input data for one-hot encoding
        input_data = pd.DataFrame({
            'Outlet_Size': [request.Outlet_Size],
            'Outlet_Location_Type': [request.Outlet_Location_Type],
            'Outlet_Type': [request.Outlet_Type],
            'New_Item_Type': [request.New_Item_Type]
        })

        # One-Hot Encoding (pd.get_dummies) for categorical variables
        input_data_encoded = pd.get_dummies(input_data, drop_first=True)

        # Now prepare the final input feature vector
        # Add the numeric fields (Item_MRP and Outlet_Years) to the one-hot encoded features
        input_features = np.concatenate([
            input_data_encoded.values.flatten(),  # Flatten the one-hot encoded columns
            [Item_MRP, Outlet_Years]  # Add the numeric features
        ]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_features)

        # Return the prediction as a response
        return {"prediction": prediction[0]}  # Assuming regression model outputs a single value

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

