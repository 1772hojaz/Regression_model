import pickle as pk
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load the trained regression model
with open('model.pkl', 'rb') as file:
    model = pk.load(file)

# Initialize FastAPI app
app = FastAPI()


class PredictionRequest(BaseModel):
    Item_MRP: float
    Outlet_Size: int  # Integer for Outlet Size
    Outlet_Location_Type: int  # Integer for Outlet Location Type
    Outlet_Type: int  # Integer for Outlet Type
    New_Item_Type: int  # Integer for New Item Type
    Outlet_Years: int  # Integer for Outlet Years

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
        # Convert input features into a list
        input_features = [
            request.Item_MRP,
            request.Outlet_Size,
            request.Outlet_Location_Type,
            request.Outlet_Type,
            request.New_Item_Type,
            request.Outlet_Years
        ]
        
        # Make prediction
        prediction = model.predict([input_features])
        
        
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")
