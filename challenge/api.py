from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from typing import List
from challenge.model import DelayModel

app = FastAPI()

# Load and preprocess the training data, then train the model upon application start-up.
model = DelayModel()
data = pd.read_csv("data/data.csv")
features, target = model.preprocess(data, target_column="delay")
model.fit(features=features, target=target)

# Valid airlines and other constant data (loaded once at start-up).
valid_opera = data['OPERA'].unique().tolist()
valid_tipovuelo = ["I", "N"]
valid_mes = list(range(1, 13))

# Pydantic BaseModels for expected input.
class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

class FlightData(BaseModel):
    flights: List[Flight]

@app.get("/health", status_code=200)
async def get_health() -> dict:
    """
    Endpoint to check the health of the API.
    """
    return {"status": "OK"}

@app.post("/predict", status_code=200)
async def post_predict(flight_data: FlightData) -> dict:
    """
    Endpoint to make predictions based on flight data.
    """
    # List to store valid flights.
    valid_flights = []

    # Validate each flight in the request.
    for flight in flight_data.flights:
        if (flight.OPERA not in valid_opera or 
            flight.TIPOVUELO not in valid_tipovuelo or 
            flight.MES not in valid_mes):          
            raise HTTPException(status_code=400, detail="Invalid input data")
        
        # Add valid flight to the list.
        valid_flights.append(flight.dict())

    # Convert the list of valid dictionaries to a DataFrame.
    data = pd.DataFrame(valid_flights)

    # Preprocess the input.
    features = model.preprocess(data)

    # Make the prediction.
    prediction = model.predict(features)

    return {"predict": prediction}
