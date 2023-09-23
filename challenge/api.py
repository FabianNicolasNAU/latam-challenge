from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from typing import List
from challenge.model import DelayModel

app = FastAPI()

# Instanciar el modelo y entrenarlo al iniciar la aplicación
model = DelayModel()
data = pd.read_csv("data/data.csv")
features, target = model.preprocess(data, target_column="delay")
model.fit(features=features, target=target)

# Aerolíneas válidas y otros datos constantes (sólo cargar una vez al inicio)
valid_opera = data['OPERA'].unique().tolist()
valid_tipovuelo = ["I", "N"]
valid_mes = list(range(1, 13))

# Pydantic BaseModels para la entrada esperada
class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

class FlightData(BaseModel):
    flights: List[Flight]

@app.get("/health", status_code=200)
async def get_health() -> dict:
    """Endpoint para verificar la salud de la API."""
    return {"status": "OK"}

@app.post("/predict", status_code=200)
async def post_predict(flight_data: FlightData) -> dict:
    """
    Endpoint para realizar predicciones basadas en datos de vuelo.
    """
    # Lista para almacenar vuelos válidos
    valid_flights = []

    # Validar cada vuelo en la solicitud
    for flight in flight_data.flights:
        if (flight.OPERA not in valid_opera or 
            flight.TIPOVUELO not in valid_tipovuelo or 
            flight.MES not in valid_mes):
            
            raise HTTPException(status_code=400, detail="Invalid input data")
        
        # Agregar vuelo válido a la lista
        valid_flights.append(flight.dict())

    # Convertir la lista de diccionarios válidos a DataFrame
    data = pd.DataFrame(valid_flights)

    # Preprocesar el input
    features = model.preprocess(data)

    # Realizar la predicción
    prediction = model.predict(features)

    return {"predict": prediction}
