from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware  # ← ADD THIS LINE


app = FastAPI()  # ← Step 1: Create app

# ═══════════════════════════════════════════
# ADD CORS MIDDLEWARE HERE (Step 2: Right after app creation)
# ═══════════════════════════════════════════
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════
# NOW your routes (Step 3)
# ═══════════════════════════════════════════
@app.get("/sources")
def get_sources():
    return {"valid_sources": [...]}

@app.post("/predict")
def predict(data: dict):
    # Your code
    pass

# Load saved components
model = joblib.load("water_model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")
label_encoder = joblib.load("label_encoder.pkl")  # <-- add this if you saved it

app = FastAPI(
    title="Smart Water Potability Prediction API",
    description="Predict water potability and recommend filtration solution",
    version="1.0"
)

class WaterInput(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float
    source: str

@app.get("/")
def home():
    return {"message": "Smart Water Potability Prediction API is running"}

@app.post("/predict")
def predict(data: WaterInput):
    try:
        # Encode source string into numeric value
        source_encoded = label_encoder.transform([data.source])[0]

        # Build DataFrame
        input_df = pd.DataFrame([{
            "ph": data.ph,
            "Hardness": data.Hardness,
            "Solids": data.Solids,
            "Chloramines": data.Chloramines,
            "Sulfate": data.Sulfate,
            "Conductivity": data.Conductivity,
            "Organic_carbon": data.Organic_carbon,
            "Trihalomethanes": data.Trihalomethanes,
            "Turbidity": data.Turbidity,
            "source_encoded": source_encoded
        }])

        # Apply preprocessing
        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)

        # Get probabilities
        prob = model.predict_proba(input_scaled)[0]  # [prob_not_safe, prob_safe]

        # Apply threshold (default 0.5, but you can adjust)
        prediction = 1 if prob[1] > 0.65 else 0 # strict threshold

        # Interpret
        if prediction == 1:
            result = "Safe"
            filtration = "No filtration needed"
        else:
            result = "Not Safe"
            filtration = "Use RO + UV filtration"

        return {
            "prediction": int(prediction),
            "result": result,
            "filtration": filtration,
            "probabilities": {
                "not_safe": float(prob[0]),
                "safe": float(prob[1])
            }
        }

    except Exception as e:
        return {"error": str(e)}
    
@app.get("/sources")
def get_sources():
    # Return all valid source labels from the encoder
    return {"valid_sources": list(label_encoder.classes_)}


