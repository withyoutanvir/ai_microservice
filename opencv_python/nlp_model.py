# nlp_model.py

import requests
import os

# Use environment variable, fallback to local dev if not set
MODEL_API_URL = os.getenv("AI_SERVICE_URL", "http://localhost:5001") + "/analyze"

def get_model_prediction(text):
    try:
        response = requests.post(MODEL_API_URL, json={"text": text})
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": "Model API error", "status_code": response.status_code, "message": response.text}
    except Exception as e:
        return {"error": str(e)}
