# nlp_model.py

import requests

MODEL_API_URL = "http://localhost:5001/predict"

def get_model_prediction(text):
    try:
        response = requests.post(MODEL_API_URL, json={"text": text})
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": "Model API error", "status_code": response.status_code}
    except Exception as e:
        return {"error": str(e)}
