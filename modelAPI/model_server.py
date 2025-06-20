import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app, origins=["*"])  # Allow all origins for local dev

# Load model (sentiment-analysis as placeholder, replace with your custom model)
model = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    try:
        text = data['text']
        prediction = model(text)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, host="0.0.0.0", port=port)
