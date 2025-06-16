from pathlib import Path
from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS
import PyPDF2
import io

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])

# Load a sentiment analysis pipeline from Hugging Face
nlp_pipeline = pipeline("sentiment-analysis")

@app.route('/predict', methods=['POST'])
def predict():
    text = ""

    if 'file' in request.files:
        file = request.files['file']
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        if not text.strip():
            return jsonify({"error": "Empty PDF or no extractable text"}), 400
    else:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No input provided. Upload a PDF file or provide text."}), 400
        text = data['text']
        if not text.strip():
            return jsonify({"error": "Empty text input"}), 400

    try:
        result = nlp_pipeline(text[:512])  # truncate to 512 tokens
        # result is like: [{'label': 'POSITIVE', 'score': 0.998}]
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)
