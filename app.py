import sys
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import requests

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import clean_text, enrich_prediction
from opencv_python.pdf_img import convert_pdf_to_images
from processing.ocr import extract_text_from_image

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])  # Frontend origin

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

MODEL_API_URL = "http://localhost:5001/predict"

@app.route('/analyze', methods=['POST'])
def analyze():
    extracted_text = ""

    # Handle PDF
    if request.content_type and "multipart/form-data" in request.content_type:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            images = convert_pdf_to_images(file_path)
            for image in images:
                extracted_text += extract_text_from_image(image) + "\n"
        except Exception as e:
            return jsonify({"error": "Failed to extract text from PDF", "details": str(e)}), 500
        finally:
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Warning: could not remove uploaded file: {e}")

    # Handle text
    elif request.is_json:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text input provided"}), 400
        extracted_text = data['text']
    else:
        return jsonify({"error": "Unsupported Content-Type. Use JSON or multipart/form-data"}), 415

    if not extracted_text.strip():
        return jsonify({"error": "Empty input text"}), 400

    try:
        cleaned_text = clean_text(extracted_text)

        # 🔁 Call AI model microservice
        model_response = requests.post(MODEL_API_URL, json={"text": cleaned_text})
        if model_response.status_code != 200:
            return jsonify({"error": "Model service failed", "details": model_response.text}), 500
        prediction = model_response.json()

        # Enrich with precaution info
        precautions = enrich_prediction(prediction, input_text=cleaned_text)

        return jsonify({
            "extracted_text": cleaned_text,
            "precautions": precautions
        }), 200

    except Exception as e:
        print("Error during prediction/enrichment:", str(e))
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
