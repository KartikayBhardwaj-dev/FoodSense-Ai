import os 
from flask import Flask, request, jsonify
import torch
from src.models.model import load_model
from src.prediction.predict import predict_images

app = Flask(__name__)

MODEL_PATH = "/Users/kartikaybhardwaj/FoodSense-Ai/artifacts/finetuned_resnet18.pth"
print("Loading Model...")
model, device = load_model(MODEL_PATH)
print(f"Model Loaded on device: {device}")

from flask import render_template

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# input validation
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    temp_path = "temp.jpg"
    file.save(temp_path)

    try:
        label, confidence = predict_images(temp_path, model, device)
        return jsonify({
            "prediction": label,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
if __name__ == "__main__":
    app.run(debug=True)