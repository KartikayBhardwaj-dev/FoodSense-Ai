from src.models.model import load_model
from src.data.preprocess import preprocess_image
from src.prediction.predict import predict_images

model, device = load_model("artifacts/finetuned_resnet18.pth")
label, confidence = predict_images("/Users/kartikaybhardwaj/FoodSense-Ai/Screenshot 2026-01-16 at 11.19.29 AM.png", model, device)

print("prediction: ", label)
print("Confidence: ", confidence)