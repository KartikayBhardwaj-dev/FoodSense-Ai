import json 
import torch
from src.data.preprocess import preprocess_image

def load_class_mappings(mapping_path):
    with open(mapping_path, "r") as f:
        class_mapping = json.load(f)
    return class_mapping

def predict_images(image_path, model, device, class_mapping_path="/Users/kartikaybhardwaj/FoodSense-Ai/artifacts/class_mapping.json"):
    model.eval()
    image = preprocess_image(image_path, device)
    
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()

    class_mapping = load_class_mappings(class_mapping_path)
    prediction_class = class_mapping[str(pred_idx)]
    confidence = probs[0][pred_idx].item()

    return prediction_class, confidence
