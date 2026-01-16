import torch
import torch.nn as nn
from torchvision import models

def get_device():
    """
    select device: MPS (Apple silicon) > CUDA > CPU
    """

    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.backends.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    
def load_model(weights_path: str, num_classes: int = 11):
    """
    Load finetuned ResNet18 model for inference only
    """
    device = get_device()

    model = models.resnet18(pretrained=False)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model, device
