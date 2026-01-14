## FoodSense-Ai
FoodSense-Ai is an end to end food image classification system built using PyTorch and deployed using Flask on Azure

## Dataset
- Food-11 Dataset
- 11 food categories
- Folder-Based labelling

## Model
- Transfer Learning (ResNet18)
- ImageNet pretrained
- Mutliclass classification 

## Tech Stack
- PyTorch
- Flask 
- Python 3.10

## Setup
```bash
conda create -n foodsense python=3.10 -y
conda activate foodsense
pip install -r requirements.txt
