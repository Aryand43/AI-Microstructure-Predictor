import torch
import cv2
import numpy as np
from model import MicrostructureCNN
import os
device = torch.device('cpu')
MODEL_PATH = 'microstructure_model.pth'
IMG_SIZE = 224
LABELS = ['d (p/h)', 'f (h/w)', 'w (width)', 'h (height)', 'p (depth)']

model = MicrostructureCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def predict_from_tif(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found or unreadable: {img_path}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        prediction = output.squeeze().tolist()
    return dict(zip(LABELS, prediction))

if __name__ == '__main__':
    image_dir = '../data/images'
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])

    if not image_files:
        raise ValueError("No .tif images found in ../data/images")

    path = os.path.join(image_dir, image_files[0]) 
    preds = predict_from_tif(path)
    for k, v in preds.items():
        print(f"{k}: {v:.4f}")