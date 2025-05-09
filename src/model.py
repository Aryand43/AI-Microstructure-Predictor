import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import resnet34
from PIL import Image
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset_loader import load_material_datasets

label_scaler = StandardScaler()

# === Dataset Class ===
class MicrostructureDataset(Dataset):
    def __init__(self, data_dict, transform=None, normalize_labels=False):
        self.data = list(data_dict.values())
        self.transform = transform
        self.label_keys = [
            "w", "h", "p",
            "Power", "Scanning Speed", "Powder Flow Rate",
            "Density", "Thermal Conductivity", "Specific Heat Capacity", "Thermal Expansion Coefficient"
        ]

        self.labels_raw = [[entry[k] for k in self.label_keys] for entry in self.data]
        if normalize_labels:
            self.labels = label_scaler.fit_transform(self.labels_raw)
        else:
            self.labels = self.labels_raw

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        image = Image.open(entry['image_path']).convert('L')
        if self.transform:
            image = self.transform(image)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label_tensor

# === Model ===
def get_resnet(model_name="resnet18", output_size=10, dropout_rate=0.3):
    if model_name == "resnet34":
        model = resnet34(pretrained=True)
    else:
        model = resnet18(pretrained=True)

    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.BatchNorm1d(num_features),
        nn.Dropout(dropout_rate),
        nn.Linear(num_features, output_size)
    )
    return model


# === Loss Function ===
def hybrid_loss(output, target):
    weights = torch.tensor([
        1, 1, 1,   
        0.5, 0.5, 0.5,   
        0.2, 0.2, 0.2, 0.2  
    ], dtype=torch.float32).to(output.device)

    mse = nn.MSELoss()(output * weights, target * weights)
    l1 = nn.L1Loss()(output * weights, target * weights)
    return 0.7 * mse + 0.3 * l1


# === Train Function ===
def train_model(model, dataloader, epochs=150, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    scaler = torch.cuda.amp.GradScaler()

    best_loss = float('inf')
    early_stop_count = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = hybrid_loss(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        scheduler.step(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            early_stop_count = 0
            torch.save(model.state_dict(), 'best_model.pt')
            joblib.dump(label_scaler, 'label_scaler.pkl')
        else:
            early_stop_count += 1

        if early_stop_count >= 20:
            print("Early stopping triggered.")
            break

# === Inference Function ===
def predict_from_image(image_path, model, label_keys, label_scaler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    image = Image.open(image_path).convert('L')
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(image).cpu().numpy().flatten()
        prediction = label_scaler.inverse_transform([prediction])[0]

    return dict(zip(label_keys, prediction))

# === Entry Point ===
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer', type=str, help='Run inference on image path')
    args = parser.parse_args()

    dict_co, dict_steel, dict_h13 = load_material_datasets()
    full_dict = {**dict_co, **dict_steel, **dict_h13}

    # Validation: Ensure all keys exist
    required_keys = [
        "w", "h", "p",
        "Power", "Scanning Speed", "Powder Flow Rate",
        "Density", "Thermal Conductivity", "Specific Heat Capacity", "Thermal Expansion Coefficient"
    ]
    for img_data in full_dict.values():
        for key in required_keys:
            if key not in img_data:
                raise ValueError(f"Missing key '{key}' in dataset entry: {img_data}")

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    transform_infer = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    if args.infer:
        sample_keys = list(full_dict.values())[0]
        label_keys = [
            "w", "h", "p",
            "Power", "Scanning Speed", "Powder Flow Rate",
            "Density", "Thermal Conductivity", "Specific Heat Capacity", "Thermal Expansion Coefficient"
        ]
        model = get_resnet(model_name="resnet34", output_size=10)
        model.load_state_dict(torch.load('best_model.pt', map_location=torch.device('cpu')))
        label_scaler = joblib.load("label_scaler.pkl")
        preds = predict_from_image(args.infer, model, label_keys, label_scaler)
        print("\nPrediction:")
        for k, v in preds.items():
            print(f"{k}: {v:.2f}")
    else:
        dataset = MicrostructureDataset(full_dict, transform=transform_train, normalize_labels=True)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        model = get_resnet18(output_size=10)
        train_model(model, dataloader, epochs=150)