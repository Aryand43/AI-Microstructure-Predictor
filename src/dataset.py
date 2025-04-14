import os
import torch
import pandas as pd
import cv2
class MicrostructureDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_csv):
        self.image_dir = image_dir
        self.labels_df = pd.read_csv(label_csv)[['d=p/h', 'f=h/w', 'w', 'h', 'p']]
        self.labels_df = self.labels_df.dropna().reset_index(drop=True)
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])
        self.image_files = self.image_files[:len(self.labels_df)]
        assert len(self.image_files) == len(self.labels_df), "Mismatch between images and labels"
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))
        img = img.astype('float32') / 255.0
        img_tensor = torch.from_numpy(img).unsqueeze(0)
        label_row = self.labels_df.iloc[idx].values.astype('float32')
        label_tensor = torch.tensor(label_row)
        return img_tensor, label_tensor 

   
        