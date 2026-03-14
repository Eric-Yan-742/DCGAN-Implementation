import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

    
class ImageNet32Batch(Dataset):
    def __init__(self, batch_path):
        with open(batch_path, 'rb') as f:
            data = pickle.load(f)
        # shape: (N, 3072) -> (N, 3, 32, 32)
        self.images = data['data'].reshape(-1, 3, 32, 32)
        self.labels = data['labels']  # 1-indexed

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]  # (3, 32, 32), uint8
        img = torch.from_numpy(img).float() / 255.0
        img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
        label = self.labels[idx] - 1  # convert to 0-indexed
        return img, label