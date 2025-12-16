"""
Data Loader for LUNA16 Dataset
Handles loading preprocessed ROI patches and annotations
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from typing import Dict, Tuple

class LUNA16Dataset(Dataset):
    """
    PyTorch Dataset for LUNA16 lung nodule detection
    (uses preprocessed ROI patches)
    """
    def __init__(self, 
                 data_dir: str,
                 annotations_file: str,
                 roi_size: Tuple[int, int, int] = (64, 64, 64),
                 transform=None,
                 mode: str = 'train'):
        """
        Args:
            data_dir: Directory containing .npy ROI files
            annotations_file: CSV file with nodule annotations
            roi_size: Size of ROI patch to extract
            transform: Data augmentation transforms
            mode: 'train', 'val', or 'test'
        """
        self.data_dir = data_dir
        self.roi_size = roi_size
        self.transform = transform
        self.mode = mode

        # Load annotation data
        self.annotations = pd.read_csv(annotations_file)
        self.annotations.fillna(0, inplace=True)
        self.samples = self.annotations.to_dict(orient='records')

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # --- Load preprocessed ROI ---
        # Expected ROI filename pattern: seriesuid_x_y_z.npy
        seriesuid = sample['seriesuid']
        roi_filename = f"{seriesuid}.npy"
        roi_path = os.path.join(self.data_dir, roi_filename)

        if not os.path.exists(roi_path):
            raise FileNotFoundError(f"ROI file missing for {seriesuid}")

        roi = np.load(roi_path).astype(np.float32)

        # --- Normalize ---
        roi = np.clip(roi, -1000, 400)  # typical HU range for lung tissue
        roi = (roi - np.mean(roi)) / (np.std(roi) + 1e-6)

        roi_tensor = torch.from_numpy(roi).unsqueeze(0)  # Add channel dim

        if self.transform:
            roi_tensor = self.transform(roi_tensor)

        # --- Labels ---
        label = torch.tensor(sample.get('label', 1), dtype=torch.long)
        malignancy = torch.tensor(sample.get('malignancy', 0), dtype=torch.float)
        bbox = torch.tensor([
            self.roi_size[0] // 2,
            self.roi_size[1] // 2,
            self.roi_size[2] // 2,
            sample.get('diameter_mm', 0)
        ], dtype=torch.float)
        
        return {
            'image': roi_tensor,
            'label': label,
            'malignancy': malignancy,
            'bbox': bbox,
            'metadata': {
                'seriesuid': seriesuid,
                'coordX': sample.get('coordX', 0),
                'coordY': sample.get('coordY', 0),
                'coordZ': sample.get('coordZ', 0)
            }
        }

# ---------------------------------------------------------------------

def create_data_loaders(config: Dict, full_dataset: Dataset):
    """
    Creates train/validation split and DataLoaders
    """
    train_ratio = config['training'].get('train_split', 0.8)
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )

    return train_loader, val_loader
