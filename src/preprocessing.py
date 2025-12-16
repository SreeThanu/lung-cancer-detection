"""
Preprocessing utilities for LUNA16 CT scans
Includes:
- CTPreprocessor class for normalization & ROI extraction
- Augmentation3D class for training-time augmentations
- Factory functions for easy instantiation
"""

import os
import numpy as np
import SimpleITK as sitk
import torch
from typing import Tuple, Dict
from monai.transforms import (
    RandFlipd, RandRotate90d, RandAffined, RandGaussianNoised, Compose
)


# ---------------------------------------------
#  Low-level helper functions
# ---------------------------------------------
def load_ct_scan(seriesuid: str, data_dir: str):
    """Load CT scan from .mhd file."""
    mhd_path = os.path.join(data_dir, f"{seriesuid}.mhd")
    itk_image = sitk.ReadImage(mhd_path)
    image = sitk.GetArrayFromImage(itk_image)
    origin = np.array(itk_image.GetOrigin())
    spacing = np.array(itk_image.GetSpacing())
    return image, origin, spacing


def world_to_voxel(world_coords: np.ndarray, origin: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """Convert world coordinates to voxel coordinates."""
    voxel_coords = np.abs((world_coords - origin) / spacing)
    return voxel_coords.astype(int)


def normalize_hu(image: np.ndarray, min_hu: int = -1000, max_hu: int = 400) -> np.ndarray:
    """Clip and normalize HU values."""
    image = np.clip(image, min_hu, max_hu)
    image = (image - min_hu) / (max_hu - min_hu)
    return image


def extract_roi(image: np.ndarray, center: np.ndarray, roi_size: Tuple[int, int, int]) -> np.ndarray:
    """Extract 3D ROI around the nodule."""
    z, y, x = center
    d, h, w = roi_size
    z_start, z_end = max(0, z - d//2), min(image.shape[0], z + d//2)
    y_start, y_end = max(0, y - h//2), min(image.shape[1], y + h//2)
    x_start, x_end = max(0, x - w//2), min(image.shape[2], x + w//2)

    roi = image[z_start:z_end, y_start:y_end, x_start:x_end]
    
    if roi.shape != roi_size:
        padded = np.zeros(roi_size, dtype=np.float32)
        padded[:roi.shape[0], :roi.shape[1], :roi.shape[2]] = roi
        roi = padded
    
    return roi


# ---------------------------------------------
#  CT Preprocessor Class
# ---------------------------------------------
class CTPreprocessor:
    """
    Full preprocessing pipeline:
    - Load CT scan
    - HU normalization
    - ROI extraction
    - Save as .npy
    """
    def __init__(self, data_dir: str, output_dir: str, roi_size=(64, 64, 64)):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.roi_size = roi_size
        os.makedirs(self.output_dir, exist_ok=True)

    def process_case(self, seriesuid: str, coord: np.ndarray):
        """Load, normalize, extract ROI, and save."""
        image, origin, spacing = load_ct_scan(seriesuid, self.data_dir)
        image = normalize_hu(image)
        voxel_coords = world_to_voxel(coord, origin, spacing)
        roi = extract_roi(image, voxel_coords, self.roi_size)
        np.save(os.path.join(self.output_dir, f"{seriesuid}.npy"), roi)
        return roi


# ---------------------------------------------
#  Data Augmentation Class (3D)
# ---------------------------------------------
class Augmentation3D:
    """
    3D augmentation pipeline using MONAI transforms.
    Applies during training to ROI patches.
    """
    def __init__(self):
        self.transforms = Compose([
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            RandRotate90d(keys=["image"], prob=0.5, max_k=3),
            RandAffined(keys=["image"], prob=0.5, rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1)),
            RandGaussianNoised(keys=["image"], prob=0.2)
        ])
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.transforms(sample)


# ---------------------------------------------
#  Factory Functions
# ---------------------------------------------
def create_preprocessor(config: Dict):
    """Factory function to create CTPreprocessor from config."""
    data_dir = config['paths']['raw_data']
    output_dir = config['paths']['roi_data']
    roi_size = tuple(config['preprocessing']['roi_size'])
    return CTPreprocessor(data_dir, output_dir, roi_size)


def create_augmentation():
    """Factory function to create 3D augmentation pipeline."""
    return Augmentation3D()