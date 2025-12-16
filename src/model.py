"""
3D Swin Transformer Model for Lung Cancer Detection
Complete model architecture with detection and malignancy heads
"""

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
from typing import Tuple, Dict

class SwinTransformer3D(nn.Module):
    """
    3D Swin Transformer backbone for medical imaging
    Uses MONAI's SwinUNETR as base
    """
    
    def __init__(self,
                 img_size: Tuple[int, int, int] = (64, 64, 64),
                 in_channels: int = 1,
                 feature_size: int = 48,
                 depths: Tuple[int, ...] = (2, 2, 6, 2),
                 num_heads: Tuple[int, ...] = (3, 6, 12, 24),
                 dropout: float = 0.2):
        """
        Args:
            img_size: Input image size (D, H, W)
            in_channels: Number of input channels
            feature_size: Base feature dimension
            depths: Number of layers in each stage
            num_heads: Number of attention heads in each stage
            dropout: Dropout rate
        """
        super().__init__()
        
        self.img_size = img_size
        
        # Use MONAI SwinUNETR as backbone (encoder only)
        self.backbone = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=2,  # Dummy output
            feature_size=feature_size,
            depths=depths,
            num_heads=num_heads,
            drop_rate=dropout,
            use_checkpoint=True  # For memory efficiency
        )
        
        # Calculate output feature dimension
        # SwinUNETR bottleneck features
        self.feature_dim = feature_size * 16  # After 4 downsample stages
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - extract features
        
        Args:
            x: Input tensor (B, C, D, H, W)
        
        Returns:
            features: Encoded features (B, feature_dim)
        """
        # Get encoder features from SwinUNETR
        # Use hidden states before final decoder
        hidden = self.backbone.swinViT(x, normalize=True)
        
        # Global average pooling
        features = torch.mean(hidden, dim=[2, 3, 4])  # (B, C)
        
        return features


class DetectionHead(nn.Module):
    """
    Detection head for nodule localization
    """
    
    def __init__(self, 
                 feature_dim: int,
                 num_classes: int = 2,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.bbox_regressor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 4)  # (z, y, x, diameter)
        )
        
        self.confidence = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with:
                - class_logits: (B, num_classes)
                - bbox: (B, 4)
                - confidence: (B, 1)
        """
        class_logits = self.classifier(features)
        bbox = self.bbox_regressor(features)
        confidence = self.confidence(features)
        
        return {
            'class_logits': class_logits,
            'bbox': bbox,
            'confidence': confidence
        }


class MalignancyHead(nn.Module):
    """
    Malignancy classification head
    """
    
    def __init__(self,
                 feature_dim: int,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            malignancy_score: (B, 1) - probability of malignancy
        """
        return self.classifier(features)


class LungCancerDetectionModel(nn.Module):
    """
    Complete model with Swin Transformer backbone and dual heads
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Backbone
        self.backbone = SwinTransformer3D(
            img_size=tuple(config['preprocessing']['roi_size']),
            in_channels=1,
            feature_size=config['model']['embed_dim'],
            depths=tuple(config['model']['depths']),
            num_heads=tuple(config['model']['num_heads']),
            dropout=config['model']['dropout']
        )
        
        feature_dim = self.backbone.feature_dim
        
        # Detection head
        self.detection_head = DetectionHead(
            feature_dim=feature_dim,
            num_classes=config['detection']['num_classes'],
            hidden_dim=256
        )
        
        # Malignancy head
        self.malignancy_head = MalignancyHead(
            feature_dim=feature_dim,
            hidden_dim=config['malignancy']['hidden_dim']
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input CT volume (B, 1, D, H, W)
        
        Returns:
            Dictionary with detection and malignancy predictions
        """
        # Extract features
        features = self.backbone(x)
        
        # Detection predictions
        detection_output = self.detection_head(features)
        
        # Malignancy predictions
        malignancy_score = self.malignancy_head(features)
        
        return {
            'detection': detection_output,
            'malignancy': malignancy_score
        }


class MultiTaskLoss(nn.Module):
    """
    Combined loss for detection and malignancy classification
    """
    
    def __init__(self, 
                 detection_weight: float = 1.0,
                 malignancy_weight: float = 1.0,
                 bbox_weight: float = 0.5):
        super().__init__()
        
        self.detection_weight = detection_weight
        self.malignancy_weight = malignancy_weight
        self.bbox_weight = bbox_weight
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()
        self.smooth_l1 = nn.SmoothL1Loss()
    
    def forward(self, 
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Compute multi-task loss
        
        Args:
            predictions: Model outputs
            targets: Ground truth labels
        
        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        # Detection classification loss
        det_class_loss = self.ce_loss(
            predictions['detection']['class_logits'],
            targets['label']
        )
        
        # Bounding box regression loss
        bbox_loss = self.smooth_l1(
            predictions['detection']['bbox'],
            targets['bbox']
        )
        
        # Malignancy classification loss
        malignancy_loss = self.bce_loss(
            predictions['malignancy'],
            targets['malignancy'].unsqueeze(1)
        )
        
        # Combine losses
        total_loss = (
            self.detection_weight * det_class_loss +
            self.bbox_weight * bbox_loss +
            self.malignancy_weight * malignancy_loss
        )
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'detection_loss': det_class_loss.item(),
            'bbox_loss': bbox_loss.item(),
            'malignancy_loss': malignancy_loss.item()
        }
        
        return total_loss, loss_dict


def create_model(config: Dict) -> LungCancerDetectionModel:
    """
    Factory function to create model from config
    
    Args:
        config: Configuration dictionary
        
    Returns:
        model: LungCancerDetectionModel instance
    """
    model = LungCancerDetectionModel(config)
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count model parameters
    
    Args:
        model: PyTorch model
        
    Returns:
        total_params: Total number of parameters
        trainable_params: Number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_summary(model: nn.Module) -> Dict:
    """
    Get model summary information
    
    Args:
        model: PyTorch model
        
    Returns:
        summary: Dictionary with model information
    """
    total_params, trainable_params = count_parameters(model)
    
    summary = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        'architecture': model.__class__.__name__
    }
    
    return summary


# Example usage and testing
if __name__ == '__main__':
    """
    Test the model architecture
    """
    print("Testing Lung Cancer Detection Model...")
    
    # Create dummy config
    config = {
        'preprocessing': {
            'roi_size': [64, 64, 64]
        },
        'model': {
            'embed_dim': 48,
            'depths': [2, 2, 6, 2],
            'num_heads': [3, 6, 12, 24],
            'dropout': 0.2
        },
        'detection': {
            'num_classes': 2
        },
        'malignancy': {
            'hidden_dim': 256
        }
    }
    
    # Create model
    model = create_model(config)
    
    # Get summary
    summary = get_model_summary(model)
    print(f"\nModel Summary:")
    print(f"  Architecture: {summary['architecture']}")
    print(f"  Total parameters: {summary['total_parameters']:,}")
    print(f"  Trainable parameters: {summary['trainable_parameters']:,}")
    print(f"  Model size: ~{summary['model_size_mb']:.2f} MB")
    
    # Test forward pass
    print(f"\nTesting forward pass...")
    batch_size = 2
    dummy_input = torch.randn(batch_size, 1, 64, 64, 64)
    
    try:
        output = model(dummy_input)
        print(f"✓ Forward pass successful!")
        print(f"  Detection logits shape: {output['detection']['class_logits'].shape}")
        print(f"  Detection bbox shape: {output['detection']['bbox'].shape}")
        print(f"  Detection confidence shape: {output['detection']['confidence'].shape}")
        print(f"  Malignancy score shape: {output['malignancy'].shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {str(e)}")
    
    # Test loss computation
    print(f"\nTesting loss computation...")
    criterion = MultiTaskLoss()
    
    dummy_targets = {
        'label': torch.randint(0, 2, (batch_size,)),
        'bbox': torch.randn(batch_size, 4),
        'malignancy': torch.rand(batch_size)
    }
    
    try:
        loss, loss_dict = criterion(output, dummy_targets)
        print(f"✓ Loss computation successful!")
        print(f"  Total loss: {loss.item():.4f}")
        print(f"  Detection loss: {loss_dict['detection_loss']:.4f}")
        print(f"  BBox loss: {loss_dict['bbox_loss']:.4f}")
        print(f"  Malignancy loss: {loss_dict['malignancy_loss']:.4f}")
    except Exception as e:
        print(f"✗ Loss computation failed: {str(e)}")
    
    print(f"\n✅ Model testing complete!")