"""
Utility functions for lung cancer detection project
"""

import os
import json
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import SimpleITK as sitk

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: Dict, save_path: str):
    """Save configuration to YAML file"""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def create_directories(config: Dict):
    """Create necessary directories for the project"""
    dirs_to_create = [
        config['data']['root_dir'],
        config['data']['raw_dir'],
        config['data']['processed_dir'],
        config['data']['roi_patches_dir'],
        config['logging']['experiment_dir'],
        config['logging']['checkpoint_dir'],
        config['logging']['results_dir']
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"✓ Created {len(dirs_to_create)} directories")

def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Count model parameters
    
    Returns:
        total_params: Total number of parameters
        trainable_params: Number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def print_model_summary(model: torch.nn.Module):
    """Print model architecture summary"""
    total, trainable = count_parameters(model)
    
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Model size: ~{total * 4 / 1024 / 1024:.2f} MB")
    print("="*60 + "\n")

def visualize_ct_slice(image: np.ndarray, 
                       slice_idx: int,
                       title: str = "CT Slice",
                       save_path: str = None):
    """Visualize a single CT slice"""
    plt.figure(figsize=(8, 8))
    plt.imshow(image[slice_idx], cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.colorbar()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

def visualize_volume_slices(volume: np.ndarray,
                            num_slices: int = 9,
                            title: str = "Volume Slices",
                            save_path: str = None):
    """Visualize multiple slices from a 3D volume"""
    depth = volume.shape[0]
    indices = np.linspace(0, depth-1, num_slices, dtype=int)
    
    rows = int(np.sqrt(num_slices))
    cols = int(np.ceil(num_slices / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten()
    
    for idx, slice_idx in enumerate(indices):
        axes[idx].imshow(volume[slice_idx], cmap='gray')
        axes[idx].set_title(f'Slice {slice_idx}')
        axes[idx].axis('off')
    
    # Hide extra subplots
    for idx in range(num_slices, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

def load_ct_scan(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load CT scan from file
    
    Returns:
        image: 3D numpy array
        origin: Physical coordinates origin
        spacing: Voxel spacing
    """
    itk_image = sitk.ReadImage(path)
    image = sitk.GetArrayFromImage(itk_image)
    origin = np.array(itk_image.GetOrigin())
    spacing = np.array(itk_image.GetSpacing())
    
    return image, origin, spacing

def save_predictions(predictions: Dict, save_path: str):
    """Save predictions to JSON file"""
    # Convert numpy arrays to lists
    serializable_preds = {}
    for key, value in predictions.items():
        if isinstance(value, np.ndarray):
            serializable_preds[key] = value.tolist()
        elif isinstance(value, torch.Tensor):
            serializable_preds[key] = value.cpu().numpy().tolist()
        else:
            serializable_preds[key] = value
    
    with open(save_path, 'w') as f:
        json.dump(serializable_preds, f, indent=2)
    
    print(f"✓ Saved predictions to {save_path}")

def load_predictions(load_path: str) -> Dict:
    """Load predictions from JSON file"""
    with open(load_path, 'r') as f:
        predictions = json.load(f)
    return predictions

def plot_training_curves(history: Dict, save_path: str = None):
    """Plot training and validation curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Detection Loss
    if 'train_metrics' in history:
        det_train = [m.get('detection_loss', 0) for m in history['train_metrics']]
        det_val = [m.get('detection_loss', 0) for m in history['val_metrics']]
        axes[0, 1].plot(det_train, label='Train')
        axes[0, 1].plot(det_val, label='Validation')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Detection Loss')
        axes[0, 1].set_title('Detection Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # BBox Loss
        bbox_train = [m.get('bbox_loss', 0) for m in history['train_metrics']]
        bbox_val = [m.get('bbox_loss', 0) for m in history['val_metrics']]
        axes[1, 0].plot(bbox_train, label='Train')
        axes[1, 0].plot(bbox_val, label='Validation')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('BBox Loss')
        axes[1, 0].set_title('Bounding Box Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Malignancy Loss
        mal_train = [m.get('malignancy_loss', 0) for m in history['train_metrics']]
        mal_val = [m.get('malignancy_loss', 0) for m in history['val_metrics']]
        axes[1, 1].plot(mal_train, label='Train')
        axes[1, 1].plot(mal_val, label='Validation')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Malignancy Loss')
        axes[1, 1].set_title('Malignancy Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved training curves to {save_path}")
    
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, 
                          class_names: List[str],
                          title: str = "Confusion Matrix",
                          save_path: str = None):
    """Plot confusion matrix"""
    import seaborn as sns
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_roc_curve(fpr: np.ndarray, 
                   tpr: np.ndarray, 
                   auc: float,
                   title: str = "ROC Curve",
                   save_path: str = None):
    """Plot ROC curve"""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("⚠ Using CPU (training will be slow)")
    return device


def save_checkpoint(model: torch.nn.Module,
                    optimizer,
                    epoch: int,
                    loss: float,
                    save_path: str,
                    **kwargs):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, save_path)
    print(f"✓ Saved checkpoint to {save_path}")


def load_checkpoint(model: torch.nn.Module,
                    optimizer,
                    load_path: str,
                    device: str = 'cuda'):
    """Load model checkpoint"""
    checkpoint = torch.load(load_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"✓ Loaded checkpoint from epoch {epoch} (loss: {loss:.4f})")
    
    return model, optimizer, epoch, loss


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def print_metrics_table(metrics: Dict):
    """Print metrics in a formatted table"""
    print("\n" + "="*60)
    print("METRICS SUMMARY")
    print("="*60)
    
    for category, values in metrics.items():
        print(f"\n{category.upper()}:")
        if isinstance(values, dict):
            for key, value in values.items():
                if isinstance(value, (int, float)):
                    print(f"  {key:.<40} {value:.4f}")
        else:
            print(f"  {category:.<40} {values:.4f}")
    
    print("="*60 + "\n")


def create_experiment_dir(base_dir: str, experiment_name: str = None) -> str:
    """Create timestamped experiment directory"""
    from datetime import datetime
    
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"exp_{timestamp}"
    
    exp_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'visualizations'), exist_ok=True)
    
    print(f"✓ Created experiment directory: {exp_dir}")
    
    return exp_dir


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def log_system_info():
    """Log system and environment information"""
    import platform
    
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    
    print("="*60 + "\n")