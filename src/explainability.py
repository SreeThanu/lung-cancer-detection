"""
Explainability module for visualizing model attention
Implements Grad-CAM and attention visualization for 3D medical images
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import cv2

class GradCAM3D:
    """
    3D Grad-CAM for volumetric medical images
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: Neural network model
            target_layer: Layer to compute gradients for
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Find target layer
        target_module = self._find_target_layer(self.model, self.target_layer)
        
        if target_module is not None:
            target_module.register_forward_hook(forward_hook)
            target_module.register_backward_hook(backward_hook)
    
    def _find_target_layer(self, model, target_name):
        """Find target layer by name"""
        for name, module in model.named_modules():
            if name == target_name:
                return module
        return None
    
    def generate_cam(self, 
                     input_image: torch.Tensor, 
                     target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_image: Input tensor (1, C, D, H, W)
            target_class: Target class index (if None, use max prediction)
        
        Returns:
            cam: Grad-CAM heatmap (D, H, W)
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        # Get target class
        if target_class is None:
            target_class = output['detection']['class_logits'].argmax(dim=1)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        class_score = output['detection']['class_logits'][0, target_class]
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2, 3))
        
        # Weighted combination
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def overlay_heatmap(self, 
                        image: np.ndarray, 
                        heatmap: np.ndarray, 
                        alpha: float = 0.5,
                        colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Overlay heatmap on original image
        
        Args:
            image: Original image (D, H, W)
            heatmap: Grad-CAM heatmap (D, H, W)
            alpha: Transparency factor
            colormap: OpenCV colormap
        
        Returns:
            overlayed: Image with heatmap overlay
        """
        # Resize heatmap to match image size
        if heatmap.shape != image.shape:
            heatmap_resized = []
            for i in range(image.shape[0]):
                h = cv2.resize(heatmap[i], (image.shape[2], image.shape[1]))
                heatmap_resized.append(h)
            heatmap = np.array(heatmap_resized)
        
        # Convert heatmap to RGB
        heatmap_colored = []
        for slice_idx in range(heatmap.shape[0]):
            h = (heatmap[slice_idx] * 255).astype(np.uint8)
            h = cv2.applyColorMap(h, colormap)
            h = cv2.cvtColor(h, cv2.COLOR_BGR2RGB)
            heatmap_colored.append(h)
        heatmap_colored = np.array(heatmap_colored)
        
        # Normalize image to [0, 255]
        image_norm = ((image - image.min()) / (image.max() - image.min() + 1e-6) * 255)
        image_norm = image_norm.astype(np.uint8)
        
        # Convert grayscale to RGB
        image_rgb = np.stack([image_norm] * 3, axis=-1)
        
        # Overlay
        overlayed = (alpha * heatmap_colored + (1 - alpha) * image_rgb).astype(np.uint8)
        
        return overlayed


class AttentionVisualizer:
    """
    Visualize attention maps from Swin Transformer
    """
    
    def __init__(self, model):
        self.model = model
        self.attention_maps = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture attention weights"""
        
        def hook_fn(module, input, output):
            # Capture attention weights
            if hasattr(module, 'attn'):
                self.attention_maps.append(module.attn)
        
        # Register hooks on attention layers
        for name, module in self.model.named_modules():
            if 'attn' in name.lower():
                module.register_forward_hook(hook_fn)
    
    def visualize_attention_rollout(self, 
                                     input_image: torch.Tensor) -> np.ndarray:
        """
        Compute attention rollout
        
        Args:
            input_image: Input tensor (1, C, D, H, W)
        
        Returns:
            attention_map: Aggregated attention (D, H, W)
        """
        self.attention_maps = []
        self.model.eval()
        
        with torch.no_grad():
            _ = self.model(input_image)
        
        # Aggregate attention maps
        # This is a simplified version - actual implementation depends on model
        if len(self.attention_maps) > 0:
            attention = torch.mean(torch.stack(self.attention_maps), dim=0)
            attention = attention.cpu().numpy()
            return attention
        else:
            return None


def visualize_gradcam_3d(model, 
                         input_volume: torch.Tensor,
                         original_volume: np.ndarray,
                         target_layer: str = 'backbone.backbone',
                         slice_indices: Optional[list] = None,
                         save_path: Optional[str] = None):
    """
    Generate and visualize Grad-CAM for 3D medical images
    
    Args:
        model: Trained model
        input_volume: Preprocessed input tensor (1, 1, D, H, W)
        original_volume: Original CT volume for visualization (D, H, W)
        target_layer: Layer name for Grad-CAM
        slice_indices: Specific slices to visualize (if None, use center slices)
        save_path: Path to save visualization
    """
    # Initialize Grad-CAM
    grad_cam = GradCAM3D(model, target_layer)
    
    # Generate CAM
    cam = grad_cam.generate_cam(input_volume)
    
    # Resize CAM to match original volume
    cam_resized = F.interpolate(
        torch.from_numpy(cam).unsqueeze(0).unsqueeze(0),
        size=original_volume.shape,
        mode='trilinear',
        align_corners=True
    ).squeeze().numpy()
    
    # Select slices to visualize
    if slice_indices is None:
        depth = original_volume.shape[0]
        slice_indices = [depth//4, depth//2, 3*depth//4]
    
    # Create visualization
    fig, axes = plt.subplots(len(slice_indices), 3, figsize=(15, 5*len(slice_indices)))
    
    if len(slice_indices) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, slice_idx in enumerate(slice_indices):
        # Original slice
        axes[idx, 0].imshow(original_volume[slice_idx], cmap='gray')
        axes[idx, 0].set_title(f'Original Slice {slice_idx}')
        axes[idx, 0].axis('off')
        
        # Heatmap
        axes[idx, 1].imshow(cam_resized[slice_idx], cmap='jet')
        axes[idx, 1].set_title(f'Grad-CAM Heatmap')
        axes[idx, 1].axis('off')
        
        # Overlay
        overlay = grad_cam.overlay_heatmap(
            original_volume,
            cam_resized,
            alpha=0.5
        )
        axes[idx, 2].imshow(overlay[slice_idx])
        axes[idx, 2].set_title(f'Overlay')
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {save_path}")
    
    plt.show()


def visualize_predictions_with_explainability(model,
                                               dataloader,
                                               device='cuda',
                                               num_samples=5,
                                               save_dir='results/'):
    """
    Visualize predictions with Grad-CAM explanations
    
    Args:
        model: Trained model
        dataloader: Data loader
        device: Device
        num_samples: Number of samples to visualize
        save_dir: Directory to save visualizations
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    grad_cam = GradCAM3D(model, 'backbone.backbone')
    
    print(f"\nGenerating {num_samples} explainability visualizations...")
    
    sample_count = 0
    for batch_idx, batch in enumerate(dataloader):
        if sample_count >= num_samples:
            break
        
        images = batch['image'].to(device)
        labels = batch['label']
        malignancy = batch['malignancy']
        
        # Get predictions
        with torch.no_grad():
            outputs = model(images)
        
        # Process each sample in batch
        for i in range(images.shape[0]):
            if sample_count >= num_samples:
                break
            
            # Generate Grad-CAM
            cam = grad_cam.generate_cam(images[i:i+1])
            
            # Get predictions
            det_prob = torch.softmax(
                outputs['detection']['class_logits'][i], dim=0
            )[1].item()
            mal_prob = outputs['malignancy'][i].item()
            
            # Visualize
            img = images[i, 0].cpu().numpy()
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Center slices
            d, h, w = img.shape
            slices = [d//4, d//2, 3*d//4]
            
            for idx, s in enumerate(slices):
                # Original
                axes[0, idx].imshow(img[s], cmap='gray')
                axes[0, idx].set_title(f'Slice {s}')
                axes[0, idx].axis('off')
                
                # Grad-CAM overlay
                cam_slice = cv2.resize(cam[min(s, cam.shape[0]-1)], (w, h))
                cam_colored = cv2.applyColorMap(
                    (cam_slice * 255).astype(np.uint8),
                    cv2.COLORMAP_JET
                )
                cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
                
                img_norm = ((img[s] - img[s].min()) / (img[s].max() - img[s].min() + 1e-6) * 255)
                img_rgb = np.stack([img_norm.astype(np.uint8)] * 3, axis=-1)
                
                overlay = (0.5 * cam_colored + 0.5 * img_rgb).astype(np.uint8)
                
                axes[1, idx].imshow(overlay)
                axes[1, idx].axis('off')
            
            # Add predictions text
            pred_text = (
                f"Detection Prob: {det_prob:.3f}\n"
                f"Malignancy Prob: {mal_prob:.3f}\n"
                f"True Label: {labels[i].item()}\n"
                f"True Malignancy: {malignancy[i].item():.0f}"
            )
            axes[1, 2].text(
                1.1, 0.5, pred_text,
                transform=axes[1, 2].transAxes,
                fontsize=12,
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
            
            plt.suptitle(f'Sample {sample_count + 1} - Explainability Visualization')
            plt.tight_layout()
            
            save_path = os.path.join(save_dir, f'explainability_sample_{sample_count+1}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            sample_count += 1
            print(f"✓ Generated visualization {sample_count}/{num_samples}")
    
    print(f"\n✅ All visualizations saved to {save_dir}\n")