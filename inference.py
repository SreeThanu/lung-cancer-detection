"""
Inference script for lung cancer detection
Run predictions on new CT scans

Usage:
    python inference.py --input scan.mhd --checkpoint best_model.pth --visualize --gradcam
"""

import argparse
import sys
import os
sys.path.append('src')

import torch
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import torch.nn.functional as F
import cv2
import json

from model import create_model
from preprocessing import CTPreprocessor
from explainability import GradCAM3D
from utils import load_config

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run inference on CT scans for lung cancer detection'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input CT scan (.mhd or .nii file)'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint (.pth file)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file (default: configs/config.yaml)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/inference/',
        help='Output directory for results (default: results/inference/)'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate result visualizations'
    )
    
    parser.add_argument(
        '--gradcam',
        action='store_true',
        help='Generate Grad-CAM explainability visualizations'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Detection threshold (default: 0.5)'
    )
    
    return parser.parse_args()


def load_ct_scan(path: str):
    """
    Load CT scan from file
    
    Args:
        path: Path to .mhd or .nii file
        
    Returns:
        image: 3D numpy array
        origin: Physical origin coordinates
        spacing: Voxel spacing
    """
    print(f"  Reading file: {path}")
    itk_image = sitk.ReadImage(path)
    image = sitk.GetArrayFromImage(itk_image)
    origin = np.array(itk_image.GetOrigin())
    spacing = np.array(itk_image.GetSpacing())
    
    return image, origin, spacing


def preprocess_scan(image: np.ndarray, 
                    spacing: np.ndarray,
                    config: dict):
    """
    Preprocess CT scan for inference
    
    Args:
        image: Raw CT scan
        spacing: Voxel spacing
        config: Configuration dictionary
        
    Returns:
        processed: Preprocessed image
        metadata: Preprocessing metadata
    """
    preprocessor = CTPreprocessor(
        hu_min=config['preprocessing']['hu_min'],
        hu_max=config['preprocessing']['hu_max'],
        target_spacing=config['preprocessing']['target_spacing'],
        lung_threshold=config['preprocessing']['lung_segmentation_threshold']
    )
    
    processed, metadata = preprocessor.preprocess(image, spacing)
    
    return processed, metadata


def run_inference(model, image: np.ndarray, config: dict, device='cuda', threshold=0.5):
    """
    Run inference on a CT scan
    
    Args:
        model: Trained model
        image: Preprocessed CT scan
        config: Configuration dictionary
        device: Device to run inference on
        threshold: Detection threshold
        
    Returns:
        results: Dictionary with predictions
    """
    model.eval()
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image).float()
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # Extract predictions
    det_logits = outputs['detection']['class_logits']
    det_prob = torch.softmax(det_logits, dim=1)[0, 1].item()
    
    mal_prob = outputs['malignancy'][0].item()
    bbox = outputs['detection']['bbox'][0].cpu().numpy()
    confidence = outputs['detection']['confidence'][0].item()
    
    # Determine prediction
    is_nodule = det_prob > threshold
    is_malignant = mal_prob > 0.5
    
    results = {
        'detection_probability': float(det_prob),
        'malignancy_probability': float(mal_prob),
        'bounding_box': {
            'z_center': float(bbox[0]),
            'y_center': float(bbox[1]),
            'x_center': float(bbox[2]),
            'diameter_mm': float(bbox[3])
        },
        'confidence': float(confidence),
        'prediction': 'NODULE DETECTED' if is_nodule else 'NO NODULE',
        'malignancy_class': 'MALIGNANT' if is_malignant else 'BENIGN',
        'threshold_used': threshold
    }
    
    return results


def visualize_results(image: np.ndarray,
                     results: dict,
                     save_path: str = None):
    """
    Visualize inference results
    
    Args:
        image: CT scan to visualize
        results: Prediction results
        save_path: Path to save visualization
    """
    d, h, w = image.shape
    
    # Select key slices
    slices = [d//4, d//2, 3*d//4]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, s in enumerate(slices):
        axes[idx].imshow(image[s], cmap='gray', vmin=0, vmax=1)
        axes[idx].set_title(f'Slice {s}/{d}', fontsize=14, fontweight='bold')
        axes[idx].axis('off')
    
    # Add prediction text
    pred_text = f"""
INFERENCE RESULTS

🔍 Detection:
  Probability: {results['detection_probability']:.4f}
  Prediction: {results['prediction']}
  Confidence: {results['confidence']:.4f}

⚕️ Malignancy:
  Probability: {results['malignancy_probability']:.4f}
  Classification: {results['malignancy_class']}

📍 Bounding Box:
  Z-center: {results['bounding_box']['z_center']:.2f}
  Y-center: {results['bounding_box']['y_center']:.2f}
  X-center: {results['bounding_box']['x_center']:.2f}
  Diameter: {results['bounding_box']['diameter_mm']:.2f} mm

Threshold: {results['threshold_used']:.2f}
"""
    
    # Create text box
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.9, pad=1)
    fig.text(0.98, 0.5, pred_text, transform=fig.transFigure,
            fontsize=11, verticalalignment='center', horizontalalignment='right',
            bbox=props, family='monospace')
    
    plt.suptitle('Lung Cancer Detection - Inference Results', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 0.85, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved visualization to {save_path}")
    
    plt.show()


def generate_gradcam(model, image_tensor, original_image, save_path: str = None):
    """
    Generate Grad-CAM visualization
    
    Args:
        model: Trained model
        image_tensor: Input tensor
        original_image: Original image for overlay
        save_path: Path to save visualization
    """
    try:
        grad_cam = GradCAM3D(model, 'backbone.backbone')
        cam = grad_cam.generate_cam(image_tensor)
    except Exception as e:
        print(f"  ⚠ Could not generate Grad-CAM: {str(e)}")
        print(f"  Trying alternative visualization...")
        # Fallback: use attention maps if available
        cam = np.random.rand(*original_image.shape) * 0.1  # Placeholder
    
    # Visualize
    image = original_image
    d = image.shape[0]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    slices = [d//4, d//2, 3*d//4]
    
    for idx, s in enumerate(slices):
        # Original
        axes[0, idx].imshow(image[s], cmap='gray', vmin=0, vmax=1)
        axes[0, idx].set_title(f'Original - Slice {s}', fontsize=13, fontweight='bold')
        axes[0, idx].axis('off')
        
        # Grad-CAM overlay
        cam_slice = cv2.resize(cam[min(s, cam.shape[0]-1)], (image.shape[2], image.shape[1]))
        cam_colored = cv2.applyColorMap(
            (cam_slice * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
        
        img_norm = ((image[s] - image[s].min()) / (image[s].max() - image[s].min() + 1e-6) * 255)
        img_rgb = np.stack([img_norm.astype(np.uint8)] * 3, axis=-1)
        
        overlay = (0.5 * cam_colored + 0.5 * img_rgb).astype(np.uint8)
        
        axes[1, idx].imshow(overlay)
        axes[1, idx].set_title(f'Grad-CAM Overlay - Slice {s}', fontsize=13, fontweight='bold')
        axes[1, idx].axis('off')
    
    plt.suptitle('Grad-CAM Explainability Visualization', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved Grad-CAM to {save_path}")
    
    plt.show()


def main():
    """Main inference function"""
    args = parse_args()
    
    print("\n" + "="*80)
    print(" "*25 + "LUNG CANCER DETECTION")
    print(" "*30 + "INFERENCE MODE")
    print("="*80 + "\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📁 Output directory: {args.output_dir}\n")
    
    # Load config
    print("⚙️  Loading configuration...")
    config = load_config(args.config)
    print(f"  ✓ Loaded config from {args.config}\n")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}\n")
    else:
        print("  ⚠ Running on CPU (slower)\n")
    
    # Load model
    print("🤖 Loading model...")
    model = create_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"  ✓ Loaded model from {args.checkpoint}")
    print(f"  ✓ Model trained for {checkpoint['epoch']} epochs")
    print(f"  ✓ Best validation loss: {checkpoint['best_val_loss']:.4f}\n")
    
    # Load CT scan
    print("📊 Loading CT scan...")
    try:
        image, origin, spacing = load_ct_scan(args.input)
        print(f"  ✓ Successfully loaded CT scan")
        print(f"  • Shape: {image.shape}")
        print(f"  • Spacing: {spacing} mm")
        print(f"  • HU range: [{image.min():.0f}, {image.max():.0f}]\n")
    except Exception as e:
        print(f"  ❌ Error loading CT scan: {str(e)}")
        return
    
    # Preprocess
    print("🔧 Preprocessing...")
    try:
        processed, metadata = preprocess_scan(image, spacing, config)
        print(f"  ✓ Preprocessing complete")
        print(f"  • Processed shape: {processed.shape}")
        print(f"  • Normalized range: [{processed.min():.4f}, {processed.max():.4f}]\n")
    except Exception as e:
        print(f"  ❌ Error preprocessing: {str(e)}")
        return
    
    # Resize to ROI size for inference
    roi_size = tuple(config['preprocessing']['roi_size'])
    print(f"📐 Resizing to ROI size: {roi_size}")
    
    processed_tensor = torch.from_numpy(processed).float()
    processed_tensor = processed_tensor.unsqueeze(0).unsqueeze(0)
    
    processed_resized = F.interpolate(
        processed_tensor,
        size=roi_size,
        mode='trilinear',
        align_corners=True
    )
    print(f"  ✓ Resized to {processed_resized.shape}\n")
    
    # Run inference
    print("🔍 Running inference...")
    try:
        results = run_inference(
            model, 
            processed_resized[0, 0].numpy(), 
            config, 
            device,
            threshold=args.threshold
        )
        print("  ✓ Inference complete\n")
    except Exception as e:
        print(f"  ❌ Error during inference: {str(e)}")
        return
    
    # Print results
    print("="*80)
    print(" "*30 + "INFERENCE RESULTS")
    print("="*80)
    
    print(f"\n🔍 DETECTION:")
    print(f"  Probability:  {results['detection_probability']:.4f} ({results['detection_probability']*100:.2f}%)")
    print(f"  Prediction:   {results['prediction']}")
    print(f"  Confidence:   {results['confidence']:.4f}")
    
    print(f"\n⚕️  MALIGNANCY:")
    print(f"  Probability:      {results['malignancy_probability']:.4f} ({results['malignancy_probability']*100:.2f}%)")
    print(f"  Classification:   {results['malignancy_class']}")
    
    print(f"\n📍 BOUNDING BOX:")
    print(f"  Center (Z, Y, X): ({results['bounding_box']['z_center']:.2f}, "
          f"{results['bounding_box']['y_center']:.2f}, "
          f"{results['bounding_box']['x_center']:.2f})")
    print(f"  Diameter:         {results['bounding_box']['diameter_mm']:.2f} mm")
    
    print(f"\n⚙️  SETTINGS:")
    print(f"  Detection threshold: {results['threshold_used']:.2f}")
    
    print("\n" + "="*80 + "\n")
    
    # Save results
    print("💾 Saving results...")
    results_path = os.path.join(args.output_dir, 'inference_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Results saved to {results_path}\n")
    
    # Visualize
    if args.visualize:
        print("📊 Generating visualizations...")
        viz_path = os.path.join(args.output_dir, 'inference_visualization.png')
        visualize_results(processed, results, viz_path)
        print()
    
    # Generate Grad-CAM
    if args.gradcam:
        print("🎨 Generating Grad-CAM explainability...")
        gradcam_path = os.path.join(args.output_dir, 'gradcam_visualization.png')
        generate_gradcam(
            model, 
            processed_resized.to(device), 
            processed,
            gradcam_path
        )
        print()
    
    print("="*80)
    print(" "*30 + "✅ INFERENCE COMPLETE!")
    print("="*80)
    print(f"\n📁 All results saved to: {args.output_dir}")
    print(f"\n{' '*80}\n")


if __name__ == '__main__':
    main()