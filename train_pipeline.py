"""
Main training pipeline script
Can be run from command line
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

import torch
import numpy as np
from sklearn.model_selection import train_test_split

from data_loader import LUNA16Dataset, create_data_loaders
from preprocessing import create_augmentation
from model import create_model, MultiTaskLoss
from train import Trainer
from evaluate import evaluate_model
from explainability import visualize_predictions_with_explainability
from utils import (
    load_config, create_directories, set_seed, 
    get_device, log_system_info, print_model_summary,
    create_experiment_dir
)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train Lung Cancer Detection Model'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Experiment name'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (overrides config)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    parser.add_argument(
        '--no-eval',
        action='store_true',
        help='Skip evaluation after training'
    )
    
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Skip explainability visualizations'
    )
    
    return parser.parse_args()


def main():
    """Main training function"""
    
    # Parse arguments
    args = parse_args()
    
    print("\n" + "="*60)
    print("LUNG CANCER DETECTION - TRAINING PIPELINE")
    print("="*60 + "\n")
    
    # Set random seed
    set_seed(args.seed)
    print(f"✓ Set random seed: {args.seed}")
    
    # Log system info
    log_system_info()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    print(f"✓ Loaded config from {args.config}")
    
    # Override config with command line arguments
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    
    # Create directories
    create_directories(config)
    
    # Create experiment directory
    exp_dir = create_experiment_dir(
        config['logging']['experiment_dir'],
        args.experiment_name
    )
    
    # Update config with experiment directory
    config['logging']['checkpoint_dir'] = os.path.join(exp_dir, 'checkpoints')
    config['logging']['results_dir'] = os.path.join(exp_dir, 'visualizations')
    
    # Get device
    device = get_device()
    
    # ========================================================================
    # DATA LOADING
    # ========================================================================
    
    print("\n" + "="*60)
    print("PREPARING DATA")
    print("="*60)
    
    # Create augmentation
    augmentation = create_augmentation(config)
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = LUNA16Dataset(
        data_dir=config['data']['processed_dir'],
        annotations_file=config['data']['annotations_file'],
        roi_size=tuple(config['preprocessing']['roi_size']),
        transform=augmentation,
        mode='train'
    )
    
    val_dataset = LUNA16Dataset(
        data_dir=config['data']['processed_dir'],
        annotations_file=config['data']['annotations_file'],
        roi_size=tuple(config['preprocessing']['roi_size']),
        transform=None,
        mode='val'
    )
    
    # Split datasets
    train_indices, val_indices = train_test_split(
        np.arange(len(train_dataset)),
        test_size=0.2,
        random_state=args.seed
    )
    
    train_dataset.samples = [train_dataset.samples[i] for i in train_indices]
    val_dataset.samples = [val_dataset.samples[i] for i in val_indices]
    
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        config, train_dataset, val_dataset
    )
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    
    # ========================================================================
    # MODEL CREATION
    # ========================================================================
    
    print("\n" + "="*60)
    print("CREATING MODEL")
    print("="*60)
    
    model = create_model(config)
    model = model.to(device)
    
    print_model_summary(model)
    
    # ========================================================================
    # TRAINING
    # ========================================================================
    
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    # Create loss function
    criterion = MultiTaskLoss(
        detection_weight=1.0,
        malignancy_weight=1.0,
        bbox_weight=0.5
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        config=config,
        device=device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✓ Resumed from checkpoint: {args.resume}")
    
    # Train
    history = trainer.train()
    
    print("\n✅ Training complete!")
    
    # ========================================================================
    # EVALUATION
    # ========================================================================
    
    if not args.no_eval:
        print("\n" + "="*60)
        print("EVALUATION")
        print("="*60)
        
        # Load best model
        best_model_path = os.path.join(
            config['logging']['checkpoint_dir'],
            'best_model.pth'
        )
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✓ Loaded best model from epoch {checkpoint['epoch']}")
        
        # Evaluate
        results = evaluate_model(model, val_loader, device)
        
        # Save results
        import json
        results_path = os.path.join(exp_dir, 'evaluation_results.json')
        
        # Convert numpy arrays to lists
        results_serializable = {}
        for key, value in results.items():
            if isinstance(value, dict):
                results_serializable[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        results_serializable[key][k] = v.tolist()
                    else:
                        results_serializable[key][k] = v
            else:
                results_serializable[key] = value
        
        with open(results_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"\n✓ Results saved to {results_path}")
    
    # ========================================================================
    # EXPLAINABILITY VISUALIZATIONS
    # ========================================================================
    
    if not args.no_viz:
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        viz_dir = os.path.join(exp_dir, 'explainability')
        visualize_predictions_with_explainability(
            model=model,
            dataloader=val_loader,
            device=device,
            num_samples=5,
            save_dir=viz_dir
        )
        
        print(f"\n✓ Visualizations saved to {viz_dir}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"\n📁 Experiment directory: {exp_dir}")
    print("\n✅ Generated:")
    print("  - Trained model checkpoints")
    print("  - Training history")
    if not args.no_eval:
        print("  - Evaluation metrics")
    if not args.no_viz:
        print("  - Explainability visualizations")
    print("\n" + "="*60 + "\n")


if __name__ == '__main__':
    main()