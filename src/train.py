"""
Training module for 3D Swin Transformer Lung Cancer Detection
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import numpy as np


class Trainer:
    """
    Trainer class for lung cancer detection model
    Handles training loop, validation, checkpointing, and logging
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader,
                 val_loader,
                 criterion: nn.Module,
                 config: Dict,
                 device: torch.device):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            config: Configuration dictionary
            device: Device to train on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.config = config
        self.device = device
        
        # Training parameters
        self.num_epochs = config['training'].get('num_epochs', config['training'].get('epochs', 50))
        self.learning_rate = config['training']['learning_rate']
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        
        # Setup learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Create checkpoint directory
        self.checkpoint_dir = config['logging']['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def train_epoch(self) -> Dict:
        """
        Train for one epoch
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        running_loss = 0.0
        running_metrics = {
            'detection_loss': 0.0,
            'bbox_loss': 0.0,
            'malignancy_loss': 0.0
        }
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            malignancy = batch['malignancy'].to(self.device)
            bbox = batch['bbox'].to(self.device)
            
            # Prepare targets
            targets = {
                'label': labels,
                'malignancy': malignancy,
                'bbox': bbox
            }
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(images)
            
            # Compute loss
            loss, loss_dict = self.criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate metrics
            running_loss += loss.item()
            running_metrics['detection_loss'] += loss_dict['detection_loss']
            running_metrics['bbox_loss'] += loss_dict['bbox_loss']
            running_metrics['malignancy_loss'] += loss_dict['malignancy_loss']
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'det': f'{loss_dict["detection_loss"]:.4f}',
                'bbox': f'{loss_dict["bbox_loss"]:.4f}',
                'mal': f'{loss_dict["malignancy_loss"]:.4f}'
            })
        
        # Average metrics
        num_batches = len(self.train_loader)
        epoch_loss = running_loss / num_batches
        epoch_metrics = {k: v / num_batches for k, v in running_metrics.items()}
        
        return {
            'loss': epoch_loss,
            **epoch_metrics
        }
    
    def validate(self) -> Dict:
        """
        Validate model
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        running_loss = 0.0
        running_metrics = {
            'detection_loss': 0.0,
            'bbox_loss': 0.0,
            'malignancy_loss': 0.0
        }
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for batch in pbar:
                # Move data to device
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                malignancy = batch['malignancy'].to(self.device)
                bbox = batch['bbox'].to(self.device)
                
                # Prepare targets
                targets = {
                    'label': labels,
                    'malignancy': malignancy,
                    'bbox': bbox
                }
                
                # Forward pass
                predictions = self.model(images)
                
                # Compute loss
                loss, loss_dict = self.criterion(predictions, targets)
                
                # Accumulate metrics
                running_loss += loss.item()
                running_metrics['detection_loss'] += loss_dict['detection_loss']
                running_metrics['bbox_loss'] += loss_dict['bbox_loss']
                running_metrics['malignancy_loss'] += loss_dict['malignancy_loss']
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'det': f'{loss_dict["detection_loss"]:.4f}',
                    'bbox': f'{loss_dict["bbox_loss"]:.4f}',
                    'mal': f'{loss_dict["malignancy_loss"]:.4f}'
                })
        
        # Average metrics
        num_batches = len(self.val_loader)
        epoch_loss = running_loss / num_batches
        epoch_metrics = {k: v / num_batches for k, v in running_metrics.items()}
        
        return {
            'loss': epoch_loss,
            **epoch_metrics
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': self.history['val_loss'][-1],
            'history': self.history
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"\n✓ Saved best model (val_loss: {checkpoint['val_loss']:.4f})")
    
    def train(self) -> Dict:
        """
        Main training loop
        
        Returns:
            Training history dictionary
        """
        print(f"\n{'='*60}")
        print(f"Starting training for {self.num_epochs} epochs")
        print(f"{'='*60}\n")
        
        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            print("-" * 60)
            
            # Train
            train_metrics = self.train_epoch()
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_metrics'].append(train_metrics)
            
            # Validate
            val_metrics = self.validate()
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_metrics'].append(val_metrics)
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f}")
            print(f"  LR:         {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Check if best model
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.best_epoch = epoch
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
        
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Best model at epoch {self.best_epoch} (val_loss: {self.best_val_loss:.4f})")
        print(f"{'='*60}\n")
        
        return self.history
