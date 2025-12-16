"""
Evaluation metrics for lung cancer detection
"""

import torch
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    roc_auc_score, average_precision_score,
    confusion_matrix
)
from typing import Dict, List, Tuple
from tqdm import tqdm

class Evaluator:
    """
    Comprehensive evaluation for lung cancer detection
    """
    
    def __init__(self, model, data_loader, device='cuda'):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        
        # Storage for predictions and targets
        self.all_predictions = {
            'detection_probs': [],
            'detection_labels': [],
            'malignancy_probs': [],
            'malignancy_labels': [],
            'bboxes_pred': [],
            'bboxes_true': []
        }
    
    def compute_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """
        Compute IoU between two 3D bounding boxes
        bbox format: [z, y, x, diameter]
        """
        # Convert diameter to box coordinates
        z1, y1, x1, d1 = bbox1
        z2, y2, x2, d2 = bbox2
        
        # Calculate box bounds
        box1_min = np.array([z1 - d1/2, y1 - d1/2, x1 - d1/2])
        box1_max = np.array([z1 + d1/2, y1 + d1/2, x1 + d1/2])
        box2_min = np.array([z2 - d2/2, y2 - d2/2, x2 - d2/2])
        box2_max = np.array([z2 + d2/2, y2 + d2/2, x2 + d2/2])
        
        # Intersection
        inter_min = np.maximum(box1_min, box2_min)
        inter_max = np.minimum(box1_max, box2_max)
        inter_dims = np.maximum(0, inter_max - inter_min)
        intersection = np.prod(inter_dims)
        
        # Union
        vol1 = d1 ** 3
        vol2 = d2 ** 3
        union = vol1 + vol2 - intersection
        
        iou = intersection / (union + 1e-6)
        return iou
    
    def collect_predictions(self):
        """
        Collect all predictions from the model
        """
        self.model.eval()
        
        print("Collecting predictions...")
        with torch.no_grad():
            for batch in tqdm(self.data_loader):
                images = batch['image'].to(self.device)
                labels = batch['label']
                malignancy = batch['malignancy']
                bbox_true = batch['bbox']
                
                # Get predictions
                outputs = self.model(images)
                
                # Detection predictions
                det_probs = torch.softmax(
                    outputs['detection']['class_logits'], dim=1
                )[:, 1]  # Probability of nodule class
                
                # Malignancy predictions
                mal_probs = outputs['malignancy']
                
                # Bounding boxes
                bbox_pred = outputs['detection']['bbox']
                
                # Store predictions
                self.all_predictions['detection_probs'].extend(
                    det_probs.cpu().numpy()
                )
                self.all_predictions['detection_labels'].extend(
                    labels.numpy()
                )
                self.all_predictions['malignancy_probs'].extend(
                    mal_probs.cpu().numpy()
                )
                self.all_predictions['malignancy_labels'].extend(
                    malignancy.numpy()
                )
                self.all_predictions['bboxes_pred'].extend(
                    bbox_pred.cpu().numpy()
                )
                self.all_predictions['bboxes_true'].extend(
                    bbox_true.numpy()
                )
    
    def compute_detection_metrics(self, threshold: float = 0.5) -> Dict:
        """
        Compute detection metrics (F1, precision, recall)
        """
        probs = np.array(self.all_predictions['detection_probs'])
        labels = np.array(self.all_predictions['detection_labels'])
        
        # Threshold predictions
        preds = (probs > threshold).astype(int)
        
        # Calculate metrics
        f1 = f1_score(labels, preds)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        
        # AUC
        if len(np.unique(labels)) > 1:
            auc = roc_auc_score(labels, probs)
            ap = average_precision_score(labels, probs)
        else:
            auc = 0.0
            ap = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        
        return {
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'auc': auc,
            'average_precision': ap,
            'confusion_matrix': cm
        }
    
    def compute_malignancy_metrics(self, threshold: float = 0.5) -> Dict:
        """
        Compute malignancy classification metrics
        """
        probs = np.array(self.all_predictions['malignancy_probs']).flatten()
        labels = np.array(self.all_predictions['malignancy_labels'])
        
        # Threshold predictions
        preds = (probs > threshold).astype(int)
        
        # Calculate metrics
        f1 = f1_score(labels, preds, zero_division=0)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        
        # AUC
        if len(np.unique(labels)) > 1:
            auc = roc_auc_score(labels, probs)
        else:
            auc = 0.0
        
        return {
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
    
    def compute_bbox_metrics(self, iou_threshold: float = 0.5) -> Dict:
        """
        Compute bounding box localization metrics
        """
        bboxes_pred = np.array(self.all_predictions['bboxes_pred'])
        bboxes_true = np.array(self.all_predictions['bboxes_true'])
        
        ious = []
        for pred, true in zip(bboxes_pred, bboxes_true):
            iou = self.compute_iou(pred, true)
            ious.append(iou)
        
        ious = np.array(ious)
        
        # Mean IoU
        mean_iou = np.mean(ious)
        
        # Localization accuracy (IoU > threshold)
        loc_acc = np.mean(ious > iou_threshold)
        
        return {
            'mean_iou': mean_iou,
            'localization_accuracy': loc_acc,
            'iou_std': np.std(ious)
        }
    
    def compute_false_positives_per_scan(self, threshold: float = 0.5) -> float:
        """
        Compute false positives per scan
        Critical metric for clinical viability
        """
        probs = np.array(self.all_predictions['detection_probs'])
        labels = np.array(self.all_predictions['detection_labels'])
        
        preds = (probs > threshold).astype(int)
        
        # False positives
        fp = np.sum((preds == 1) & (labels == 0))
        
        # Total number of scans
        num_scans = len(probs)
        
        fps_per_scan = fp / num_scans if num_scans > 0 else 0
        
        return fps_per_scan
    
    def evaluate(self) -> Dict:
        """
        Run complete evaluation pipeline
        
        Returns:
            Dictionary with all metrics
        """
        # Collect predictions
        self.collect_predictions()
        
        print("\nComputing metrics...")
        
        # Detection metrics
        det_metrics = self.compute_detection_metrics()
        
        # Malignancy metrics
        mal_metrics = self.compute_malignancy_metrics()
        
        # Bounding box metrics
        bbox_metrics = self.compute_bbox_metrics()
        
        # False positives per scan
        fps_per_scan = self.compute_false_positives_per_scan()
        
        # Combine all metrics
        results = {
            'detection': det_metrics,
            'malignancy': mal_metrics,
            'bbox': bbox_metrics,
            'fps_per_scan': fps_per_scan
        }
        
        return results
    
    def print_results(self, results: Dict):
        """
        Print evaluation results in a formatted way
        """
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        print("\n📊 Detection Metrics:")
        print(f"  F1 Score:           {results['detection']['f1_score']:.4f}")
        print(f"  Precision:          {results['detection']['precision']:.4f}")
        print(f"  Recall:             {results['detection']['recall']:.4f}")
        print(f"  AUC:                {results['detection']['auc']:.4f}")
        print(f"  Average Precision:  {results['detection']['average_precision']:.4f}")
        
        print("\n🎯 Malignancy Classification:")
        print(f"  F1 Score:           {results['malignancy']['f1_score']:.4f}")
        print(f"  Precision:          {results['malignancy']['precision']:.4f}")
        print(f"  Recall:             {results['malignancy']['recall']:.4f}")
        print(f"  AUC:                {results['malignancy']['auc']:.4f}")
        
        print("\n📍 Bounding Box Localization:")
        print(f"  Mean IoU:           {results['bbox']['mean_iou']:.4f}")
        print(f"  Localization Acc:   {results['bbox']['localization_accuracy']:.4f}")
        
        print("\n⚠️  Clinical Metrics:")
        print(f"  FPs per Scan:       {results['fps_per_scan']:.4f}")
        
        print("\n" + "="*60)
        
        # Check against targets
        print("\n✅ Target Achievement:")
        print(f"  F1 > 0.80:          {'✓' if results['detection']['f1_score'] > 0.80 else '✗'}")
        print(f"  AUC > 0.85:         {'✓' if results['malignancy']['auc'] > 0.85 else '✗'}")
        print(f"  FPs/scan < 2:       {'✓' if results['fps_per_scan'] < 2 else '✗'}")
        print("="*60 + "\n")


def evaluate_model(model, data_loader, device='cuda') -> Dict:
    """
    Convenience function to evaluate a model
    """
    evaluator = Evaluator(model, data_loader, device)
    results = evaluator.evaluate()
    evaluator.print_results(results)
    return results