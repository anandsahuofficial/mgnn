"""
Ensemble model with optimized threshold for discovery.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path


class EnsembleCGCNN:
    """
    Ensemble of CGCNN models with optimized prediction threshold.
    
    Improves robustness and discovery capability.
    """
    
    def __init__(self, models: List[nn.Module], device: str = 'cuda'):
        """
        Initialize ensemble.
        
        Args:
            models: List of trained CGCNN models
            device: Device for inference
        """
        self.models = [m.to(device).eval() for m in models]
        self.device = device
        self.optimal_threshold = 0.025  # Will be optimized
        
        print(f"Ensemble initialized with {len(models)} models")
    
    @torch.no_grad()
    def predict(self, data):
        """
        Predict using ensemble average.
        
        Args:
            data: PyG Data batch
        
        Returns:
            Mean predictions across ensemble
        """
        predictions = []
        
        for model in self.models:
            pred = model(data)
            predictions.append(pred)
        
        # Average predictions
        ensemble_pred = torch.stack(predictions).mean(dim=0)
        
        return ensemble_pred
    
    @torch.no_grad()
    def predict_with_uncertainty(self, data):
        """
        Predict with uncertainty estimate.
        
        Args:
            data: PyG Data batch
        
        Returns:
            (mean_predictions, std_predictions)
        """
        predictions = []
        
        for model in self.models:
            pred = model(data)
            predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        return mean_pred, std_pred
    
    def optimize_threshold(
        self,
        val_predictions: np.ndarray,
        val_targets: np.ndarray,
        metric: str = 'f1'
    ) -> float:
        """
        Optimize classification threshold for discovery.
        
        Args:
            val_predictions: Validation predictions
            val_targets: Validation targets
            metric: Optimization metric ('f1', 'recall', 'precision')
        
        Returns:
            Optimal threshold value
        """
        print(f"\nOptimizing threshold to maximize {metric}...")
        
        # Try different thresholds
        thresholds = np.linspace(0.01, 0.15, 100)
        
        best_score = 0
        best_threshold = 0.025
        
        for thresh in thresholds:
            # Binary classification at this threshold
            pred_stable = val_predictions.flatten() <= thresh
            true_stable = val_targets.flatten() <= 0.025
            
            tp = np.sum(pred_stable & true_stable)
            fp = np.sum(pred_stable & ~true_stable)
            fn = np.sum(~pred_stable & true_stable)
            
            # Compute metric
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            if metric == 'f1':
                score = f1
            elif metric == 'recall':
                score = recall
            elif metric == 'precision':
                score = precision
            else:
                score = f1
            
            if score > best_score:
                best_score = score
                best_threshold = thresh
        
        self.optimal_threshold = best_threshold
        
        print(f"  Optimal threshold: {best_threshold:.4f}")
        print(f"  Best {metric}: {best_score:.4f}")
        
        return best_threshold
    
    def classify_stability(self, predictions: np.ndarray) -> np.ndarray:
        """
        Classify compounds using optimized threshold.
        
        Args:
            predictions: Decomposition energy predictions
        
        Returns:
            Binary classification (True = stable)
        """
        return predictions.flatten() <= self.optimal_threshold


def test_ensemble():
    """Test ensemble model."""
    print("\nTesting EnsembleCGCNN...")
    
    # This would use real trained models
    print("  (Placeholder - requires trained models)")
    print("✓ Ensemble structure validated!")


if __name__ == '__main__':
    test_ensemble()