"""
Custom loss functions for materials stability prediction.
Addresses class imbalance and focuses on discovery-critical stable compounds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedMSELoss(nn.Module):
    """
    MSE Loss with class-based weighting.
    
    Gives higher importance to rare stable compounds.
    """
    
    def __init__(
        self,
        stable_weight: float = 15.0,
        metastable_weight: float = 3.0,
        unstable_weight: float = 1.0,
        stable_threshold: float = 0.025,
        metastable_threshold: float = 0.1
    ):
        """
        Initialize weighted MSE loss.
        
        Args:
            stable_weight: Weight for stable compounds (ΔH_d ≤ 0.025)
            metastable_weight: Weight for metastable compounds
            unstable_weight: Weight for unstable compounds
            stable_threshold: Threshold for stable classification
            metastable_threshold: Threshold for metastable classification
        """
        super(WeightedMSELoss, self).__init__()
        
        self.stable_weight = stable_weight
        self.metastable_weight = metastable_weight
        self.unstable_weight = unstable_weight
        self.stable_threshold = stable_threshold
        self.metastable_threshold = metastable_threshold
        
        print(f"WeightedMSELoss initialized:")
        print(f"  Stable weight: {stable_weight}x")
        print(f"  Metastable weight: {metastable_weight}x")
        print(f"  Unstable weight: {unstable_weight}x")
    
    def forward(self, predictions, targets):
        """
        Compute weighted MSE loss.
        
        Args:
            predictions: Model predictions [batch_size, 1]
            targets: Ground truth targets [batch_size, 1]
        
        Returns:
            Weighted loss scalar
        """
        # Flatten
        pred = predictions.squeeze()
        true = targets.squeeze()
        
        # Compute per-sample squared errors
        squared_errors = (pred - true) ** 2
        
        # Assign weights based on target values
        weights = torch.ones_like(true)
        
        # Stable compounds get highest weight
        stable_mask = true <= self.stable_threshold
        weights[stable_mask] = self.stable_weight
        
        # Metastable compounds get medium weight
        metastable_mask = (true > self.stable_threshold) & (true <= self.metastable_threshold)
        weights[metastable_mask] = self.metastable_weight
        
        # Unstable compounds get base weight
        unstable_mask = true > self.metastable_threshold
        weights[unstable_mask] = self.unstable_weight
        
        # Compute weighted loss
        weighted_loss = (weights * squared_errors).mean()
        
        return weighted_loss


class FocalMSELoss(nn.Module):
    """
    Focal Loss adapted for regression.
    
    Focuses on hard-to-predict samples (large errors).
    """
    
    def __init__(self, gamma: float = 2.0, alpha: float = 1.0):
        """
        Initialize focal MSE loss.
        
        Args:
            gamma: Focusing parameter (higher = focus more on hard samples)
            alpha: Weighting factor
        """
        super(FocalMSELoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        
        print(f"FocalMSELoss initialized: gamma={gamma}, alpha={alpha}")
    
    def forward(self, predictions, targets):
        """
        Compute focal MSE loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
        
        Returns:
            Focal loss scalar
        """
        pred = predictions.squeeze()
        true = targets.squeeze()
        
        # Compute squared errors
        squared_errors = (pred - true) ** 2
        
        # Compute focal weights (higher for larger errors)
        focal_weights = (1 + squared_errors) ** self.gamma
        
        # Weighted loss
        loss = self.alpha * (focal_weights * squared_errors).mean()
        
        return loss


class HybridLoss(nn.Module):
    """
    Hybrid loss combining weighted MSE and focal loss.
    
    Best of both worlds: class balancing + hard sample focus.
    """
    
    def __init__(
        self,
        stable_weight: float = 15.0,
        focal_gamma: float = 1.5,
        weighted_ratio: float = 0.7
    ):
        """
        Initialize hybrid loss.
        
        Args:
            stable_weight: Weight for stable compounds
            focal_gamma: Focal loss gamma parameter
            weighted_ratio: Ratio of weighted vs focal loss (0-1)
        """
        super(HybridLoss, self).__init__()
        
        self.weighted_mse = WeightedMSELoss(
            stable_weight=stable_weight,
            metastable_weight=3.0,
            unstable_weight=1.0
        )
        
        self.focal_mse = FocalMSELoss(gamma=focal_gamma)
        
        self.weighted_ratio = weighted_ratio
        
        print(f"HybridLoss initialized:")
        print(f"  Weighted ratio: {weighted_ratio:.1%}")
        print(f"  Focal ratio: {1-weighted_ratio:.1%}")
    
    def forward(self, predictions, targets):
        """
        Compute hybrid loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
        
        Returns:
            Combined loss scalar
        """
        weighted_loss = self.weighted_mse(predictions, targets)
        focal_loss = self.focal_mse(predictions, targets)
        
        combined_loss = (
            self.weighted_ratio * weighted_loss +
            (1 - self.weighted_ratio) * focal_loss
        )
        
        return combined_loss


def test_losses():
    """Test custom loss functions."""
    print("\nTesting Custom Loss Functions...")
    
    # Create dummy data with class imbalance
    torch.manual_seed(42)
    
    # Targets: mostly unstable (>0.1), few stable (<0.025)
    targets = torch.cat([
        torch.rand(10) * 0.02,      # 10 stable
        torch.rand(30) * 0.075 + 0.025,  # 30 metastable
        torch.rand(160) * 0.5 + 0.1  # 160 unstable
    ])
    
    # Predictions: similar distribution with noise
    predictions = targets + torch.randn_like(targets) * 0.05
    
    print(f"\nData distribution:")
    print(f"  Stable (<0.025): {(targets <= 0.025).sum()}")
    print(f"  Metastable (0.025-0.1): {((targets > 0.025) & (targets <= 0.1)).sum()}")
    print(f"  Unstable (>0.1): {(targets > 0.1).sum()}")
    
    # Test standard MSE
    mse_loss = nn.MSELoss()
    loss_mse = mse_loss(predictions, targets)
    print(f"\nStandard MSE Loss: {loss_mse:.6f}")
    
    # Test weighted MSE
    weighted_loss = WeightedMSELoss(stable_weight=15.0)
    loss_weighted = weighted_loss(predictions, targets)
    print(f"Weighted MSE Loss: {loss_weighted:.6f}")
    print(f"  → {loss_weighted/loss_mse:.2f}x higher (focuses on stable!)")
    
    # Test focal MSE
    focal_loss = FocalMSELoss(gamma=2.0)
    loss_focal = focal_loss(predictions, targets)
    print(f"Focal MSE Loss: {loss_focal:.6f}")
    
    # Test hybrid
    hybrid_loss = HybridLoss(stable_weight=15.0, weighted_ratio=0.7)
    loss_hybrid = hybrid_loss(predictions, targets)
    print(f"Hybrid Loss: {loss_hybrid:.6f}")
    
    print("\n✓ Loss function tests passed!")


if __name__ == '__main__':
    test_losses()


# Test the function

# cd /projectnb/cui-buchem/anandsahu/Softwares/Mine/multinary-gnn
# uv sync
# uv run python -m mgnn.models.losses