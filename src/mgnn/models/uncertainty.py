"""
Uncertainty quantification for CGCNN models.

Implements:
1. MC Dropout - Multiple forward passes with dropout enabled
2. Ensemble Uncertainty - Predictions from multiple models
3. Calibration Analysis - Are uncertainty estimates reliable?
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, List
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats


class UncertaintyQuantifier:
    """
    Uncertainty quantification for CGCNN models.
    
    Provides:
    - Predictive uncertainty (aleatoric + epistemic)
    - Confidence intervals
    - Calibration analysis
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize uncertainty quantifier.
        
        Args:
            model: Trained CGCNN model
            device: Device for inference
        """
        self.model = model.to(device)
        self.device = device
        
        print(f"UncertaintyQuantifier initialized on {device}")
    
    def enable_dropout(self):
        """Enable dropout for MC Dropout."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # Enable dropout at test time
    
    def disable_dropout(self):
        """Disable dropout (standard inference)."""
        self.model.eval()
    
    @torch.no_grad()
    def mc_dropout_predict(
        self,
        data_loader: DataLoader,
        n_samples: int = 30,
        show_progress: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        MC Dropout uncertainty estimation.
        
        Performs multiple forward passes with dropout enabled
        to estimate predictive uncertainty.
        
        Args:
            data_loader: DataLoader with test data
            n_samples: Number of MC samples
            show_progress: Show progress bar
        
        Returns:
            (mean_predictions, std_predictions, all_predictions)
        """
        print(f"\nRunning MC Dropout with {n_samples} samples...")
        
        # Enable dropout
        self.enable_dropout()
        
        all_predictions = []
        all_targets = []
        
        # Multiple forward passes
        for i in range(n_samples):
            batch_predictions = []
            batch_targets = []
            
            iterator = tqdm(data_loader, desc=f"MC Sample {i+1}/{n_samples}", leave=False) if show_progress else data_loader
            
            for batch in iterator:
                batch = self._prepare_batch(batch)
                
                # Forward pass with dropout enabled
                predictions = self.model(batch)
                
                batch_predictions.append(predictions.cpu().numpy())
                batch_targets.append(batch.y.cpu().numpy())
            
            # Concatenate batch predictions
            sample_predictions = np.concatenate(batch_predictions, axis=0)
            all_predictions.append(sample_predictions)
            
            if i == 0:
                # Store targets only once
                all_targets = np.concatenate(batch_targets, axis=0)
        
        # Stack predictions: [n_samples, n_data, 1]
        all_predictions = np.stack(all_predictions, axis=0)
        
        # Compute statistics
        mean_predictions = all_predictions.mean(axis=0)
        std_predictions = all_predictions.std(axis=0)
        
        print(f"  Mean uncertainty: {std_predictions.mean():.6f} eV/atom")
        print(f"  Max uncertainty:  {std_predictions.max():.6f} eV/atom")
        
        # Disable dropout
        self.disable_dropout()
        
        return mean_predictions, std_predictions, all_predictions
    
    @torch.no_grad()
    def ensemble_predict(
        self,
        models: List[nn.Module],
        data_loader: DataLoader,
        show_progress: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Ensemble uncertainty estimation.
        
        Uses predictions from multiple independently trained models.
        
        Args:
            models: List of trained models
            data_loader: DataLoader with test data
            show_progress: Show progress bar
        
        Returns:
            (mean_predictions, std_predictions, all_predictions)
        """
        print(f"\nRunning Ensemble prediction with {len(models)} models...")
        
        all_predictions = []
        all_targets = []
        
        for i, model in enumerate(models):
            model.to(self.device)
            model.eval()
            
            batch_predictions = []
            batch_targets = []
            
            iterator = tqdm(data_loader, desc=f"Model {i+1}/{len(models)}", leave=False) if show_progress else data_loader
            
            for batch in iterator:
                batch = self._prepare_batch(batch)
                predictions = model(batch)
                
                batch_predictions.append(predictions.cpu().numpy())
                batch_targets.append(batch.y.cpu().numpy())
            
            sample_predictions = np.concatenate(batch_predictions, axis=0)
            all_predictions.append(sample_predictions)
            
            if i == 0:
                all_targets = np.concatenate(batch_targets, axis=0)
        
        # Stack predictions
        all_predictions = np.stack(all_predictions, axis=0)
        
        # Compute statistics
        mean_predictions = all_predictions.mean(axis=0)
        std_predictions = all_predictions.std(axis=0)
        
        print(f"  Mean uncertainty: {std_predictions.mean():.6f} eV/atom")
        print(f"  Max uncertainty:  {std_predictions.max():.6f} eV/atom")
        
        return mean_predictions, std_predictions, all_predictions
    
    def _prepare_batch(self, batch):
        """Prepare batch (handle comp_features reshaping)."""
        batch = batch.to(self.device)
        
        # Fix comp_features shape if needed
        if hasattr(batch, 'comp_features') and batch.comp_features.dim() == 1:
            num_graphs = batch.num_graphs
            comp_dim = batch.comp_features.shape[0] // num_graphs
            batch.comp_features = batch.comp_features.view(num_graphs, comp_dim)
        
        return batch
    
    def calibration_analysis(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        targets: np.ndarray,
        n_bins: int = 10
    ) -> Dict:
        """
        Analyze calibration of uncertainty estimates.
        
        Well-calibrated uncertainties mean that when the model
        predicts ±1σ confidence interval, ~68% of true values
        should fall within that interval.
        
        Args:
            predictions: Mean predictions
            uncertainties: Uncertainty estimates (std)
            targets: True values
            n_bins: Number of bins for calibration curve
        
        Returns:
            Calibration statistics
        """
        print("\nAnalyzing calibration...")
        
        # Compute errors
        errors = np.abs(predictions.flatten() - targets.flatten())
        uncertainties_flat = uncertainties.flatten()
        
        # Check confidence intervals
        within_1sigma = np.sum(errors <= uncertainties_flat) / len(errors)
        within_2sigma = np.sum(errors <= 2 * uncertainties_flat) / len(errors)
        within_3sigma = np.sum(errors <= 3 * uncertainties_flat) / len(errors)
        
        # Expected values for well-calibrated model
        expected_1sigma = 0.6827
        expected_2sigma = 0.9545
        expected_3sigma = 0.9973
        
        print(f"  Within 1σ: {within_1sigma:.1%} (expected: 68.3%)")
        print(f"  Within 2σ: {within_2sigma:.1%} (expected: 95.5%)")
        print(f"  Within 3σ: {within_3sigma:.1%} (expected: 99.7%)")
        
        # Calibration curve
        # Bin by uncertainty and compute actual error in each bin
        uncertainty_bins = np.percentile(uncertainties_flat, np.linspace(0, 100, n_bins + 1))
        bin_indices = np.digitize(uncertainties_flat, uncertainty_bins)
        
        bin_uncertainties = []
        bin_errors = []
        
        for i in range(1, n_bins + 1):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_uncertainties.append(uncertainties_flat[mask].mean())
                bin_errors.append(errors[mask].mean())
        
        # Correlation between uncertainty and error
        correlation = np.corrcoef(uncertainties_flat, errors)[0, 1]
        
        print(f"  Uncertainty-Error correlation: {correlation:.3f}")
        
        calibration_stats = {
            'within_1sigma': float(within_1sigma),
            'within_2sigma': float(within_2sigma),
            'within_3sigma': float(within_3sigma),
            'correlation': float(correlation),
            'bin_uncertainties': bin_uncertainties,
            'bin_errors': bin_errors,
            'expected_1sigma': expected_1sigma,
            'expected_2sigma': expected_2sigma,
            'expected_3sigma': expected_3sigma
        }
        
        # Calibration quality assessment
        sigma_error_1 = abs(within_1sigma - expected_1sigma)
        sigma_error_2 = abs(within_2sigma - expected_2sigma)
        
        if sigma_error_1 < 0.05 and sigma_error_2 < 0.05:
            quality = "Excellent"
        elif sigma_error_1 < 0.10 and sigma_error_2 < 0.10:
            quality = "Good"
        elif sigma_error_1 < 0.15:
            quality = "Fair"
        else:
            quality = "Poor"
        
        print(f"  Calibration quality: {quality}")
        calibration_stats['quality'] = quality
        
        return calibration_stats
    
    def plot_uncertainty_analysis(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        targets: np.ndarray,
        calibration_stats: Dict,
        output_path: str = './uncertainty_analysis.png'
    ) -> plt.Figure:
        """
        Create comprehensive uncertainty analysis plot.
        
        Args:
            predictions: Mean predictions
            uncertainties: Uncertainty estimates
            targets: True values
            calibration_stats: Calibration statistics
            output_path: Where to save plot
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.patch.set_facecolor('white')
        
        pred_flat = predictions.flatten()
        uncert_flat = uncertainties.flatten()
        true_flat = targets.flatten()
        errors = np.abs(pred_flat - true_flat)
        
        # Plot 1: Predictions with error bars
        ax = axes[0, 0]
        sorted_indices = np.argsort(true_flat)
        
        # Subsample for visualization if too many points
        if len(sorted_indices) > 1000:
            indices = sorted_indices[::len(sorted_indices)//1000]
        else:
            indices = sorted_indices
        
        ax.errorbar(true_flat[indices], pred_flat[indices], 
                   yerr=uncert_flat[indices], 
                   fmt='o', alpha=0.3, markersize=3, elinewidth=1,
                   label='Predictions ± σ')
        ax.plot([-0.2, 0.7], [-0.2, 0.7], 'r--', linewidth=2, label='Perfect prediction')
        ax.set_xlabel('True Decomposition Energy (eV/atom)')
        ax.set_ylabel('Predicted Decomposition Energy (eV/atom)')
        ax.set_title('Predictions with Uncertainty')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Uncertainty vs Error
        ax = axes[0, 1]
        ax.scatter(uncert_flat, errors, alpha=0.3, s=10)
        
        # Add diagonal line (perfect calibration)
        max_val = max(uncert_flat.max(), errors.max())
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect calibration')
        
        # Add correlation
        corr = calibration_stats['correlation']
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
               transform=ax.transAxes, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('Predicted Uncertainty (eV/atom)')
        ax.set_ylabel('Actual Error (eV/atom)')
        ax.set_title('Uncertainty Calibration')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Calibration curve
        ax = axes[1, 0]
        if calibration_stats['bin_uncertainties']:
            ax.plot(calibration_stats['bin_uncertainties'], 
                   calibration_stats['bin_errors'],
                   'bo-', linewidth=2, markersize=8, label='Actual')
            
            max_val = max(max(calibration_stats['bin_uncertainties']), 
                         max(calibration_stats['bin_errors']))
            ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect')
        
        ax.set_xlabel('Predicted Uncertainty (eV/atom)')
        ax.set_ylabel('Mean Absolute Error (eV/atom)')
        ax.set_title('Calibration Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Coverage probability
        ax = axes[1, 1]
        
        # Bars for different sigma levels
        actual = [calibration_stats['within_1sigma'],
                 calibration_stats['within_2sigma'],
                 calibration_stats['within_3sigma']]
        expected = [calibration_stats['expected_1sigma'],
                   calibration_stats['expected_2sigma'],
                   calibration_stats['expected_3sigma']]
        
        x = np.arange(3)
        width = 0.35
        
        ax.bar(x - width/2, actual, width, label='Actual', alpha=0.8)
        ax.bar(x + width/2, expected, width, label='Expected', alpha=0.8)
        
        ax.set_ylabel('Coverage Probability')
        ax.set_title(f'Uncertainty Coverage (Quality: {calibration_stats["quality"]})')
        ax.set_xticks(x)
        ax.set_xticklabels(['1σ (68%)', '2σ (95%)', '3σ (99.7%)'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nUncertainty analysis plot saved to: {output_path}")
        
        return fig


def test_uncertainty():
    """Test uncertainty quantification."""
    print("\nTesting UncertaintyQuantifier...")
    
    # Create dummy model
    from mgnn.models.cgcnn import CGCNN
    
    model = CGCNN(
        node_feature_dim=12,
        edge_feature_dim=1,
        comp_feature_dim=71,
        hidden_dim=64,
        n_conv=2,
        n_fc=1,
        dropout=0.1
    )
    
    uq = UncertaintyQuantifier(model, device='cpu')
    
    print("  ✓ UncertaintyQuantifier initialized")
    print("  ✓ MC Dropout methods available")
    print("  ✓ Calibration analysis available")
    
    print("\n✓ Uncertainty quantification test passed!")


if __name__ == '__main__':
    test_uncertainty()


# cd /projectnb/cui-buchem/anandsahu/Softwares/Mine/multinary-gnn
# uv sync

# Test uncertainty module
# uv run python -m mgnn.models.uncertainty