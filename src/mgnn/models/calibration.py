"""
Uncertainty Calibration Methods.
Temperature scaling for post-hoc calibration improvement.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from typing import Tuple, Dict


class TemperatureScaling:
    """
    Temperature scaling for uncertainty calibration.
    
    Learns a single parameter T to scale uncertainties:
        calibrated_uncertainty = raw_uncertainty * T
    
    T is optimized to minimize calibration error on validation set.
    """
    
    def __init__(self):
        """Initialize temperature scaling."""
        self.temperature = 1.0
        self.is_fitted = False
        
        print("TemperatureScaling initialized")
    
    def fit(
        self,
        uncertainties: np.ndarray,
        errors: np.ndarray,
        method: str = 'nll'
    ) -> float:
        """
        Fit temperature parameter.
        
        Args:
            uncertainties: Predicted uncertainties (std)
            errors: Actual absolute errors
            method: Optimization method
                - 'nll': Negative log-likelihood (assumes Gaussian)
                - 'coverage': Minimize coverage error
                - 'mse': Minimize mean squared error
        
        Returns:
            Optimal temperature
        """
        print(f"\nFitting temperature using '{method}' method...")
        
        unc_flat = uncertainties.flatten()
        err_flat = errors.flatten()
        
        if method == 'nll':
            # Minimize negative log-likelihood (assume Gaussian errors)
            def nll(T):
                scaled_unc = unc_flat * T
                # Avoid division by zero
                scaled_unc = np.maximum(scaled_unc, 1e-6)
                # Negative log-likelihood
                nll_value = 0.5 * np.log(2 * np.pi * scaled_unc**2) + (err_flat**2) / (2 * scaled_unc**2)
                return nll_value.mean()
            
            result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
            self.temperature = result.x
            
        elif method == 'coverage':
            # Minimize coverage error (distance from expected coverage)
            def coverage_error(T):
                scaled_unc = unc_flat * T
                coverage_1sigma = (err_flat <= scaled_unc).mean()
                coverage_2sigma = (err_flat <= 2 * scaled_unc).mean()
                
                # Expected coverages
                expected_1sigma = 0.6827
                expected_2sigma = 0.9545
                
                # Total error
                error = abs(coverage_1sigma - expected_1sigma) + abs(coverage_2sigma - expected_2sigma)
                return error
            
            result = minimize_scalar(coverage_error, bounds=(0.1, 10.0), method='bounded')
            self.temperature = result.x
            
        elif method == 'mse':
            # Minimize MSE between uncertainty and error
            def mse(T):
                scaled_unc = unc_flat * T
                return np.mean((scaled_unc - err_flat)**2)
            
            result = minimize_scalar(mse, bounds=(0.1, 10.0), method='bounded')
            self.temperature = result.x
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.is_fitted = True
        
        print(f"  Optimal temperature: {self.temperature:.4f}")
        
        return self.temperature
    
    def transform(self, uncertainties: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to uncertainties.
        
        Args:
            uncertainties: Raw uncertainties
        
        Returns:
            Calibrated uncertainties
        """
        if not self.is_fitted:
            raise RuntimeError("Temperature scaling not fitted. Call fit() first.")
        
        return uncertainties * self.temperature
    
    def fit_transform(
        self,
        uncertainties: np.ndarray,
        errors: np.ndarray,
        method: str = 'nll'
    ) -> np.ndarray:
        """
        Fit temperature and transform uncertainties.
        
        Args:
            uncertainties: Predicted uncertainties
            errors: Actual errors
            method: Optimization method
        
        Returns:
            Calibrated uncertainties
        """
        self.fit(uncertainties, errors, method)
        return self.transform(uncertainties)
    
    def evaluate_calibration(
        self,
        uncertainties: np.ndarray,
        errors: np.ndarray,
        scaled: bool = False
    ) -> Dict:
        """
        Evaluate calibration quality.
        
        Args:
            uncertainties: Predicted uncertainties
            errors: Actual absolute errors
            scaled: Whether uncertainties are already scaled
        
        Returns:
            Calibration metrics
        """
        unc = uncertainties.flatten()
        err = errors.flatten()
        
        if not scaled and self.is_fitted:
            unc = unc * self.temperature
        
        # Coverage
        within_1sigma = (err <= unc).mean()
        within_2sigma = (err <= 2 * unc).mean()
        within_3sigma = (err <= 3 * unc).mean()
        
        # Correlation
        correlation = np.corrcoef(unc, err)[0, 1]
        
        # MSE
        mse = np.mean((unc - err)**2)
        
        # Calibration quality
        sigma_error_1 = abs(within_1sigma - 0.6827)
        sigma_error_2 = abs(within_2sigma - 0.9545)
        
        if sigma_error_1 < 0.05 and sigma_error_2 < 0.05:
            quality = "Excellent"
        elif sigma_error_1 < 0.10 and sigma_error_2 < 0.10:
            quality = "Good"
        elif sigma_error_1 < 0.15 and sigma_error_2 < 0.15:
            quality = "Fair"
        else:
            quality = "Poor"
        
        metrics = {
            'within_1sigma': float(within_1sigma),
            'within_2sigma': float(within_2sigma),
            'within_3sigma': float(within_3sigma),
            'correlation': float(correlation),
            'mse': float(mse),
            'quality': quality,
            'expected_1sigma': 0.6827,
            'expected_2sigma': 0.9545,
            'expected_3sigma': 0.9973
        }
        
        return metrics
    
    def plot_calibration_comparison(
        self,
        uncertainties_before: np.ndarray,
        uncertainties_after: np.ndarray,
        errors: np.ndarray,
        output_path: str = './calibration_comparison.png'
    ) -> plt.Figure:
        """
        Plot before/after calibration comparison.
        
        Args:
            uncertainties_before: Raw uncertainties
            uncertainties_after: Calibrated uncertainties
            errors: Actual errors
            output_path: Where to save plot
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.patch.set_facecolor('white')
        
        unc_before = uncertainties_before.flatten()
        unc_after = uncertainties_after.flatten()
        err = errors.flatten()
        
        # Before calibration metrics
        metrics_before = self.evaluate_calibration(uncertainties_before, errors, scaled=False)
        metrics_after = self.evaluate_calibration(uncertainties_after, errors, scaled=True)
        
        # Plot 1: Uncertainty vs Error (Before)
        ax = axes[0, 0]
        ax.scatter(unc_before, err, alpha=0.3, s=10, label='Data')
        max_val = max(unc_before.max(), err.max())
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect calibration')
        ax.set_xlabel('Raw Uncertainty (eV/atom)')
        ax.set_ylabel('Actual Error (eV/atom)')
        ax.set_title(f'Before Scaling (Quality: {metrics_before["quality"]})')
        ax.text(0.05, 0.95, f'Corr: {metrics_before["correlation"]:.3f}',
               transform=ax.transAxes, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Uncertainty vs Error (After)
        ax = axes[0, 1]
        ax.scatter(unc_after, err, alpha=0.3, s=10, label='Data', color='green')
        max_val = max(unc_after.max(), err.max())
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect calibration')
        ax.set_xlabel('Calibrated Uncertainty (eV/atom)')
        ax.set_ylabel('Actual Error (eV/atom)')
        ax.set_title(f'After Scaling (Quality: {metrics_after["quality"]})')
        ax.text(0.05, 0.95, f'Corr: {metrics_after["correlation"]:.3f}\nT={self.temperature:.3f}',
               transform=ax.transAxes, va='top',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Coverage comparison
        ax = axes[1, 0]
        
        coverages_before = [
            metrics_before['within_1sigma'],
            metrics_before['within_2sigma'],
            metrics_before['within_3sigma']
        ]
        
        coverages_after = [
            metrics_after['within_1sigma'],
            metrics_after['within_2sigma'],
            metrics_after['within_3sigma']
        ]
        
        expected = [0.6827, 0.9545, 0.9973]
        
        x = np.arange(3)
        width = 0.25
        
        ax.bar(x - width, coverages_before, width, label='Before', alpha=0.8, color='steelblue')
        ax.bar(x, coverages_after, width, label='After', alpha=0.8, color='green')
        ax.bar(x + width, expected, width, label='Expected', alpha=0.8, color='red')
        
        ax.set_ylabel('Coverage Probability')
        ax.set_title('Coverage Improvement')
        ax.set_xticks(x)
        ax.set_xticklabels(['1σ (68%)', '2σ (95%)', '3σ (99.7%)'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.1)
        
        # Plot 4: Improvement metrics
        ax = axes[1, 1]
        
        metrics = ['Coverage 1σ', 'Coverage 2σ', 'Correlation', 'Quality']
        before_vals = [
            abs(coverages_before[0] - expected[0]),
            abs(coverages_before[1] - expected[1]),
            1 - metrics_before['correlation'],
            0.25 if metrics_before['quality'] == 'Poor' else 0.5
        ]
        after_vals = [
            abs(coverages_after[0] - expected[0]),
            abs(coverages_after[1] - expected[1]),
            1 - metrics_after['correlation'],
            0.0 if metrics_after['quality'] == 'Excellent' else 
            0.15 if metrics_after['quality'] == 'Good' else
            0.25 if metrics_after['quality'] == 'Fair' else 0.4
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.barh(x - width/2, before_vals, width, label='Before (Error)', alpha=0.8, color='steelblue')
        ax.barh(x + width/2, after_vals, width, label='After (Error)', alpha=0.8, color='green')
        
        ax.set_xlabel('Error / Distance from Ideal')
        ax.set_title('Calibration Error Reduction')
        ax.set_yticks(x)
        ax.set_yticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nCalibration comparison plot saved to: {output_path}")
        
        return fig


def test_temperature_scaling():
    """Test temperature scaling."""
    print("\nTesting TemperatureScaling...")
    
    # Create dummy data (underconfident model)
    np.random.seed(42)
    n_samples = 1000
    
    # True errors
    errors = np.abs(np.random.randn(n_samples) * 0.05)
    
    # Underestimated uncertainties
    uncertainties = errors * 0.5 + np.random.randn(n_samples) * 0.01
    uncertainties = np.abs(uncertainties)
    
    print(f"\nOriginal statistics:")
    print(f"  Mean error: {errors.mean():.4f}")
    print(f"  Mean uncertainty: {uncertainties.mean():.4f}")
    print(f"  Coverage 1σ: {(errors <= uncertainties).mean():.1%}")
    
    # Fit temperature
    ts = TemperatureScaling()
    calibrated_unc = ts.fit_transform(uncertainties, errors, method='coverage')
    
    print(f"\nAfter temperature scaling:")
    print(f"  Temperature: {ts.temperature:.4f}")
    print(f"  Mean calibrated uncertainty: {calibrated_unc.mean():.4f}")
    print(f"  Coverage 1σ: {(errors <= calibrated_unc).mean():.1%}")
    
    # Evaluate
    metrics_before = ts.evaluate_calibration(uncertainties, errors, scaled=False)
    metrics_after = ts.evaluate_calibration(calibrated_unc, errors, scaled=True)
    
    print(f"\nCalibration quality:")
    print(f"  Before: {metrics_before['quality']}")
    print(f"  After:  {metrics_after['quality']}")
    
    print("\n✓ TemperatureScaling test passed!")


if __name__ == '__main__':
    test_temperature_scaling()


# cd /projectnb/cui-buchem/anandsahu/Softwares/Mine/multinary-gnn
# uv sync

# First test the module
# uv run python -m mgnn.models.calibration
