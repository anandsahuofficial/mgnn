"""
Comprehensive text-based results summary generator.
Perfect for papers, comparisons, and quick analysis.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class ResultsSummaryGenerator:
    """
    Generate comprehensive text-based results summaries.
    
    Produces markdown and plain text reports with all metrics,
    statistics, and analysis suitable for papers and documentation.
    """
    
    def __init__(self, output_dir: str = './results'):
        """
        Initialize results summary generator.
        
        Args:
            output_dir: Directory to save summary files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_training_summary(
        self,
        model_name: str,
        model_params: int,
        history: Dict,
        test_results: Dict,
        config: Dict,
        train_time_minutes: float,
        predictions: Dict[str, Tuple[np.ndarray, np.ndarray]] = None
    ) -> str:
        """
        Generate comprehensive training summary.
        
        Args:
            model_name: Name of the model
            model_params: Number of model parameters
            history: Training history dictionary
            test_results: Test evaluation results
            config: Model configuration
            train_time_minutes: Total training time in minutes
            predictions: Dict of {'split': (predictions, targets)} for all splits
        
        Returns:
            Summary text
        """
        lines = []
        
        # Header
        lines.append("="*80)
        lines.append("MULTINARY GNN (MGNN) - TRAINING RESULTS SUMMARY")
        lines.append("="*80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Model: {model_name}")
        lines.append("")
        
        # Model Architecture
        lines.append("-"*80)
        lines.append("MODEL ARCHITECTURE")
        lines.append("-"*80)
        lines.append(f"Total Parameters: {model_params:,}")
        lines.append(f"Node Features: {config.get('NODE_FEATURE_DIM', 'N/A')}")
        lines.append(f"Edge Features: {config.get('EDGE_FEATURE_DIM', 'N/A')}")
        lines.append(f"Composition Features: {config.get('COMP_FEATURE_DIM', 'N/A')}")
        lines.append(f"Hidden Dimension: {config.get('HIDDEN_DIM', 'N/A')}")
        lines.append(f"Conv Layers: {config.get('N_CONV_LAYERS', 'N/A')}")
        lines.append(f"FC Layers: {config.get('N_FC_LAYERS', 'N/A')}")
        lines.append(f"Dropout: {config.get('DROPOUT', 'N/A')}")
        lines.append("")
        
        # Training Configuration
        lines.append("-"*80)
        lines.append("TRAINING CONFIGURATION")
        lines.append("-"*80)
        lines.append(f"Batch Size: {config.get('BATCH_SIZE', 'N/A')}")
        lines.append(f"Initial Learning Rate: {config.get('LEARNING_RATE', 'N/A')}")
        lines.append(f"Weight Decay: {config.get('WEIGHT_DECAY', 'N/A')}")
        lines.append(f"Max Epochs: {config.get('N_EPOCHS', 'N/A')}")
        lines.append(f"Early Stopping Patience: {config.get('EARLY_STOPPING_PATIENCE', 'N/A')}")
        lines.append(f"Device: {config.get('DEVICE', 'N/A')}")
        lines.append("")
        
        # Training Progress
        lines.append("-"*80)
        lines.append("TRAINING PROGRESS")
        lines.append("-"*80)
        n_epochs = len(history['train_loss'])
        best_epoch = np.argmin(history['val_loss']) + 1
        
        lines.append(f"Epochs Completed: {n_epochs}")
        lines.append(f"Best Epoch: {best_epoch}")
        lines.append(f"Training Time: {train_time_minutes:.1f} minutes ({train_time_minutes/60:.1f} hours)")
        lines.append(f"Time per Epoch: {train_time_minutes/n_epochs:.1f} minutes")
        lines.append("")
        
        lines.append("Training Metrics:")
        lines.append(f"  Initial Train Loss: {history['train_loss'][0]:.6f}")
        lines.append(f"  Final Train Loss:   {history['train_loss'][-1]:.6f}")
        lines.append(f"  Initial Train MAE:  {history['train_mae'][0]:.6f} eV/atom")
        lines.append(f"  Final Train MAE:    {history['train_mae'][-1]:.6f} eV/atom")
        lines.append("")
        
        lines.append("Validation Metrics:")
        lines.append(f"  Initial Val Loss: {history['val_loss'][0]:.6f}")
        lines.append(f"  Best Val Loss:    {history['val_loss'][best_epoch-1]:.6f} (epoch {best_epoch})")
        lines.append(f"  Final Val Loss:   {history['val_loss'][-1]:.6f}")
        lines.append(f"  Initial Val MAE:  {history['val_mae'][0]:.6f} eV/atom")
        lines.append(f"  Best Val MAE:     {history['val_mae'][best_epoch-1]:.6f} eV/atom (epoch {best_epoch})")
        lines.append(f"  Final Val MAE:    {history['val_mae'][-1]:.6f} eV/atom")
        lines.append("")
        
        lines.append("Learning Rate Schedule:")
        lines.append(f"  Initial LR: {history['learning_rate'][0]:.6f}")
        lines.append(f"  Final LR:   {history['learning_rate'][-1]:.6f}")
        lines.append(f"  Min LR:     {min(history['learning_rate']):.6f}")
        lines.append("")
        
        # Test Results
        lines.append("-"*80)
        lines.append("TEST SET PERFORMANCE")
        lines.append("-"*80)
        lines.append(f"Test Loss (MSE): {test_results['test_loss']:.6f}")
        lines.append(f"Test MAE:        {test_results['mae']:.6f} eV/atom")
        lines.append(f"Test RMSE:       {test_results['rmse']:.6f} eV/atom")
        lines.append(f"Test R²:         {test_results['r2']:.6f}")
        lines.append("")
        
        # Detailed predictions analysis if provided
        if predictions:
            lines.append("-"*80)
            lines.append("DETAILED PREDICTION ANALYSIS")
            lines.append("-"*80)
            
            for split_name, (pred, true) in predictions.items():
                lines.append(f"\n{split_name.upper()} SET:")
                lines.append(f"  Samples: {len(pred):,}")
                
                # Basic statistics
                errors = pred.flatten() - true.flatten()
                abs_errors = np.abs(errors)
                
                lines.append(f"  Mean Error:        {np.mean(errors):.6f} eV/atom")
                lines.append(f"  Std Error:         {np.std(errors):.6f} eV/atom")
                lines.append(f"  Mean Abs Error:    {np.mean(abs_errors):.6f} eV/atom")
                lines.append(f"  Median Abs Error:  {np.median(abs_errors):.6f} eV/atom")
                lines.append(f"  Max Abs Error:     {np.max(abs_errors):.6f} eV/atom")
                lines.append(f"  Min Abs Error:     {np.min(abs_errors):.6f} eV/atom")
                
                # Percentiles
                lines.append(f"\n  Error Percentiles:")
                for p in [10, 25, 50, 75, 90, 95, 99]:
                    val = np.percentile(abs_errors, p)
                    lines.append(f"    {p}th:  {val:.6f} eV/atom")
                
                # Prediction quality by stability class
                lines.append(f"\n  Performance by Stability Class:")
                
                stable_mask = true.flatten() <= 0.025
                metastable_mask = (true.flatten() > 0.025) & (true.flatten() <= 0.1)
                unstable_mask = true.flatten() > 0.1
                
                for class_name, mask in [
                    ('Stable (≤0.025 eV/atom)', stable_mask),
                    ('Metastable (0.025-0.1 eV/atom)', metastable_mask),
                    ('Unstable (>0.1 eV/atom)', unstable_mask)
                ]:
                    if mask.sum() > 0:
                        class_mae = np.mean(np.abs(pred.flatten()[mask] - true.flatten()[mask]))
                        class_rmse = np.sqrt(np.mean((pred.flatten()[mask] - true.flatten()[mask])**2))
                        lines.append(f"    {class_name}:")
                        lines.append(f"      Count: {mask.sum():,}")
                        lines.append(f"      MAE:   {class_mae:.6f} eV/atom")
                        lines.append(f"      RMSE:  {class_rmse:.6f} eV/atom")
                
                # Prediction range statistics
                lines.append(f"\n  Prediction Statistics:")
                lines.append(f"    True range:  [{true.min():.4f}, {true.max():.4f}] eV/atom")
                lines.append(f"    Pred range:  [{pred.min():.4f}, {pred.max():.4f}] eV/atom")
                lines.append(f"    True mean:   {true.mean():.4f} ± {true.std():.4f} eV/atom")
                lines.append(f"    Pred mean:   {pred.mean():.4f} ± {pred.std():.4f} eV/atom")
                
                # Classification accuracy (using 0.025 eV/atom threshold)
                true_stable = true.flatten() <= 0.025
                pred_stable = pred.flatten() <= 0.025
                
                tp = np.sum(true_stable & pred_stable)
                tn = np.sum(~true_stable & ~pred_stable)
                fp = np.sum(~true_stable & pred_stable)
                fn = np.sum(true_stable & ~pred_stable)
                
                accuracy = (tp + tn) / len(true_stable)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                lines.append(f"\n  Binary Classification (Stable vs Unstable, threshold=0.025):")
                lines.append(f"    Accuracy:  {accuracy:.4f}")
                lines.append(f"    Precision: {precision:.4f}")
                lines.append(f"    Recall:    {recall:.4f}")
                lines.append(f"    F1 Score:  {f1:.4f}")
                lines.append(f"    Confusion Matrix:")
                lines.append(f"      TP: {tp:6,}  FP: {fp:6,}")
                lines.append(f"      FN: {fn:6,}  TN: {tn:6,}")
        
        # Footer
        lines.append("")
        lines.append("="*80)
        lines.append("END OF SUMMARY")
        lines.append("="*80)
        
        return "\n".join(lines)
    
    def save_summary(self, summary_text: str, filename: str = "training_summary.txt"):
        """
        Save summary to file.
        
        Args:
            summary_text: Summary text to save
            filename: Output filename
        """
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            f.write(summary_text)
        
        print(f"Results summary saved to: {output_path}")
        
        return output_path
    
    def generate_comparison_table(self, results_list: List[Dict]) -> str:
        """
        Generate comparison table for multiple model runs.
        
        Args:
            results_list: List of result dictionaries from different runs
        
        Returns:
            Comparison table as text
        """
        lines = []
        
        lines.append("="*120)
        lines.append("MODEL COMPARISON TABLE")
        lines.append("="*120)
        lines.append("")
        
        # Header
        header = f"{'Model':<20} | {'Params':>10} | {'Epochs':>7} | {'Time (min)':>10} | {'Val MAE':>10} | {'Test MAE':>10} | {'Test RMSE':>10} | {'Test R²':>10}"
        lines.append(header)
        lines.append("-"*120)
        
        # Sort by test MAE
        results_sorted = sorted(results_list, key=lambda x: x.get('test_mae', float('inf')))
        
        for result in results_sorted:
            model_name = result.get('model_name', 'Unknown')[:20]
            params = result.get('params', 0)
            epochs = result.get('epochs', 0)
            time_min = result.get('train_time_minutes', 0)
            val_mae = result.get('best_val_mae', 0)
            test_mae = result.get('test_mae', 0)
            test_rmse = result.get('test_rmse', 0)
            test_r2 = result.get('test_r2', 0)
            
            row = f"{model_name:<20} | {params:>10,} | {epochs:>7} | {time_min:>10.1f} | {val_mae:>10.4f} | {test_mae:>10.4f} | {test_rmse:>10.4f} | {test_r2:>10.4f}"
            lines.append(row)
        
        lines.append("="*120)
        
        return "\n".join(lines)


def test_results_summary():
    """Test results summary generator."""
    print("\nTesting ResultsSummaryGenerator...")
    
    # Create dummy data
    history = {
        'train_loss': [0.05, 0.03, 0.02, 0.015, 0.012],
        'train_mae': [0.15, 0.12, 0.10, 0.09, 0.08],
        'val_loss': [0.055, 0.035, 0.025, 0.020, 0.018],
        'val_mae': [0.16, 0.13, 0.11, 0.10, 0.09],
        'learning_rate': [0.001, 0.001, 0.001, 0.0005, 0.0005]
    }
    
    test_results = {
        'test_loss': 0.019,
        'mae': 0.095,
        'rmse': 0.138,
        'r2': 0.875
    }
    
    config = {
        'NODE_FEATURE_DIM': 12,
        'EDGE_FEATURE_DIM': 1,
        'COMP_FEATURE_DIM': 71,
        'HIDDEN_DIM': 128,
        'N_CONV_LAYERS': 4,
        'N_FC_LAYERS': 2,
        'DROPOUT': 0.1,
        'BATCH_SIZE': 64,
        'LEARNING_RATE': 0.001,
        'WEIGHT_DECAY': 0.0001,
        'N_EPOCHS': 100,
        'EARLY_STOPPING_PATIENCE': 15,
        'DEVICE': 'cuda'
    }
    
    # Generate predictions
    np.random.seed(42)
    n_samples = 1000
    true = np.random.rand(n_samples) * 0.5
    pred = true + np.random.randn(n_samples) * 0.05
    
    predictions = {
        'test': (pred, true)
    }
    
    # Generate summary
    generator = ResultsSummaryGenerator(output_dir='./test_results')
    
    summary = generator.generate_training_summary(
        model_name='test_cgcnn_v1',
        model_params=234625,
        history=history,
        test_results=test_results,
        config=config,
        train_time_minutes=26.3,
        predictions=predictions
    )
    
    print("\n" + summary)
    
    # Save
    generator.save_summary(summary, "test_summary.txt")
    
    print("\n✓ ResultsSummaryGenerator test passed!")


if __name__ == '__main__':
    test_results_summary()