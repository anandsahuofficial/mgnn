"""
Uncertainty Calibration Workflow.
Apply temperature scaling to improve calibration.
"""

import pickle
import json
import numpy as np
from pathlib import Path

from mgnn.config_manager import ConfigManager
from mgnn.data_manager import DataManager
from mgnn.models.calibration import TemperatureScaling


def convert_to_serializable(obj):
    """Convert numpy types to Python native types."""
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64, np.floating)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.integer)):
        return int(obj)
    else:
        return obj


def run(config: ConfigManager, dm: DataManager):
    """
    Run uncertainty calibration workflow.
    
    Steps:
    1. Load ensemble predictions and targets
    2. Split into calibration (val) and test sets
    3. Fit temperature scaling on validation set
    4. Apply to test set
    5. Evaluate improvement
    6. Generate comparison plots
    """
    
    print("\n" + "="*70)
    print("⚡ TEMPERATURE SCALING CALIBRATION ⚡")
    print("="*70)
    print("\nImproving uncertainty calibration via temperature scaling")
    print("="*70 + "\n")
    
    # Step 1: Load predictions
    print("[STEP 1/5] Loading ensemble predictions...")
    
    output_dir = Path(config.get('OUTPUT_DIR', default='./results'))
    
    # Load validation predictions for calibration
    val_pred_mean = np.load(output_dir / 'ensemble_predictions_val_mean.npy')
    val_pred_std = np.load(output_dir / 'ensemble_predictions_val_std.npy')
    val_targets = np.load(output_dir / 'val_targets.npy')
    
    # Load test predictions
    test_pred_mean = np.load(output_dir / 'ensemble_predictions_mean.npy')
    test_pred_std = np.load(output_dir / 'ensemble_predictions_std.npy')
    test_targets = np.load(output_dir / 'test_targets.npy')
    
    print(f"  Loaded validation predictions: {len(val_targets)} samples")
    print(f"  Loaded test predictions: {len(test_targets)} samples")
    
    # Compute errors
    val_errors = np.abs(val_pred_mean.flatten() - val_targets)
    test_errors = np.abs(test_pred_mean.flatten() - test_targets)
    
    # Step 2: Fit temperature scaling on validation set
    print("\n[STEP 2/5] Fitting temperature scaling on validation set...")
    
    ts = TemperatureScaling()
    
    # Try different methods and pick best
    methods = ['nll', 'coverage', 'mse']
    best_quality_score = -1
    best_method = None
    best_temperature = None
    
    quality_scores = {'Excellent': 4, 'Good': 3, 'Fair': 2, 'Poor': 1}
    
    for method in methods:
        print(f"\n  Trying method: {method}")
        ts_temp = TemperatureScaling()
        ts_temp.fit(val_pred_std, val_errors, method=method)
        
        calibrated_val_std = ts_temp.transform(val_pred_std)
        metrics = ts_temp.evaluate_calibration(calibrated_val_std, val_errors, scaled=True)
        
        print(f"    Temperature: {ts_temp.temperature:.4f}")
        print(f"    Quality: {metrics['quality']}")
        print(f"    Coverage 1σ: {metrics['within_1sigma']:.1%}")
        
        score = quality_scores[metrics['quality']]
        if score > best_quality_score:
            best_quality_score = score
            best_method = method
            best_temperature = ts_temp.temperature
    
    print(f"\n  ✓ Best method: {best_method}")
    print(f"  ✓ Optimal temperature: {best_temperature:.4f}")
    
    # Fit with best method
    ts.fit(val_pred_std, val_errors, method=best_method)
    
    # Step 3: Apply to test set
    print("\n[STEP 3/5] Applying temperature scaling to test set...")
    
    test_pred_std_calibrated = ts.transform(test_pred_std)
    
    # Save calibrated predictions
    np.save(output_dir / 'ensemble_predictions_std_calibrated.npy', test_pred_std_calibrated)
    
    print(f"  Calibrated uncertainties saved to: {output_dir}/ensemble_predictions_std_calibrated.npy")
    
    # Step 4: Evaluate improvement
    print("\n[STEP 4/5] Evaluating calibration improvement...")
    
    # Before calibration
    metrics_before = ts.evaluate_calibration(test_pred_std, test_errors, scaled=False)
    
    # After calibration
    metrics_after = ts.evaluate_calibration(test_pred_std_calibrated, test_errors, scaled=True)
    
    print(f"\n  BEFORE Temperature Scaling:")
    print(f"    Quality: {metrics_before['quality']}")
    print(f"    Coverage 1σ: {metrics_before['within_1sigma']*100:.1f}% (expected: 68.3%)")
    print(f"    Coverage 2σ: {metrics_before['within_2sigma']*100:.1f}% (expected: 95.5%)")
    print(f"    Correlation: {metrics_before['correlation']:.3f}")
    
    print(f"\n  AFTER Temperature Scaling:")
    print(f"    Quality: {metrics_after['quality']}")
    print(f"    Coverage 1σ: {metrics_after['within_1sigma']*100:.1f}% (expected: 68.3%)")
    print(f"    Coverage 2σ: {metrics_after['within_2sigma']*100:.1f}% (expected: 95.5%)")
    print(f"    Correlation: {metrics_after['correlation']:.3f}")
    
    # Compute improvement
    improvement_1sigma = metrics_after['within_1sigma'] - metrics_before['within_1sigma']
    improvement_2sigma = metrics_after['within_2sigma'] - metrics_before['within_2sigma']
    
    print(f"\n  IMPROVEMENT:")
    print(f"    Coverage 1σ: {improvement_1sigma*100:+.1f}%")
    print(f"    Coverage 2σ: {improvement_2sigma*100:+.1f}%")
    print(f"    Quality: {metrics_before['quality']} → {metrics_after['quality']}")
    
    # Save calibration results
    calibration_results = {
        'temperature': float(ts.temperature),
        'method': best_method,
        'metrics_before': convert_to_serializable(metrics_before),
        'metrics_after': convert_to_serializable(metrics_after),
        'improvement_1sigma': float(improvement_1sigma),
        'improvement_2sigma': float(improvement_2sigma)
    }
    
    dm.save({
        "calibration/results": {
            "data": json.dumps(calibration_results, indent=2),
            "metadata": {
                "temperature": float(ts.temperature),
                "quality_before": metrics_before['quality'],
                "quality_after": metrics_after['quality'],
                "coverage_1sigma_after": float(metrics_after['within_1sigma'])
            }
        }
    })
    
    # Step 5: Generate plots
    print("\n[STEP 5/5] Generating calibration comparison plots...")
    
    fig = ts.plot_calibration_comparison(
        test_pred_std,
        test_pred_std_calibrated,
        test_errors,
        output_path=str(output_dir / 'calibration_comparison.png')
    )
    
    dm.save({
        "calibration/figures/comparison": {
            "data": fig,
            "metadata": {"plot_type": "calibration_comparison"}
        }
    })
    
    # Final summary
    print("\n" + "="*70)
    print("🎉 CALIBRATION COMPLETE! 🎉")
    print("="*70)
    print(f"\nTemperature Scaling Results:")
    print(f"  Optimal temperature: {ts.temperature:.4f}")
    print(f"  Method: {best_method}")
    print(f"\nCalibration Quality:")
    print(f"  Before: {metrics_before['quality']}")
    print(f"  After:  {metrics_after['quality']}")
    print(f"\nCoverage:")
    print(f"  1σ: {metrics_before['within_1sigma']*100:.1f}% → {metrics_after['within_1sigma']*100:.1f}%")
    print(f"  2σ: {metrics_before['within_2sigma']*100:.1f}% → {metrics_after['within_2sigma']*100:.1f}%")
    print(f"\nOutput files:")
    print(f"  Calibrated uncertainties: {output_dir}/ensemble_predictions_std_calibrated.npy")
    print(f"  Comparison plot: {output_dir}/calibration_comparison.png")
    print("="*70 + "\n")
    
    return ts, metrics_before, metrics_after