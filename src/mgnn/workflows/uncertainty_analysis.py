"""
Uncertainty Quantification Workflow.
Analyzes model confidence and calibration.
"""

import pickle
import json
import numpy as np
import torch
from pathlib import Path
from torch_geometric.loader import DataLoader

from mgnn.config_manager import ConfigManager
from mgnn.data_manager import DataManager
from mgnn.models.cgcnn import CGCNN
from mgnn.models.uncertainty import UncertaintyQuantifier


def convert_to_serializable(obj):
    """
    Convert numpy types to Python native types for JSON serialization.
    
    Args:
        obj: Object to convert (dict, list, numpy type, etc.)
    
    Returns:
        JSON-serializable version
    """
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
    Run uncertainty quantification workflow.
    
    Steps:
    1. Load best trained model
    2. MC Dropout uncertainty estimation
    3. Calibration analysis
    4. Identify high/low confidence predictions
    5. Generate uncertainty plots
    """
    
    print("\n" + "="*70)
    print("UNCERTAINTY QUANTIFICATION WORKFLOW")
    print("="*70 + "\n")
    
    # Step 1: Load model and data
    print("[STEP 1/5] Loading model and data...")
    
    device = config.get('DEVICE', default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint_path = config.get('CHECKPOINT_PATH', required=True, vartype='file')
    
    model = CGCNN(
        node_feature_dim=config.get('NODE_FEATURE_DIM', default=12),
        edge_feature_dim=config.get('EDGE_FEATURE_DIM', default=1),
        comp_feature_dim=config.get('COMP_FEATURE_DIM', default=71),
        hidden_dim=config.get('HIDDEN_DIM', default=128),
        n_conv=config.get('N_CONV_LAYERS', default=4),
        n_fc=config.get('N_FC_LAYERS', default=2),
        dropout=config.get('DROPOUT', default=0.1),
        use_comp_features=config.get('USE_COMP_FEATURES', default=True)
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"  Loaded model from: {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Best val loss: {checkpoint.get('best_val_loss', 'unknown')}")
    
    # Load test data
    graphs_test_path = config.get('GRAPHS_TEST', required=True, vartype='file')
    
    with open(graphs_test_path, 'rb') as f:
        graphs_test = pickle.load(f)
    
    print(f"  Loaded {len(graphs_test)} test graphs")
    
    test_loader = DataLoader(
        graphs_test,
        batch_size=config.get('BATCH_SIZE', default=64),
        shuffle=False,
        num_workers=0
    )
    
    # Step 2: MC Dropout uncertainty
    print("\n[STEP 2/5] Computing MC Dropout uncertainty...")
    
    uq = UncertaintyQuantifier(model, device=device)
    
    mean_pred, std_pred, all_pred = uq.mc_dropout_predict(
        test_loader,
        n_samples=config.get('MC_SAMPLES', default=30),
        show_progress=True
    )
    
    # Get true targets
    targets = np.array([g.y.item() for g in graphs_test])
    
    # Save predictions with uncertainty
    output_dir = Path(config.get('OUTPUT_DIR', default='./results'))
    
    np.save(output_dir / 'test_predictions_mean.npy', mean_pred)
    np.save(output_dir / 'test_predictions_std.npy', std_pred)
    np.save(output_dir / 'test_targets.npy', targets)
    
    print(f"  Saved predictions to: {output_dir}")
    
    # Step 3: Calibration analysis
    print("\n[STEP 3/5] Analyzing calibration...")
    
    calibration_stats = uq.calibration_analysis(
        mean_pred,
        std_pred,
        targets,
        n_bins=10
    )
    
    # Convert to JSON-serializable format
    calibration_stats_json = convert_to_serializable(calibration_stats)
    
    # Save calibration stats
    dm.save({
        "uncertainty/calibration_stats": {
            "data": json.dumps(calibration_stats_json, indent=2),
            "metadata": {
                "quality": calibration_stats['quality'],
                "correlation": float(calibration_stats['correlation']),
                "within_1sigma": float(calibration_stats['within_1sigma'])
            }
        }
    })
    
    # Step 4: Identify high/low confidence predictions
    print("\n[STEP 4/5] Identifying high/low confidence predictions...")
    
    # Sort by uncertainty
    uncertainty_indices = np.argsort(std_pred.flatten())
    
    # Most confident (low uncertainty)
    n_top = 50
    most_confident_idx = uncertainty_indices[:n_top]
    least_confident_idx = uncertainty_indices[-n_top:]
    
    print(f"\n  Most confident predictions:")
    print(f"    Mean uncertainty: {std_pred[most_confident_idx].mean():.6f} eV/atom")
    print(f"    Mean MAE: {np.abs(mean_pred[most_confident_idx] - targets[most_confident_idx].reshape(-1, 1)).mean():.6f} eV/atom")
    
    print(f"\n  Least confident predictions:")
    print(f"    Mean uncertainty: {std_pred[least_confident_idx].mean():.6f} eV/atom")
    print(f"    Mean MAE: {np.abs(mean_pred[least_confident_idx] - targets[least_confident_idx].reshape(-1, 1)).mean():.6f} eV/atom")
    
    # Save indices (convert to serializable)
    confident_data = {
        'most_confident_indices': [int(x) for x in most_confident_idx.tolist()],
        'least_confident_indices': [int(x) for x in least_confident_idx.tolist()],
        'most_confident_uncertainties': [float(x) for x in std_pred[most_confident_idx].flatten().tolist()],
        'least_confident_uncertainties': [float(x) for x in std_pred[least_confident_idx].flatten().tolist()]
    }
    
    dm.save({
        "uncertainty/confidence_analysis": {
            "data": json.dumps(confident_data, indent=2),
            "metadata": {
                "n_samples": len(graphs_test),
                "mean_uncertainty": float(std_pred.mean()),
                "max_uncertainty": float(std_pred.max())
            }
        }
    })
    
    # Step 5: Generate plots
    print("\n[STEP 5/5] Generating uncertainty plots...")
    
    fig = uq.plot_uncertainty_analysis(
        mean_pred,
        std_pred,
        targets,
        calibration_stats,
        output_path=str(output_dir / 'uncertainty_analysis.png')
    )
    
    dm.save({
        "uncertainty/figures/uncertainty_analysis": {
            "data": fig,
            "metadata": {"plot_type": "uncertainty_analysis"}
        }
    })
    
    # Summary
    print("\n" + "="*70)
    print("UNCERTAINTY QUANTIFICATION COMPLETE")
    print("="*70)
    print(f"\nCalibration Quality: {calibration_stats['quality']}")
    print(f"Mean Uncertainty: {std_pred.mean():.6f} eV/atom")
    print(f"Uncertainty-Error Correlation: {calibration_stats['correlation']:.3f}")
    print(f"\nCoverage:")
    print(f"  Within 1σ: {calibration_stats['within_1sigma']*100:.1f}% (expected: 68.3%)")
    print(f"  Within 2σ: {calibration_stats['within_2sigma']*100:.1f}% (expected: 95.5%)")
    print(f"\nOutput files:")
    print(f"  Predictions: {output_dir}/test_predictions_mean.npy")
    print(f"  Uncertainties: {output_dir}/test_predictions_std.npy")
    print(f"  Plot: {output_dir}/uncertainty_analysis.png")
    print("="*70 + "\n")
    
    return mean_pred, std_pred, calibration_stats