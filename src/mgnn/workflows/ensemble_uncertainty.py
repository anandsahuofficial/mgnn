"""
Ensemble Uncertainty Analysis Workflow.
Uses deep ensemble for robust uncertainty quantification.
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


def run(config: ConfigManager, dm: DataManager):
    """
    Run ensemble uncertainty quantification.
    
    Steps:
    1. Load all ensemble models
    2. Compute ensemble predictions
    3. Calibration analysis
    4. Compare with MC Dropout
    5. Generate plots
    """
    
    print("\n" + "="*70)
    print("🎯 ENSEMBLE UNCERTAINTY QUANTIFICATION 🎯")
    print("="*70)
    print("\nUsing Deep Ensemble for reliable uncertainty estimates")
    print("="*70 + "\n")
    
    # Step 1: Load ensemble models
    print("[STEP 1/5] Loading ensemble models...")
    
    device = config.get('DEVICE', default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get checkpoint paths
    checkpoint_dir = Path(config.get('CHECKPOINT_DIR', required=True))
    n_models = config.get('ENSEMBLE_SIZE', default=5)
    seeds = config.get('ENSEMBLE_SEEDS', default=[42, 123, 456, 789, 1011])
    
    models = []
    
    for i, seed in enumerate(seeds):
        checkpoint_path = checkpoint_dir / f"cgcnn_ensemble_{i+1}_seed{seed}_best.pth"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Create model
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
        model.eval()
        
        models.append(model)
        
        print(f"  ✓ Loaded model {i+1}/{n_models} from: {checkpoint_path.name}")
    
    print(f"\n  Loaded {len(models)} ensemble members")
    
    # Step 2: Load test data
    print("\n[STEP 2/5] Loading test data...")
    
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
    
    targets = np.array([g.y.item() for g in graphs_test])

    # Also load validation data for calibration
    graphs_val_path = config.get('GRAPHS_VAL', required=True, vartype='file')

    with open(graphs_val_path, 'rb') as f:
        graphs_val = pickle.load(f)

    print(f"  Loaded {len(graphs_val)} validation graphs")

    val_loader = DataLoader(
        graphs_val,
        batch_size=config.get('BATCH_SIZE', default=64),
        shuffle=False,
        num_workers=0
    )

    val_targets = np.array([g.y.item() for g in graphs_val])
    
    # Step 3: Compute ensemble predictions
    print("\n[STEP 3/5] Computing ensemble predictions...")
    
    uq = UncertaintyQuantifier(models[0], device=device)  # Use first model for init
    
    mean_pred, std_pred, all_pred = uq.ensemble_predict(
        models,
        test_loader,
        show_progress=True
    )
    
    # Compute validation predictions for calibration
    print("\n  Computing validation predictions (for calibration)...")
    val_mean_pred, val_std_pred, val_all_pred = uq.ensemble_predict(
        models,
        val_loader,
        show_progress=True
    )


    # Save predictions
    output_dir = Path(config.get('OUTPUT_DIR', default='./results'))
    
    np.save(output_dir / 'ensemble_predictions_mean.npy', mean_pred)
    np.save(output_dir / 'ensemble_predictions_std.npy', std_pred)
    np.save(output_dir / 'ensemble_predictions_all.npy', all_pred)

    # Save validation predictions
    np.save(output_dir / 'ensemble_predictions_val_mean.npy', val_mean_pred)
    np.save(output_dir / 'ensemble_predictions_val_std.npy', val_std_pred)
    np.save(output_dir / 'val_targets.npy', val_targets)
    
    # Compute ensemble metrics
    ensemble_mae = np.mean(np.abs(mean_pred.flatten() - targets))
    ensemble_rmse = np.sqrt(np.mean((mean_pred.flatten() - targets) ** 2))
    
    ss_res = np.sum((targets - mean_pred.flatten()) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    ensemble_r2 = 1 - (ss_res / ss_tot)
    
    print(f"\n  Ensemble Performance:")
    print(f"    MAE:  {ensemble_mae:.6f} eV/atom")
    print(f"    RMSE: {ensemble_rmse:.6f} eV/atom")
    print(f"    R²:   {ensemble_r2:.6f}")
    
    # Step 4: Calibration analysis
    print("\n[STEP 4/5] Analyzing calibration...")
    
    calibration_stats = uq.calibration_analysis(
        mean_pred,
        std_pred,
        targets,
        n_bins=10
    )
    
    # Convert to JSON-serializable
    from mgnn.workflows.uncertainty_analysis import convert_to_serializable
    calibration_stats_json = convert_to_serializable(calibration_stats)
    
    # Save calibration stats
    dm.save({
        "ensemble/calibration_stats": {
            "data": json.dumps(calibration_stats_json, indent=2),
            "metadata": {
                "quality": calibration_stats['quality'],
                "correlation": float(calibration_stats['correlation']),
                "within_1sigma": float(calibration_stats['within_1sigma']),
                "ensemble_mae": float(ensemble_mae),
                "ensemble_r2": float(ensemble_r2)
            }
        }
    })
    
    # Step 5: Generate plots
    print("\n[STEP 5/5] Generating ensemble uncertainty plots...")
    
    fig = uq.plot_uncertainty_analysis(
        mean_pred,
        std_pred,
        targets,
        calibration_stats,
        output_path=str(output_dir / 'ensemble_uncertainty_analysis.png')
    )
    
    dm.save({
        "ensemble/figures/uncertainty_analysis": {
            "data": fig,
            "metadata": {"plot_type": "ensemble_uncertainty"}
        }
    })
    
    # Comparison with single model
    print("\n" + "="*70)
    print("📊 MC DROPOUT vs DEEP ENSEMBLE COMPARISON")
    print("="*70)
    
    # Load MC Dropout results if available
    mc_results_path = output_dir / 'test_predictions_std.npy'
    if mc_results_path.exists():
        mc_std = np.load(mc_results_path)
        mc_mean = np.load(output_dir / 'test_predictions_mean.npy')
        
        # Compare
        print("\nMC Dropout:")
        print(f"  Mean uncertainty: {mc_std.mean():.6f} eV/atom")
        print(f"  Uncertainty range: [{mc_std.min():.6f}, {mc_std.max():.6f}]")
        
        print("\nDeep Ensemble:")
        print(f"  Mean uncertainty: {std_pred.mean():.6f} eV/atom")
        print(f"  Uncertainty range: [{std_pred.min():.6f}, {std_pred.max():.6f}]")
        
        print(f"\nImprovement factor: {std_pred.mean() / mc_std.mean():.2f}x")
    
    # Final summary
    print("\n" + "="*70)
    print("🎉 ENSEMBLE UNCERTAINTY ANALYSIS COMPLETE! 🎉")
    print("="*70)
    print(f"\nEnsemble Performance:")
    print(f"  MAE:  {ensemble_mae:.6f} eV/atom")
    print(f"  R²:   {ensemble_r2:.6f}")
    print(f"\nCalibration Quality: {calibration_stats['quality']}")
    print(f"Mean Uncertainty: {std_pred.mean():.6f} eV/atom")
    print(f"Uncertainty-Error Correlation: {calibration_stats['correlation']:.3f}")
    print(f"\nCoverage:")
    print(f"  Within 1σ: {calibration_stats['within_1sigma']*100:.1f}% (expected: 68.3%)")
    print(f"  Within 2σ: {calibration_stats['within_2sigma']*100:.1f}% (expected: 95.5%)")
    print(f"\nOutput files:")
    print(f"  Predictions: {output_dir}/ensemble_predictions_mean.npy")
    print(f"  Uncertainties: {output_dir}/ensemble_predictions_std.npy")
    print(f"  Plot: {output_dir}/ensemble_uncertainty_analysis.png")
    print("="*70 + "\n")
    
    return mean_pred, std_pred, calibration_stats, ensemble_mae, ensemble_r2

