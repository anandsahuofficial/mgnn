"""
Deep Ensemble Training Workflow.
Trains multiple models with different random seeds for robust uncertainty quantification.
"""

import pickle
import json
import numpy as np
import torch
import time
from pathlib import Path
from torch_geometric.loader import DataLoader

from mgnn.config_manager import ConfigManager
from mgnn.data_manager import DataManager
from mgnn.models.cgcnn import CGCNN, count_parameters
from mgnn.models.trainer import Trainer
from mgnn.models.losses import HybridLoss


def run(config: ConfigManager, dm: DataManager):
    """
    Train deep ensemble of CGCNN models.
    
    Steps:
    1. Load augmented training data
    2. Train N models with different random seeds
    3. Save all ensemble members
    4. Compute ensemble statistics
    """
    
    print("\n" + "="*70)
    print("🎯 DEEP ENSEMBLE TRAINING WORKFLOW 🎯")
    print("="*70)
    print("\nDeep Ensembles provide:")
    print("  • Better uncertainty quantification than MC Dropout")
    print("  • Often improved prediction accuracy")
    print("  • Robust performance estimates")
    print("="*70 + "\n")
    
    # Configuration
    n_models = config.get('ENSEMBLE_SIZE', default=5)
    seeds = config.get('ENSEMBLE_SEEDS', default=[42, 123, 456, 789, 1011])
    
    if len(seeds) != n_models:
        seeds = [42 + i * 111 for i in range(n_models)]
    
    print(f"Training ensemble of {n_models} models")
    print(f"Random seeds: {seeds}\n")
    
    # Step 1: Load data
    print("[STEP 1/3] Loading augmented training data...")
    
    graphs_train_path = config.get('GRAPHS_TRAIN', required=True, vartype='file')
    graphs_val_path = config.get('GRAPHS_VAL', required=True, vartype='file')
    
    with open(graphs_train_path, 'rb') as f:
        graphs_train = pickle.load(f)
    
    with open(graphs_val_path, 'rb') as f:
        graphs_val = pickle.load(f)
    
    print(f"  Loaded {len(graphs_train)} training graphs")
    print(f"  Loaded {len(graphs_val)} validation graphs")
    
    # Create data loaders
    batch_size = config.get('BATCH_SIZE', default=64)
    
    train_loader = DataLoader(
        graphs_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        graphs_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Step 2: Train ensemble members
    print(f"\n[STEP 2/3] Training {n_models} ensemble members...")
    
    device = config.get('DEVICE', default='cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = Path(config.get('CHECKPOINT_DIR', default='./checkpoints_ensemble'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    ensemble_histories = []
    ensemble_metrics = []
    total_train_time = 0
    
    for i, seed in enumerate(seeds):
        print("\n" + "-"*70)
        print(f"ENSEMBLE MEMBER {i+1}/{n_models} (seed={seed})")
        print("-"*70)
        
        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
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
        
        # Create loss function
        criterion = HybridLoss(
            stable_weight=config.get('STABLE_WEIGHT', default=10.0),
            focal_gamma=config.get('FOCAL_GAMMA', default=1.5),
            weighted_ratio=config.get('WEIGHTED_RATIO', default=0.7)
        )
        
        # Create trainer
        model_name = f"cgcnn_ensemble_{i+1}_seed{seed}"
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=config.get('LEARNING_RATE', default=0.001),
            weight_decay=config.get('WEIGHT_DECAY', default=0.0001),
            device=device,
            output_dir=str(checkpoint_dir),
            model_name=model_name,
            criterion=criterion
        )
        
        # Train
        model_start_time = time.time()
        
        history = trainer.train(
            n_epochs=config.get('N_EPOCHS', default=150),
            early_stopping_patience=config.get('EARLY_STOPPING_PATIENCE', default=20),
            save_best_only=True,
            verbose=True
        )
        
        model_end_time = time.time()
        model_train_time = (model_end_time - model_start_time) / 60
        total_train_time += model_train_time
        
        # Store results
        ensemble_histories.append(history)
        
        metrics = {
            'seed': int(seed),
            'model_id': i + 1,
            'n_epochs': len(history['train_loss']),
            'best_epoch': int(np.argmin(history['val_loss']) + 1),
            'best_val_loss': float(min(history['val_loss'])),
            'best_val_mae': float(min(history['val_mae'])),
            'final_train_loss': float(history['train_loss'][-1]),
            'final_val_loss': float(history['val_loss'][-1]),
            'train_time_minutes': float(model_train_time)
        }
        
        ensemble_metrics.append(metrics)
        
        print(f"\n✓ Model {i+1} complete:")
        print(f"  Best val MAE: {metrics['best_val_mae']:.6f} eV/atom")
        print(f"  Training time: {model_train_time:.1f} minutes")
    
    # Step 3: Ensemble statistics
    print("\n" + "="*70)
    print("[STEP 3/3] Computing ensemble statistics...")
    print("="*70)
    
    # Extract metrics
    best_val_maes = [m['best_val_mae'] for m in ensemble_metrics]
    train_times = [m['train_time_minutes'] for m in ensemble_metrics]
    
    ensemble_stats = {
        'n_models': n_models,
        'seeds': [int(s) for s in seeds],
        'total_train_time_minutes': float(total_train_time),
        'mean_best_val_mae': float(np.mean(best_val_maes)),
        'std_best_val_mae': float(np.std(best_val_maes)),
        'min_best_val_mae': float(np.min(best_val_maes)),
        'max_best_val_mae': float(np.max(best_val_maes)),
        'mean_train_time_minutes': float(np.mean(train_times)),
        'individual_models': ensemble_metrics
    }
    
    print(f"\n📊 ENSEMBLE STATISTICS:")
    print(f"  Number of models: {n_models}")
    print(f"  Mean best val MAE: {ensemble_stats['mean_best_val_mae']:.6f} ± {ensemble_stats['std_best_val_mae']:.6f} eV/atom")
    print(f"  Range: [{ensemble_stats['min_best_val_mae']:.6f}, {ensemble_stats['max_best_val_mae']:.6f}]")
    print(f"  Total training time: {total_train_time:.1f} minutes ({total_train_time/60:.1f} hours)")
    print(f"  Average per model: {np.mean(train_times):.1f} minutes")
    
    # Save ensemble statistics
    dm.save({
        "ensemble/training_stats": {
            "data": json.dumps(ensemble_stats, indent=2),
            "metadata": {
                "n_models": n_models,
                "mean_best_val_mae": ensemble_stats['mean_best_val_mae'],
                "total_train_time_minutes": ensemble_stats['total_train_time_minutes']
            }
        }
    })
    
    # Save checkpoint paths
    checkpoint_paths = [
        str(checkpoint_dir / f"cgcnn_ensemble_{i+1}_seed{seeds[i]}_best.pth")
        for i in range(n_models)
    ]
    
    checkpoint_info = {
        'checkpoint_dir': str(checkpoint_dir),
        'checkpoint_paths': checkpoint_paths,
        'n_models': n_models,
        'seeds': [int(s) for s in seeds]
    }
    
    dm.save({
        "ensemble/checkpoint_info": {
            "data": json.dumps(checkpoint_info, indent=2),
            "metadata": {"n_models": n_models}
        }
    })
    
    # Final summary
    print("\n" + "="*70)
    print("🎉 ENSEMBLE TRAINING COMPLETE! 🎉")
    print("="*70)
    print(f"\nTrained {n_models} models successfully!")
    print(f"Ensemble checkpoints saved to: {checkpoint_dir}")
    print(f"\nNext step: Run ensemble uncertainty analysis")
    print(f"  Config: workflows/08_ensemble_uncertainty.inp")
    print("="*70 + "\n")
    
    return ensemble_stats, ensemble_metrics


if __name__ == '__main__':
    print("Ensemble training workflow")