"""
Model Training Workflow.
Purpose: Train CGCNN model and evaluate performance.
"""

import pickle
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch_geometric.loader import DataLoader

from mgnn.config_manager import ConfigManager
from mgnn.data_manager import DataManager
from mgnn.models.cgcnn import CGCNN, count_parameters
from mgnn.models.trainer import Trainer
import time

def plot_training_curves(history: dict, output_dir: Path) -> plt.Figure:
    """
    Plot training and validation curves.
    
    Args:
        history: Training history dictionary
        output_dir: Directory to save plots
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor('white')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # MAE curves
    ax = axes[1]
    ax.plot(epochs, history['train_mae'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, history['val_mae'], 'r-', label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE (eV/atom)')
    ax.set_title('Training and Validation MAE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Learning rate
    ax = axes[2]
    ax.plot(epochs, history['learning_rate'], 'g-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def plot_parity(predictions: np.ndarray, targets: np.ndarray, 
                split: str, output_dir: Path) -> plt.Figure:
    """
    Create parity plot (predicted vs actual).
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        split: Data split name ('train', 'val', 'test')
        output_dir: Directory to save plots
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('white')
    
    # Flatten arrays
    pred = predictions.flatten()
    true = targets.flatten()
    
    # Compute metrics
    mae = np.mean(np.abs(pred - true))
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    r2 = 1 - np.sum((true - pred) ** 2) / np.sum((true - np.mean(true)) ** 2)
    
    # Scatter plot with density coloring
    from matplotlib.colors import LogNorm
    h = ax.hist2d(true, pred, bins=100, cmap='viridis', 
                  norm=LogNorm(), cmin=1)
    plt.colorbar(h[3], ax=ax, label='Count')
    
    # Perfect prediction line
    min_val = min(true.min(), pred.min())
    max_val = max(true.max(), pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2, label='Perfect prediction')
    
    # Labels and metrics
    ax.set_xlabel('True Decomposition Energy (eV/atom)', fontsize=12)
    ax.set_ylabel('Predicted Decomposition Energy (eV/atom)', fontsize=12)
    ax.set_title(f'Parity Plot - {split.capitalize()} Set', fontsize=14, weight='bold')
    
    # Add metrics text box
    textstr = f'MAE:  {mae:.4f} eV/atom\nRMSE: {rmse:.4f} eV/atom\nR²:   {r2:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    return fig


def plot_error_distribution(predictions: np.ndarray, targets: np.ndarray,
                            split: str, output_dir: Path) -> plt.Figure:
    """
    Plot error distribution.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        split: Data split name
        output_dir: Directory to save plots
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('white')
    
    # Compute errors
    errors = predictions.flatten() - targets.flatten()
    abs_errors = np.abs(errors)
    
    # Error histogram
    ax = axes[0]
    ax.hist(errors, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax.axvline(np.mean(errors), color='orange', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(errors):.4f}')
    ax.set_xlabel('Prediction Error (eV/atom)')
    ax.set_ylabel('Count')
    ax.set_title(f'Error Distribution - {split.capitalize()} Set')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Absolute error histogram
    ax = axes[1]
    ax.hist(abs_errors, bins=50, alpha=0.7, color='coral', edgecolor='black')
    ax.axvline(np.median(abs_errors), color='green', linestyle='--', linewidth=2,
               label=f'Median: {np.median(abs_errors):.4f}')
    ax.axvline(np.mean(abs_errors), color='orange', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(abs_errors):.4f}')
    ax.set_xlabel('Absolute Error (eV/atom)')
    ax.set_ylabel('Count')
    ax.set_title(f'Absolute Error Distribution - {split.capitalize()} Set')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def plot_error_by_stability(predictions: np.ndarray, targets: np.ndarray,
                            output_dir: Path) -> plt.Figure:
    """
    Plot error by stability class.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        output_dir: Directory to save plots
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    
    # Flatten
    pred = predictions.flatten()
    true = targets.flatten()
    
    # Classify by stability
    stable_mask = true <= 0.025
    metastable_mask = (true > 0.025) & (true <= 0.1)
    unstable_mask = true > 0.1
    
    # Compute MAE for each class
    classes = ['Stable\n(≤0.025)', 'Metastable\n(0.025-0.1)', 'Unstable\n(>0.1)']
    masks = [stable_mask, metastable_mask, unstable_mask]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    maes = []
    counts = []
    
    for mask in masks:
        if mask.sum() > 0:
            mae = np.mean(np.abs(pred[mask] - true[mask]))
            maes.append(mae)
            counts.append(mask.sum())
        else:
            maes.append(0)
            counts.append(0)
    
    # Bar plot
    bars = ax.bar(range(len(classes)), maes, color=colors, alpha=0.7, 
                  edgecolor='black', linewidth=1.5)
    
    # Add count labels on bars
    for i, (bar, mae, count) in enumerate(zip(bars, maes, counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'MAE: {mae:.4f}\nn={count:,}',
               ha='center', va='bottom', fontsize=10, weight='bold')
    
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_ylabel('Mean Absolute Error (eV/atom)')
    ax.set_title('Prediction Error by Stability Class', fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    return fig


def run(config: ConfigManager, dm: DataManager):
    """
    Run model training workflow.
    
    Steps:
    1. Load preprocessed graphs
    2. Create data loaders
    3. Initialize model
    4. Train model
    5. Evaluate on test set
    6. Generate plots and save results
    """
    
    print("\n" + "="*70)
    print("MODEL TRAINING WORKFLOW")
    print("="*70 + "\n")
    
    # Step 1: Load preprocessed graphs
    print("[STEP 1/6] Loading preprocessed graphs...")
    
    graphs_train_path = config.get('GRAPHS_TRAIN', required=True, vartype='file')
    graphs_val_path = config.get('GRAPHS_VAL', required=True, vartype='file')
    graphs_test_path = config.get('GRAPHS_TEST', required=True, vartype='file')
    
    with open(graphs_train_path, 'rb') as f:
        graphs_train = pickle.load(f)
    
    with open(graphs_val_path, 'rb') as f:
        graphs_val = pickle.load(f)
    
    with open(graphs_test_path, 'rb') as f:
        graphs_test = pickle.load(f)
    
    print(f"  Loaded {len(graphs_train)} training graphs")
    print(f"  Loaded {len(graphs_val)} validation graphs")
    print(f"  Loaded {len(graphs_test)} test graphs")
    
    # Step 2: Create data loaders
    print("\n[STEP 2/6] Creating data loaders...")
    
    batch_size = config.get('BATCH_SIZE', default=64)
    n_workers = config.get('N_WORKERS', default=4)
    
    train_loader = DataLoader(
        graphs_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    val_loader = DataLoader(
        graphs_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        graphs_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    
    # Step 3: Initialize model
    print("\n[STEP 3/6] Initializing model...")
    
    device = config.get('DEVICE', default='cuda' if torch.cuda.is_available() else 'cpu')
    
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
    
    n_params = count_parameters(model)
    print(f"  Total parameters: {n_params:,}")
    print(f"  Device: {device}")
    
    # Step 4: Train model
    print("\n[STEP 4/6] Training model...")
    
    checkpoint_dir = config.get('CHECKPOINT_DIR', default='./checkpoints')
    model_name = config.get('MODEL_NAME', default='cgcnn_v1')

    # Check for existing checkpoint to resume from
    resume_from = None
    resume_checkpoint = config.get('RESUME_FROM_CHECKPOINT', default=None)

    if resume_checkpoint:
        # User specified a checkpoint
        resume_from = resume_checkpoint
    elif config.get('AUTO_RESUME', default=True):
        # Auto-detect latest checkpoint
        latest_checkpoint = Path(checkpoint_dir) / f'{model_name}_latest.pth'
        if latest_checkpoint.exists():
            print(f"\nFound existing checkpoint: {latest_checkpoint}")
            user_input = input("Resume training from this checkpoint? (y/n): ").strip().lower()
            if user_input == 'y':
                resume_from = str(latest_checkpoint)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=config.get('LEARNING_RATE', default=0.001),
        weight_decay=config.get('WEIGHT_DECAY', default=0.0001),
        device=device,
        output_dir=checkpoint_dir,
        model_name=model_name
    )

    # Track training time
    import time
    train_start_time = time.time()
    
    history = trainer.train(
        n_epochs=config.get('N_EPOCHS', default=100),
        early_stopping_patience=config.get('EARLY_STOPPING_PATIENCE', default=15),
        save_best_only=config.get('SAVE_BEST_ONLY', default=True),
        verbose=True,
        resume_from=resume_from 
    )

    train_end_time = time.time()
    train_time_minutes = (train_end_time - train_start_time) / 60
    
    # Save training history
    dm.save({
        "training/history": {
            "data": json.dumps(history, indent=2),
            "metadata": {
                "n_epochs": len(history['train_loss']),
                "best_val_loss": float(min(history['val_loss'])),
                "final_train_loss": float(history['train_loss'][-1]),
                "final_val_loss": float(history['val_loss'][-1]),
                "train_time_minutes": train_time_minutes
            }
        }
    })
    
    # Step 5: Evaluate on test set
    # print("\n[STEP 5/6] Evaluating on test set...")
    
    # # Load best model
    # best_model_path = Path(checkpoint_dir) / f'{model_name}_best.pth'
    # if best_model_path.exists():
    #     trainer.load_checkpoint(best_model_path)
    
    # test_results = trainer.test()
    
    # # Save test results
    # test_metrics = {
    #     'test_loss': float(test_results['test_loss']),
    #     'test_mae': float(test_results['mae']),
    #     'test_rmse': float(test_results['rmse']),
    #     'test_r2': float(test_results['r2'])
    # }
    
    # dm.save({
    #     "training/test_metrics": {
    #         "data": json.dumps(test_metrics, indent=2),
    #         "metadata": test_metrics
    #     }
    # })

    # Step 5: Evaluate on test set
    print("\n[STEP 5/6] Evaluating on test set...")

    # Load best model
    best_model_path = Path(checkpoint_dir) / f'{model_name}_best.pth'
    if best_model_path.exists():
        print(f"Loading best model from: {best_model_path}")
        trainer.load_checkpoint(best_model_path)

    # Get predictions for validation and test
    print("Computing validation predictions...")
    val_loss, val_mae, val_pred, val_true = trainer.validate(val_loader)

    print("Computing test predictions...")
    test_loss, test_mae, test_pred, test_true = trainer.validate(test_loader)

    # Compute correct test metrics
    test_pred_flat = test_pred.flatten()
    test_true_flat = test_true.flatten()

    test_mse = np.mean((test_pred_flat - test_true_flat) ** 2)
    test_rmse = np.sqrt(test_mse)
    test_mae_correct = np.mean(np.abs(test_pred_flat - test_true_flat))

    # Compute R² correctly
    ss_res = np.sum((test_true_flat - test_pred_flat) ** 2)
    ss_tot = np.sum((test_true_flat - np.mean(test_true_flat)) ** 2)
    test_r2 = 1 - (ss_res / ss_tot)

    test_results = {
        'test_loss': float(test_loss),
        'test_mae': float(test_mae),
        'mae': float(test_mae_correct),
        'mse': float(test_mse),
        'rmse': float(test_rmse),
        'r2': float(test_r2),
        'predictions': test_pred,
        'targets': test_true
    }

    print(f"\nTest Results (CORRECTED):")
    print(f"  Loss: {test_loss:.6f}")
    print(f"  MAE:  {test_mae_correct:.6f} eV/atom")
    print(f"  RMSE: {test_rmse:.6f} eV/atom")
    print(f"  R²:   {test_r2:.6f}")

    # Save test results
    test_metrics = {
        'test_loss': float(test_loss),
        'test_mae': float(test_mae_correct),
        'test_rmse': float(test_rmse),
        'test_r2': float(test_r2)
    }

    dm.save({
        "training/test_metrics": {
            "data": json.dumps(test_metrics, indent=2),
            "metadata": test_metrics
        }
    })
        
    # Step 6: Generate plots
    if config.get('GENERATE_PLOTS', default=True):
        print("\n[STEP 6/6] Generating evaluation plots...")
        
        # Training curves
        fig_curves = plot_training_curves(history, Path(checkpoint_dir))
        dm.save({
            "training/figures/training_curves": {
                "data": fig_curves,
                "metadata": {
                    "plot_type": "training_curves",
                    "n_epochs": len(history['train_loss'])
                }
            }
        })
        
        # Get predictions for all splits
        _, _, val_pred, val_true = trainer.validate(val_loader)
        _, _, test_pred, test_true = trainer.validate(test_loader)
        
        # Parity plots
        fig_parity_val = plot_parity(val_pred, val_true, 'validation', Path(checkpoint_dir))
        dm.save({
            "training/figures/parity_validation": {
                "data": fig_parity_val,
                "metadata": {"plot_type": "parity", "split": "validation"}
            }
        })
        
        fig_parity_test = plot_parity(test_pred, test_true, 'test', Path(checkpoint_dir))
        dm.save({
            "training/figures/parity_test": {
                "data": fig_parity_test,
                "metadata": {"plot_type": "parity", "split": "test"}
            }
        })
        
        # Error distributions
        fig_error_val = plot_error_distribution(val_pred, val_true, 'validation', Path(checkpoint_dir))
        dm.save({
            "training/figures/error_distribution_validation": {
                "data": fig_error_val,
                "metadata": {"plot_type": "error_distribution", "split": "validation"}
            }
        })
        
        fig_error_test = plot_error_distribution(test_pred, test_true, 'test', Path(checkpoint_dir))
        dm.save({
            "training/figures/error_distribution_test": {
                "data": fig_error_test,
                "metadata": {"plot_type": "error_distribution", "split": "test"}
            }
        })
        
        # Error by stability
        fig_stability = plot_error_by_stability(test_pred, test_true, Path(checkpoint_dir))
        dm.save({
            "training/figures/error_by_stability": {
                "data": fig_stability,
                "metadata": {"plot_type": "error_by_stability"}
            }
        })
        
        print("  Generated 6 evaluation plots")
    
    # Generate PDF report
    print("\n[STEP 6/6] Generating training report PDF...")
    output_dir = Path(config.get('OUTPUT_DIR', default='./results'))
    pdf_path = output_dir / 'training_report.pdf'
    dm.generate_pdf_report(
        output_file=pdf_path,
        include_archived=False,
        title="CGCNN Training Report",
        author="MGNN Pipeline"
    )

    # Generate comprehensive text summary
    print("\n[STEP 7/7] Generating comprehensive text summary...")

    from mgnn.analysis.results_summary import ResultsSummaryGenerator

    summary_generator = ResultsSummaryGenerator(output_dir=output_dir)

    # Collect all predictions
    predictions_all = {
        'validation': (val_pred, val_true),
        'test': (test_pred, test_true)
    }

    # Generate summary
    summary_text = summary_generator.generate_training_summary(
        model_name=model_name,
        model_params=n_params,
        history=history,
        test_results=test_results,
        config=config.config,  # The full config dictionary
        train_time_minutes=train_time_minutes,
        predictions=predictions_all
    )

    # Save summary
    summary_path = summary_generator.save_summary(summary_text, f"{model_name}_results_summary.txt")

    # Also print key results to console
    print("\n" + "="*70)
    print("KEY RESULTS (see full summary in text file)")
    print("="*70)
    print(summary_text.split("TEST SET PERFORMANCE")[1].split("-"*80)[0])

    # Also print key results to console
    print("\n" + "="*70)
    print("KEY RESULTS (see full summary in text file)")
    print("="*70)
    try:
        test_section = summary_text.split("TEST SET PERFORMANCE")
        if len(test_section) > 1:
            key_results = test_section[1].split("-"*80)[0]
            print(key_results)
    except:
        pass


    # Generate PDF report
    print("\nGenerating training report PDF...")
    pdf_path = output_dir / 'training_report.pdf'
    dm.generate_pdf_report(
        output_file=pdf_path,
        include_archived=False,
        title="CGCNN Training Report",
        author="MGNN Pipeline"
    )



    # Final summary
    print("\n" + "="*70)
    print("MODEL TRAINING COMPLETE")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Parameters: {n_params:,}")
    print(f"Epochs trained: {len(history['train_loss'])}")
    print(f"Training time: {train_time_minutes:.1f} minutes ({train_time_minutes/60:.1f} hours)")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")
    print(f"\nTest Results:")
    print(f"  MAE:  {test_metrics['test_mae']:.4f} eV/atom")
    print(f"  RMSE: {test_metrics['test_rmse']:.4f} eV/atom")
    print(f"  R²:   {test_metrics['test_r2']:.4f}")
    print(f"\nSaved files:")
    print(f"  Checkpoint: {best_model_path}")
    print(f"  Report PDF: {pdf_path}")
    print(f"  Summary TXT: {summary_path}")
    print("="*70 + "\n")

    return model, trainer, test_results
    
    # # Final summary
    # print("\n" + "="*70)
    # print("MODEL TRAINING COMPLETE")
    # print("="*70)
    # print(f"Model: {model_name}")
    # print(f"Parameters: {n_params:,}")
    # print(f"Epochs trained: {len(history['train_loss'])}")
    # print(f"Best validation loss: {min(history['val_loss']):.4f}")
    # print(f"\nTest Results:")
    # print(f"  MAE:  {test_metrics['test_mae']:.4f} eV/atom")
    # print(f"  RMSE: {test_metrics['test_rmse']:.4f} eV/atom")
    # print(f"  R²:   {test_metrics['test_r2']:.4f}")
    # print(f"\nCheckpoint: {best_model_path}")
    # print(f"Report: {pdf_path}")
    # print("="*70 + "\n")
    
    # return model, trainer, test_results


# # To run this workflow independently for testing:

# cd /projectnb/cui-buchem/anandsahu/Softwares/Mine/multinary-gnn
# uv sync
# uv run python -m mgnn.analysis.results_summary

# cd /projectnb/cui-buchem/anandsahu/Softwares/Mine/multinary-gnn
# uv sync

# Since your training already finished, you can test by loading the results
# Or just check that the module imports work
# uv run python -c "from mgnn.workflows.model_training import run; print('OK')"