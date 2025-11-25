"""
SHAP Interpretability Workflow.
Explains model predictions and discovers design rules.
"""

import pickle
import json
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from mgnn.config_manager import ConfigManager
from mgnn.data_manager import DataManager
from mgnn.models.cgcnn import CGCNN
from mgnn.analysis.shap_analysis import CGCNNShapExplainer, get_composition_feature_names


def convert_to_serializable(obj):
    """Convert numpy/pandas types to JSON-serializable."""
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
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    else:
        return obj


def run(config: ConfigManager, dm: DataManager):
    """
    Run SHAP interpretability analysis.
    
    Steps:
    1. Load trained model
    2. Load test graphs
    3. Compute SHAP values for composition features
    4. Analyze feature importance
    5. Create visualization plots
    6. Explain specific predictions
    7. Extract design rules
    """
    
    print("\n" + "="*70)
    print("🔬 SHAP INTERPRETABILITY ANALYSIS 🔬")
    print("="*70)
    print("\nDiscovering which features drive stability predictions")
    print("="*70 + "\n")
    
    # Step 1: Load model
    print("[STEP 1/7] Loading trained ensemble model...")
    
    device = config.get('DEVICE', default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load best ensemble model (or single model)
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
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"  Loaded model from: {checkpoint_path}")
    
    # Step 2: Load test data
    print("\n[STEP 2/7] Loading test data...")
    
    graphs_test_path = config.get('GRAPHS_TEST', required=True, vartype='file')
    
    with open(graphs_test_path, 'rb') as f:
        graphs_test = pickle.load(f)
    
    print(f"  Loaded {len(graphs_test)} test graphs")
    
    # Step 3: Initialize SHAP explainer
    print("\n[STEP 3/7] Computing SHAP values...")
    
    explainer = CGCNNShapExplainer(model, device=device)
    
    # Get feature names
    feature_names = get_composition_feature_names()
    print(f"  Using {len(feature_names)} composition features")
    
    # Compute SHAP values
    background_size = config.get('SHAP_BACKGROUND_SIZE', default=100)
    test_size = config.get('SHAP_TEST_SIZE', default=200)
    
    shap_values, base_value, _ = explainer.compute_composition_shap(
        graphs_test,
        background_size=background_size,
        test_size=test_size,
        feature_names=feature_names
    )
    
    # Save SHAP values
    output_dir = Path(config.get('OUTPUT_DIR', default='./results'))
    
    np.save(output_dir / 'shap_values.npy', shap_values)
    np.save(output_dir / 'shap_base_value.npy', base_value)
    
    print(f"  SHAP values saved to: {output_dir}/shap_values.npy")
    
    # Step 4: Analyze feature importance
    print("\n[STEP 4/7] Analyzing feature importance...")
    
    importance_df = explainer.analyze_feature_importance(
        shap_values,
        feature_names,
        top_k=config.get('TOP_K_FEATURES', default=20)
    )
    
    # Save importance dataframe
    importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)
    print(f"  Feature importance saved to: {output_dir}/feature_importance.csv")
    
    # Save to data manager
    importance_json = convert_to_serializable(importance_df.head(20).to_dict('records'))
    
    dm.save({
        "shap/feature_importance": {
            "data": json.dumps(importance_json, indent=2),
            "metadata": {
                "top_feature": importance_df.iloc[0]['feature'],
                "top_importance": float(importance_df.iloc[0]['importance']),
                "n_features": len(feature_names)
            }
        }
    })
    
    # Step 5: Create visualization plots
    print("\n[STEP 5/7] Creating SHAP visualizations...")
    
    # Plot 1: Feature importance bar chart
    fig_importance = explainer.plot_feature_importance(
        importance_df,
        top_k=config.get('TOP_K_FEATURES', default=20),
        output_path=str(output_dir / 'shap_feature_importance.png')
    )
    
    dm.save({
        "shap/figures/feature_importance": {
            "data": fig_importance,
            "metadata": {"plot_type": "feature_importance"}
        }
    })
    
    plt.close(fig_importance)
    
    # Plot 2: SHAP summary plot (beeswarm)
    print("  Creating SHAP summary plot...")
    
    # Get test features
    test_features = np.stack([
        graphs_test[i].comp_features.numpy() 
        for i in range(min(test_size, len(graphs_test)))
    ])
    
    fig_summary = explainer.plot_shap_summary(
        shap_values,
        test_features,
        feature_names,
        top_k=config.get('TOP_K_FEATURES', default=20),
        output_path=str(output_dir / 'shap_summary.png')
    )
    
    dm.save({
        "shap/figures/summary": {
            "data": fig_summary,
            "metadata": {"plot_type": "shap_summary"}
        }
    })
    
    plt.close(fig_summary)
    
    # Plot 3: SHAP dependence plots for top features
    print("  Creating SHAP dependence plots...")
    
    top_features_idx = importance_df.head(5).index.tolist()
    
    for i, feat_idx in enumerate(top_features_idx):
        fig_dep = explainer.plot_shap_dependence(
            shap_values,
            test_features,
            feature_names,
            feature_idx=feat_idx,
            output_path=str(output_dir / f'shap_dependence_top{i+1}.png')
        )
        
        dm.save({
            f"shap/figures/dependence_top{i+1}": {
                "data": fig_dep,
                "metadata": {
                    "plot_type": "shap_dependence",
                    "feature": feature_names[feat_idx]
                }
            }
        })
        
        plt.close(fig_dep)
    
    # Step 6: Explain specific predictions
    print("\n[STEP 6/7] Generating detailed explanations...")
    
    # Select interesting cases to explain
    targets = np.array([g.y.item() for g in graphs_test[:test_size]])
    
    # Most stable
    stable_idx = np.argsort(targets)[:5]
    # Least stable
    unstable_idx = np.argsort(targets)[-5:]
    # Medium stability
    median_val = np.median(targets)
    medium_idx = np.argsort(np.abs(targets - median_val))[:5]
    
    explain_indices = list(stable_idx) + list(medium_idx) + list(unstable_idx)
    
    explanations = explainer.explain_predictions(
        graphs_test[:test_size],
        explain_indices,
        feature_names,
        output_dir=str(output_dir / 'shap_explanations')
    )
    
    # Save explanations
    explanations_json = convert_to_serializable(explanations)
    
    with open(output_dir / 'prediction_explanations.json', 'w') as f:
        json.dump(explanations_json, f, indent=2)
    
    print(f"  Explanations saved to: {output_dir}/prediction_explanations.json")
    
    dm.save({
        "shap/explanations": {
            "data": json.dumps(explanations_json, indent=2),
            "metadata": {"n_explanations": len(explanations)}
        }
    })
    
    # Step 7: Extract design rules
    print("\n[STEP 7/7] Extracting design rules...")
    
    design_rules = extract_design_rules(
        shap_values,
        test_features,
        targets,
        feature_names,
        importance_df
    )
    
    # Save design rules
    dm.save({
        "shap/design_rules": {
            "data": json.dumps(design_rules, indent=2),
            "metadata": {"n_rules": len(design_rules.get('rules', []))}
        }
    })
    
    # Print design rules
    print("\n" + "="*70)
    print("🎯 DESIGN RULES FOR STABLE PEROVSKITES")
    print("="*70)
    
    for i, rule in enumerate(design_rules['rules'][:10], 1):
        print(f"\n{i}. {rule['description']}")
        print(f"   Impact: {rule['impact']}")
        print(f"   Evidence: {rule['evidence']}")
    
    # Summary statistics
    print("\n" + "="*70)
    print("📊 SHAP ANALYSIS SUMMARY")
    print("="*70)
    
    print(f"\nTop 5 Most Important Features:")
    for i, row in importance_df.head(5).iterrows():
        print(f"  {i+1}. {row['feature']:<35} {row['importance']:.6f} ({row['percentage']:.1f}%)")
    
    print(f"\nTop 5 features account for {importance_df.head(5)['percentage'].sum():.1f}% of importance")
    
    print(f"\nKey Insights:")
    print(f"  • {design_rules['n_rules']} design rules extracted")
    print(f"  • {len(explanations)} predictions explained in detail")
    print(f"  • Base prediction (mean): {base_value:.4f} eV/atom")
    
    print("\nOutput files:")
    print(f"  Feature importance: {output_dir}/feature_importance.csv")
    print(f"  SHAP values: {output_dir}/shap_values.npy")
    print(f"  Visualizations: {output_dir}/shap_*.png")
    print(f"  Explanations: {output_dir}/prediction_explanations.json")
    print("="*70 + "\n")
    
    return shap_values, importance_df, design_rules


def extract_design_rules(
    shap_values: np.ndarray,
    features: np.ndarray,
    targets: np.ndarray,
    feature_names: list,
    importance_df: pd.DataFrame
) -> dict:
    """
    Extract interpretable design rules from SHAP analysis.
    
    Args:
        shap_values: SHAP values [n_samples, n_features]
        features: Feature values [n_samples, n_features]
        targets: Target values [n_samples]
        feature_names: Feature names
        importance_df: Feature importance dataframe
    
    Returns:
        Dictionary of design rules
    """
    print("\n  Extracting design rules from SHAP analysis...")
    
    rules = []
    
    # Get stable vs unstable samples
    stable_mask = targets <= 0.025
    unstable_mask = targets > 0.1
    
    # Analyze top features
    top_features = importance_df.head(10)
    
    for i, row in top_features.iterrows():
        feature = row['feature']
        feat_idx = feature_names.index(feature)
        
        # Get SHAP values for this feature
        shap_feat = shap_values[:, feat_idx]
        feat_vals = features[:, feat_idx]
        
        # Correlation between feature value and SHAP value
        correlation = np.corrcoef(feat_vals, shap_feat)[0, 1]
        
        # Compare stable vs unstable
        stable_mean = feat_vals[stable_mask].mean() if stable_mask.sum() > 0 else 0
        unstable_mean = feat_vals[unstable_mask].mean() if unstable_mask.sum() > 0 else 0
        
        # Determine direction
        if correlation > 0.3:
            direction = "increase"
            effect = "increases"
        elif correlation < -0.3:
            direction = "decrease"
            effect = "decreases"
        else:
            direction = "affects"
            effect = "affects"
        
        # Create rule
        if abs(stable_mean - unstable_mean) > 0.1 * abs(unstable_mean):
            if stable_mean > unstable_mean:
                rule = {
                    'feature': feature,
                    'importance': float(row['importance']),
                    'description': f"Higher {feature} promotes stability",
                    'impact': f"{direction.capitalize()} in {feature} {effect} decomposition energy",
                    'evidence': f"Stable materials have {stable_mean:.3f} vs {unstable_mean:.3f} for unstable",
                    'correlation': float(correlation),
                    'stable_mean': float(stable_mean),
                    'unstable_mean': float(unstable_mean)
                }
            else:
                rule = {
                    'feature': feature,
                    'importance': float(row['importance']),
                    'description': f"Lower {feature} promotes stability",
                    'impact': f"{direction.capitalize()} in {feature} {effect} decomposition energy",
                    'evidence': f"Stable materials have {stable_mean:.3f} vs {unstable_mean:.3f} for unstable",
                    'correlation': float(correlation),
                    'stable_mean': float(stable_mean),
                    'unstable_mean': float(unstable_mean)
                }
            
            rules.append(rule)
    
    print(f"    Extracted {len(rules)} design rules")
    
    design_rules = {
        'n_rules': len(rules),
        'rules': rules,
        'summary': {
            'n_stable_samples': int(stable_mask.sum()),
            'n_unstable_samples': int(unstable_mask.sum()),
            'most_important_feature': feature_names[np.argmax(np.abs(shap_values).mean(axis=0))],
            'top_10_importance_coverage': float(top_features['percentage'].sum())
        }
    }
    
    return design_rules

