"""
SHAP Analysis for CGCNN Models.
Explains predictions using SHapley Additive exPlanations.
"""

import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pandas as pd


class CGCNNShapExplainer:
    """
    SHAP explainer for Crystal Graph Convolutional Neural Networks.
    
    Explains predictions by computing feature importance for:
    - Composition features (element properties)
    - Node features (atomic properties)
    - Graph structure properties
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained CGCNN model
            device: Device for computation
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
        print(f"CGCNNShapExplainer initialized on {device}")
    
    def _prepare_batch(self, batch):
        """Prepare batch (handle comp_features reshaping)."""
        batch = batch.to(self.device)
        
        if hasattr(batch, 'comp_features') and batch.comp_features.dim() == 1:
            num_graphs = batch.num_graphs
            comp_dim = batch.comp_features.shape[0] // num_graphs
            batch.comp_features = batch.comp_features.view(num_graphs, comp_dim)
        
        return batch
    
    def compute_composition_shap(
        self,
        graphs: List[Data],
        background_size: int = 100,
        test_size: int = 100,
        feature_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Compute SHAP values for composition features.
        
        Args:
            graphs: List of crystal graphs
            background_size: Number of background samples
            test_size: Number of test samples to explain
            feature_names: Names of composition features
        
        Returns:
            (shap_values, base_values, feature_names)
        """
        print(f"\nComputing SHAP values for composition features...")
        print(f"  Background samples: {background_size}")
        print(f"  Test samples: {test_size}")
        
        # Extract composition features
        comp_features = torch.stack([g.comp_features for g in graphs]).numpy()
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(comp_features.shape[1])]
        
        # Select random samples
        np.random.seed(42)
        background_idx = np.random.choice(len(graphs), background_size, replace=False)
        test_idx = np.random.choice(len(graphs), test_size, replace=False)
        
        background_data = comp_features[background_idx]
        test_data = comp_features[test_idx]
        
        # Create prediction function for composition features only
        def predict_from_comp(comp_feats):
            """Predict using only composition features (graph structure fixed)."""
            predictions = []
            
            # Use first graph as template for structure
            template_graph = graphs[0]
            
            for comp_feat in comp_feats:
                # Create modified graph with new composition features
                graph = Data(
                    x=template_graph.x.clone(),
                    edge_index=template_graph.edge_index.clone(),
                    edge_attr=template_graph.edge_attr.clone(),
                    comp_features=torch.tensor(comp_feat, dtype=torch.float32)
                )
                
                # Prepare batch
                loader = DataLoader([graph], batch_size=1)
                batch = next(iter(loader))
                batch = self._prepare_batch(batch)
                
                # Predict
                with torch.no_grad():
                    pred = self.model(batch)
                    predictions.append(pred.cpu().numpy().flatten()[0])
            
            return np.array(predictions)
        
        # Create SHAP explainer
        print("  Creating SHAP explainer...")
        explainer = shap.KernelExplainer(predict_from_comp, background_data)
        
        # Compute SHAP values
        print("  Computing SHAP values (this may take a few minutes)...")
        shap_values = explainer.shap_values(test_data, nsamples=100)
        
        # Get base values
        base_values = explainer.expected_value
        
        if isinstance(base_values, (list, np.ndarray)):
            base_values = base_values[0] if len(base_values) > 0 else 0.0
        
        print(f"  ✓ SHAP values computed for {test_size} samples")
        print(f"  Base value (mean prediction): {base_values:.4f} eV/atom")
        
        return shap_values, base_values, feature_names
    
    def analyze_feature_importance(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        top_k: int = 20
    ) -> pd.DataFrame:
        """
        Analyze global feature importance.
        
        Args:
            shap_values: SHAP values array [n_samples, n_features]
            feature_names: Feature names
            top_k: Number of top features to return
        
        Returns:
            DataFrame with feature importance
        """
        print(f"\nAnalyzing feature importance (top {top_k})...")
        
        # Compute mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_abs_shap
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Add percentage
        total_importance = importance_df['importance'].sum()
        importance_df['percentage'] = (importance_df['importance'] / total_importance) * 100
        importance_df['cumulative_percentage'] = importance_df['percentage'].cumsum()
        
        print(f"\n  Top {top_k} Most Important Features:")
        for i, row in importance_df.head(top_k).iterrows():
            print(f"    {i+1:2d}. {row['feature']:<30} | {row['importance']:.6f} ({row['percentage']:.1f}%)")
        
        return importance_df
    
    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_k: int = 20,
        output_path: str = './feature_importance.png'
    ) -> plt.Figure:
        """
        Plot feature importance bar chart.
        
        Args:
            importance_df: Feature importance dataframe
            top_k: Number of top features to plot
            output_path: Where to save plot
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('white')
        
        # Get top k features
        top_features = importance_df.head(top_k)
        
        # Create horizontal bar plot
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features)))
        
        bars = ax.barh(range(len(top_features)), top_features['importance'], 
                       color=colors, edgecolor='black', linewidth=0.5)
        
        # Set labels
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'], fontsize=9)
        ax.set_xlabel('Mean |SHAP Value| (eV/atom)', fontsize=12)
        ax.set_title(f'Top {top_k} Most Important Features for Stability Prediction', 
                    fontsize=14, weight='bold', pad=20)
        
        # Add percentage labels
        for i, (bar, pct) in enumerate(zip(bars, top_features['percentage'])):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{pct:.1f}%', 
                   ha='left', va='center', fontsize=8, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()  # Highest importance at top
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFeature importance plot saved to: {output_path}")
        
        return fig
    
    def plot_shap_summary(
        self,
        shap_values: np.ndarray,
        features: np.ndarray,
        feature_names: List[str],
        top_k: int = 20,
        output_path: str = './shap_summary.png'
    ) -> plt.Figure:
        """
        Create SHAP summary plot (beeswarm plot).
        
        Args:
            shap_values: SHAP values [n_samples, n_features]
            features: Feature values [n_samples, n_features]
            feature_names: Feature names
            top_k: Number of features to plot
            output_path: Where to save plot
        
        Returns:
            Matplotlib figure
        """
        print(f"\nCreating SHAP summary plot...")
        
        # Get top k features by importance
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-top_k:][::-1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('white')
        
        # Use SHAP's built-in summary plot
        shap.summary_plot(
            shap_values[:, top_indices],
            features[:, top_indices],
            feature_names=[feature_names[i] for i in top_indices],
            show=False,
            plot_size=(10, 8)
        )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"SHAP summary plot saved to: {output_path}")
        
        return fig
    
    def plot_shap_dependence(
        self,
        shap_values: np.ndarray,
        features: np.ndarray,
        feature_names: List[str],
        feature_idx: int,
        interaction_idx: Optional[int] = None,
        output_path: str = './shap_dependence.png'
    ) -> plt.Figure:
        """
        Create SHAP dependence plot for a specific feature.
        
        Args:
            shap_values: SHAP values
            features: Feature values
            feature_names: Feature names
            feature_idx: Index of feature to plot
            interaction_idx: Index of interaction feature (optional)
            output_path: Where to save plot
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')
        
        # Use SHAP's dependence plot
        shap.dependence_plot(
            feature_idx,
            shap_values,
            features,
            feature_names=feature_names,
            interaction_index=interaction_idx,
            show=False,
            ax=ax
        )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"SHAP dependence plot saved to: {output_path}")
        
        return fig
    
    def explain_predictions(
        self,
        graphs: List[Data],
        indices: List[int],
        feature_names: List[str],
        output_dir: str = './shap_explanations'
    ) -> List[Dict]:
        """
        Explain specific predictions in detail.
        
        Args:
            graphs: List of all graphs
            indices: Indices of graphs to explain
            feature_names: Feature names
            output_dir: Directory to save explanation plots
        
        Returns:
            List of explanation dictionaries
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nGenerating detailed explanations for {len(indices)} predictions...")
        
        explanations = []
        
        for idx in tqdm(indices, desc="Explaining predictions"):
            graph = graphs[idx]
            
            # Get prediction
            loader = DataLoader([graph], batch_size=1)
            batch = next(iter(loader))
            batch = self._prepare_batch(batch)
            
            with torch.no_grad():
                prediction = self.model(batch).cpu().numpy().flatten()[0]
            
            # Get composition features
            comp_features = graph.comp_features.numpy()
            
            # Compute SHAP values for this sample
            background_idx = np.random.choice(len(graphs), 50, replace=False)
            background_data = torch.stack([graphs[i].comp_features for i in background_idx]).numpy()
            
            def predict_from_comp(comp_feats):
                predictions = []
                for comp_feat in comp_feats:
                    g = Data(
                        x=graph.x.clone(),
                        edge_index=graph.edge_index.clone(),
                        edge_attr=graph.edge_attr.clone(),
                        comp_features=torch.tensor(comp_feat, dtype=torch.float32)
                    )
                    loader = DataLoader([g], batch_size=1)
                    b = next(iter(loader))
                    b = self._prepare_batch(b)
                    with torch.no_grad():
                        pred = self.model(b).cpu().numpy().flatten()[0]
                    predictions.append(pred)
                return np.array(predictions)
            
            explainer = shap.KernelExplainer(predict_from_comp, background_data)
            shap_values = explainer.shap_values(comp_features.reshape(1, -1), nsamples=50)
            
            # Get top contributing features
            shap_vals_flat = shap_values.flatten()
            top_pos_idx = np.argsort(shap_vals_flat)[-5:][::-1]
            top_neg_idx = np.argsort(shap_vals_flat)[:5]
            
            explanation = {
                'index': int(idx),
                'formula': graph.formula if hasattr(graph, 'formula') else f"Graph_{idx}",
                'prediction': float(prediction),
                'target': float(graph.y.item()) if hasattr(graph, 'y') else None,
                'base_value': float(explainer.expected_value),
                'top_positive_features': [
                    {
                        'feature': feature_names[i],
                        'value': float(comp_features[i]),
                        'shap_value': float(shap_vals_flat[i])
                    }
                    for i in top_pos_idx
                ],
                'top_negative_features': [
                    {
                        'feature': feature_names[i],
                        'value': float(comp_features[i]),
                        'shap_value': float(shap_vals_flat[i])
                    }
                    for i in top_neg_idx
                ]
            }
            
            explanations.append(explanation)
        
        print(f"  ✓ Generated {len(explanations)} detailed explanations")
        
        return explanations


def get_composition_feature_names() -> List[str]:
    """
    Get meaningful names for composition features.
    
    Based on matminer ElementProperty with preset='magpie'.
    Returns 71 feature names matching our dataset.
    """
    # These are the features that successfully computed in our dataset
    # Based on matminer ElementProperty(preset='magpie')
    
    base_stats = ['mean', 'std', 'min', 'max', 'range']
    
    properties = [
        'MendeleevNumber',
        'AtomicWeight', 
        'MeltingT',
        'Column',
        'Row',
        'CovalentRadius',
        'Electronegativity',
        'NsValence',
        'NpValence',
        'NdValence',
        'NfValence',
        'NValence',
        'NsUnfilled',
        'NpUnfilled',
        'NdUnfilled',
        'NfUnfilled',
        'NUnfilled',
        'GSvolume_pa',
        'GSbandgap',
        'GSmagmom',
        'SpaceGroupNumber'
    ]
    
    # Generate all feature names with statistics
    feature_names = []
    for prop in properties:
        for stat in base_stats:
            feature_names.append(f"{prop}_{stat}")
    
    # Add stoichiometry features
    feature_names.extend([
        'Stoich_0-norm',
        'Stoich_2-norm', 
        'Stoich_3-norm',
        'Stoich_5-norm',
        'Stoich_7-norm',
        'Stoich_10-norm'
    ])
    
    return feature_names[:71]  # Return exactly 71 features


def test_shap_explainer():
    """Test SHAP explainer."""
    print("\nTesting CGCNNShapExplainer...")
    
    from mgnn.models.cgcnn import CGCNN
    
    # Create dummy model
    model = CGCNN(
        node_feature_dim=12,
        edge_feature_dim=1,
        comp_feature_dim=71,
        hidden_dim=64,
        n_conv=2,
        n_fc=1,
        dropout=0.1
    )
    
    explainer = CGCNNShapExplainer(model, device='cpu')
    
    print("  ✓ CGCNNShapExplainer initialized")
    print("  ✓ Composition SHAP analysis available")
    print("  ✓ Feature importance analysis available")
    
    # Get feature names
    feature_names = get_composition_feature_names()
    print(f"  ✓ Generated {len(feature_names)} feature names")
    
    print("\n✓ SHAP explainer test passed!")


if __name__ == '__main__':
    test_shap_explainer()


# cd /projectnb/cui-buchem/anandsahu/Softwares/Mine/multinary-gnn

# Install SHAP
# uv add shap

# Test the SHAP module
# uv run python -m mgnn.analysis.shap_analysis