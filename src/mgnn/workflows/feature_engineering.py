"""
Feature Engineering Workflow.
Purpose: Convert structures to graph representations with features.
"""

import json
import pickle
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from mgnn.config_manager import ConfigManager
from mgnn.data_manager import DataManager
from mgnn.data.loader import load_pickle_data, create_dataframe, filter_dataframe
from mgnn.features.graph_constructor import CrystalGraphConstructor
from mgnn.features.composition_features import CompositionFeaturizer


def run(config: ConfigManager, dm: DataManager):
    """
    Run feature engineering workflow.
    
    Steps:
    1. Load filtered data
    2. Initialize featurizers
    3. Convert structures to graphs
    4. Extract composition features
    5. Normalize features
    6. Split train/val/test
    7. Save processed data
    """
    
    print("\n" + "="*70)
    print("FEATURE ENGINEERING WORKFLOW")
    print("="*70 + "\n")
    
    # Step 1: Load data
    print("[STEP 1/7] Loading data...")
    pickle_file = config.get('PICKLE_FILE', required=True, vartype='file')
    data_list = load_pickle_data(pickle_file)
    df = create_dataframe(data_list, show_progress=False)
    df_filtered, _ = filter_dataframe(df, perovskites_only=True)
    
    print(f"  Loaded {len(df_filtered)} compounds")
    
    # Step 2: Initialize featurizers
    print("\n[STEP 2/7] Initializing featurizers...")
    
    graph_constructor = None
    if config.get('USE_STRUCTURE_GRAPHS', default=True):
        graph_constructor = CrystalGraphConstructor(
            cutoff_radius=config.get('GRAPH_CUTOFF_RADIUS', default=5.0),
            max_neighbors=config.get('MAX_NEIGHBORS', default=12)
        )
    
    comp_featurizer = None
    if config.get('USE_COMPOSITIONAL_FEATURES', default=True):
        preset = config.get('COMPOSITIONAL_FEATURIZER', default='magpie')
        comp_featurizer = CompositionFeaturizer(preset=preset)
    
    # Step 3: Convert structures to graphs
    print("\n[STEP 3/7] Converting structures to graphs...")
    
    graphs = []
    comp_features = []
    targets = []
    formulas = []
    unique_ids = []
    
    failed_count = 0
    
    for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Processing"):
        try:
            structure = row['Structure']
            formula = row['Formula']
            unique_id = row['Unique_ID']
            target = row['dH_decomposition_ev']
            
            # Extract oxidation states
            oxidation_states = {
                0: row['Oxidation_a1'],
                1: row['Oxidation_a2'],
                2: row['Oxidation_b1'],
                3: row['Oxidation_b2'],
                4: -2.0,  # Oxygen
                5: -2.0   # Oxygen
            }
            
            # Build graph
            if graph_constructor:
                graph = graph_constructor.structure_to_graph(
                    structure,
                    oxidation_states=oxidation_states,
                    property_label=target
                )
                graphs.append(graph)
            
            # Extract composition features
            if comp_featurizer:
                comp_feat = comp_featurizer.featurize(structure.composition)
                comp_features.append(comp_feat)
            
            # Store metadata
            targets.append(target)
            formulas.append(formula)
            unique_ids.append(unique_id)
            
        except Exception as e:
            failed_count += 1
            if failed_count <= 5:
                print(f"  Warning: Failed to process {row.get('Formula', 'unknown')}: {e}")
    
    print(f"  Successfully processed: {len(graphs)} compounds")
    print(f"  Failed: {failed_count} compounds")
    
    # Step 4: Add composition features to graphs
    if comp_featurizer and graphs:
        print("\n[STEP 4/7] Adding composition features to graphs...")
        comp_features_array = np.array(comp_features, dtype=np.float32)
        
        for i, graph in enumerate(graphs):
            graph.comp_features = torch.tensor(comp_features_array[i], dtype=torch.float)
    else:
        print("\n[STEP 4/7] Skipping composition features")
    
    # Step 5: Normalize features
    print("\n[STEP 5/7] Normalizing features...")
    
    if config.get('NORMALIZE_FEATURES', default=True):
        method = config.get('NORMALIZATION_METHOD', default='standard')
        
        # Collect all node features
        all_node_features = torch.cat([g.x for g in graphs], dim=0).numpy()
        
        # Fit scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        scaler.fit(all_node_features)
        
        # Transform features in each graph
        for graph in graphs:
            graph.x = torch.tensor(
                scaler.transform(graph.x.numpy()),
                dtype=torch.float
            )
        
        # Save scaler
        dm.save({
            "features/node_feature_scaler": {
                "data": pickle.dumps(scaler),
                "metadata": {
                    "method": method,
                    "n_features": all_node_features.shape[1]
                }
            }
        })
        
        print(f"  Normalized node features using {method} scaler")
        
        # Normalize composition features if present
        if comp_featurizer:
            comp_scaler = StandardScaler()
            comp_scaler.fit(comp_features_array)
            
            for graph in graphs:
                graph.comp_features = torch.tensor(
                    comp_scaler.transform(graph.comp_features.numpy().reshape(1, -1)).flatten(),
                    dtype=torch.float
                )
            
            dm.save({
                "features/comp_feature_scaler": {
                    "data": pickle.dumps(comp_scaler),
                    "metadata": {
                        "method": "standard",
                        "n_features": comp_features_array.shape[1]
                    }
                }
            })
            
            print(f"  Normalized composition features")
    
    # Step 6: Train/val/test split
    print("\n[STEP 6/7] Splitting data...")
    
    train_ratio = config.get('TRAIN_RATIO', default=0.7)
    val_ratio = config.get('VAL_RATIO', default=0.15)
    test_ratio = config.get('TEST_RATIO', default=0.15)
    random_seed = config.get('SPLIT_RANDOM_SEED', default=42)
    
    # Stratify by stability class
    stratify_labels = None
    if config.get('STRATIFY_BY_STABILITY', default=True):
        stratify_labels = np.digitize(
            targets,
            bins=[0.025, 0.1]
        )
    
    # Split train+val vs test
    indices = np.arange(len(graphs))
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=stratify_labels
    )
    
    # Split train vs val
    if stratify_labels is not None:
        stratify_train_val = stratify_labels[train_val_idx]
    else:
        stratify_train_val = None
    
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio / (train_ratio + val_ratio),
        random_state=random_seed,
        stratify=stratify_train_val
    )
    
    print(f"  Train: {len(train_idx)} ({len(train_idx)/len(graphs)*100:.1f}%)")
    print(f"  Val:   {len(val_idx)} ({len(val_idx)/len(graphs)*100:.1f}%)")
    print(f"  Test:  {len(test_idx)} ({len(test_idx)/len(graphs)*100:.1f}%)")
    
    # Step 7: Save processed data
    print("\n[STEP 7/7] Saving processed data...")
    
    # Save graphs as pickle (HDF5 doesn't handle PyG Data well)
    output_dir = Path(config.get('OUTPUT_DIR', default='./results'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'graphs_train.pkl', 'wb') as f:
        pickle.dump([graphs[i] for i in train_idx], f)
    
    with open(output_dir / 'graphs_val.pkl', 'wb') as f:
        pickle.dump([graphs[i] for i in val_idx], f)
    
    with open(output_dir / 'graphs_test.pkl', 'wb') as f:
        pickle.dump([graphs[i] for i in test_idx], f)
    
    print(f"  Saved graphs to {output_dir}")
    
    # Save metadata to HDF5
    dm.save({
        "features/metadata": {
            "data": json.dumps({
                "n_total": len(graphs),
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_test": len(test_idx),
                "node_feature_dim": graphs[0].x.shape[1],
                "edge_feature_dim": graphs[0].edge_attr.shape[1],
                "comp_feature_dim": graphs[0].comp_features.shape[0] if hasattr(graphs[0], 'comp_features') else 0,
                "cutoff_radius": config.get('GRAPH_CUTOFF_RADIUS', default=5.0),
                "max_neighbors": config.get('MAX_NEIGHBORS', default=12)
            }, indent=2),
            "metadata": {
                "n_total": len(graphs),
                "n_failed": failed_count
            }
        }
    })
    
    # Compute and save statistics
    graph_stats = {
        'num_nodes': [g.num_nodes for g in graphs],
        'num_edges': [g.num_edges for g in graphs],
        'avg_degree': [g.num_edges / g.num_nodes for g in graphs]
    }
    
    stats_summary = {
        'num_nodes_mean': float(np.mean(graph_stats['num_nodes'])),
        'num_nodes_std': float(np.std(graph_stats['num_nodes'])),
        'num_edges_mean': float(np.mean(graph_stats['num_edges'])),
        'num_edges_std': float(np.std(graph_stats['num_edges'])),
        'avg_degree_mean': float(np.mean(graph_stats['avg_degree'])),
        'avg_degree_std': float(np.std(graph_stats['avg_degree']))
    }
    
    dm.save({
        "features/graph_statistics": {
            "data": json.dumps(stats_summary, indent=2),
            "metadata": stats_summary
        }
    })
    
    print("\n" + "="*70)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*70)
    print(f"Total graphs: {len(graphs)}")
    print(f"Node features: {graphs[0].x.shape[1]}")
    print(f"Edge features: {graphs[0].edge_attr.shape[1]}")
    if hasattr(graphs[0], 'comp_features'):
        print(f"Composition features: {graphs[0].comp_features.shape[0]}")
    print(f"\nSaved to: {output_dir}")
    print("="*70 + "\n")
    
    return graphs, train_idx, val_idx, test_idx