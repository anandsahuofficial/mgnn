"""
Inverse Design Workflow.
Discover novel stable perovskite compositions using SHAP-guided GA.
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
from mgnn.inverse.structure_generator import PerovskiteStructureGenerator
from mgnn.inverse.genetic_algorithm import ShapGuidedGA


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


def plot_evolution_history(
    best_history: list,
    output_path: str
) -> plt.Figure:
    """
    Plot GA evolution history.
    
    Args:
        best_history: List of best individuals per generation
        output_path: Where to save plot
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('white')
    
    generations = [h['generation'] for h in best_history]
    fitness = [h['fitness'] for h in best_history]
    stability = [h['stability'] for h in best_history if h['stability'] is not None]
    uncertainty = [h['uncertainty'] for h in best_history if h['uncertainty'] is not None]
    
    # Plot 1: Fitness evolution
    ax = axes[0, 0]
    ax.plot(generations, fitness, 'b-o', linewidth=2, markersize=4)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.set_title('Best Fitness Evolution', fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Stability evolution
    ax = axes[0, 1]
    if stability:
        ax.plot(generations[:len(stability)], stability, 'r-o', linewidth=2, markersize=4)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Stable threshold')
        ax.axhline(y=0.025, color='orange', linestyle='--', alpha=0.5, label='Metastable threshold')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Predicted ΔHd (eV/atom)')
        ax.set_title('Best Stability Evolution', fontsize=14, weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Uncertainty evolution
    ax = axes[1, 0]
    if uncertainty:
        ax.plot(generations[:len(uncertainty)], uncertainty, 'g-o', linewidth=2, markersize=4)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Uncertainty (eV/atom)')
        ax.set_title('Best Uncertainty Evolution', fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Best compositions over time
    ax = axes[1, 1]
    best_formulas = [f"{h['composition']['A']}{h['composition']['B']}O₃" 
                     for h in best_history]
    
    # Show every 5th generation or so
    step = max(1, len(best_formulas) // 10)
    shown_gens = generations[::step]
    shown_formulas = best_formulas[::step]
    
    ax.barh(range(len(shown_formulas)), shown_gens, color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(shown_formulas)))
    ax.set_yticklabels(shown_formulas, fontsize=9)
    ax.set_xlabel('Generation')
    ax.set_title('Best Composition Evolution', fontsize=14, weight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nEvolution history plot saved to: {output_path}")
    
    return fig


def run(config: ConfigManager, dm: DataManager):
    """
    Run inverse design workflow.
    
    Steps:
    1. Load trained model
    2. Initialize structure generator
    3. Run SHAP-guided GA
    4. Analyze top candidates
    5. Generate structure files
    6. Create comprehensive reports
    """
    
    print("\n" + "="*70)
    print("🧬 INVERSE DESIGN: SHAP-GUIDED DISCOVERY 🧬")
    print("="*70)
    print("\nDiscover novel stable perovskite compositions")
    print("="*70 + "\n")
    
    # Step 1: Load model
    print("[STEP 1/6] Loading trained model...")
    
    device = config.get('DEVICE', default='cuda' if torch.cuda.is_available() else 'cpu')
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
    
    print(f"  ✓ Loaded model from: {checkpoint_path}")
    
    # Step 2: Initialize generators
    print("\n[STEP 2/6] Initializing structure generator...")
    
    structure_gen = PerovskiteStructureGenerator()
    
    # Load SHAP rules if available
    shap_rules_path = Path(config.get('OUTPUT_DIR', default='./results')) / 'shap' / 'design_rules.json'
    shap_rules = {}
    
    if shap_rules_path.exists():
        with open(shap_rules_path, 'r') as f:
            shap_rules = json.load(f)
        print(f"  ✓ Loaded SHAP design rules: {shap_rules['n_rules']} rules")
    else:
        print("  ⚠ SHAP rules not found, proceeding without guidance")
    
    # Step 3: Run GA
    print("\n[STEP 3/6] Running genetic algorithm...")
    
    ga = ShapGuidedGA(
        model=model,
        structure_generator=structure_gen,
        shap_rules=shap_rules,
        device=device
    )
    
    final_population, best_history, results_df = ga.evolve(
        n_generations=config.get('N_GENERATIONS', default=50),
        population_size=config.get('POPULATION_SIZE', default=100),
        mutation_rate=config.get('MUTATION_RATE', default=0.15),
        elite_size=config.get('ELITE_SIZE', default=10),
        shap_guided=config.get('SHAP_GUIDED', default=True),
        verbose=True
    )
    
    # Step 4: Get top candidates
    print("\n[STEP 4/6] Analyzing top candidates...")
    
    n_candidates = config.get('N_TOP_CANDIDATES', default=20)
    top_candidates = ga.get_top_candidates(final_population, n_candidates=n_candidates)
    
    # Step 5: Generate structure files
    print("\n[STEP 5/6] Generating structure files and reports...")
    
    output_dir = Path(config.get('OUTPUT_DIR', default='./results'))
    structures_dir = output_dir / 'inverse_design' / 'structures'
    structures_dir.mkdir(parents=True, exist_ok=True)
    
    candidate_reports = []
    
    for i, candidate in enumerate(top_candidates, 1):
        comp = candidate['composition']
        structure = candidate['structure']
        
        formula = f"{comp['A']}{comp['B']}O3"
        
        print(f"\n  [{i}/{len(top_candidates)}] {formula}")
        print(f"    Predicted ΔHd: {candidate['stability']:.4f} eV/atom")
        print(f"    Uncertainty: {candidate['uncertainty']:.4f} eV/atom")
        
        # Determine structure type
        structure_type = structure.get_space_group_info()[1]
        
        # Generate comprehensive report
        report = structure_gen.generate_structure_report(
            A_element=comp['A'],
            B_element=comp['B'],
            structure=structure,
            structure_type=structure_type,
            predicted_stability=candidate['stability'],
            uncertainty=candidate['uncertainty'],
            shap_contributions=None  # Could add SHAP values here
        )
        
        report['rank'] = i
        candidate_reports.append(report)
        
        # Save structure files
        saved_files = structure_gen.save_structure(
            structure=structure,
            output_dir=structures_dir,
            filename=f"rank{i:02d}_{formula}",
            formats=['cif', 'poscar']
        )
        
        # Save individual report
        report_path = structures_dir / f"rank{i:02d}_{formula}_report.json"
        with open(report_path, 'w') as f:
            json.dump(convert_to_serializable(report), f, indent=2)
        
        print(f"    Saved: {saved_files['cif'].name}, {saved_files['poscar'].name}")
    
    # Step 6: Generate plots and summary
    print("\n[STEP 6/6] Generating plots and summary...")
    
    # Evolution history plot
    fig_evolution = plot_evolution_history(
        best_history,
        output_path=str(output_dir / 'inverse_design_evolution.png')
    )
    
    dm.save({
        "inverse_design/figures/evolution": {
            "data": fig_evolution,
            "metadata": {"plot_type": "evolution_history"}
        }
    })
    
    plt.close(fig_evolution)
    
    # Save all candidate reports
    candidates_summary_path = output_dir / 'inverse_design' / 'candidates_summary.json'
    with open(candidates_summary_path, 'w') as f:
        json.dump(convert_to_serializable(candidate_reports), f, indent=2)
    
    print(f"  ✓ Saved candidates summary: {candidates_summary_path}")
    
    # Save GA results
    results_df.to_csv(output_dir / 'inverse_design' / 'ga_results.csv', index=False)
    
    # Create summary table
    summary_data = []
    for report in candidate_reports[:20]:
        summary_data.append({
            'Rank': report['rank'],
            'Formula': report['composition']['formula'],
            'Predicted_ΔHd': f"{report['predictions']['stability_eV_per_atom']:.4f}",
            'Uncertainty': f"{report['predictions']['uncertainty_eV_per_atom']:.4f}",
            'Confidence': report['predictions']['confidence_level'],
            'Tolerance_Factor': f"{report['geometric_analysis']['tolerance_factor']:.3f}",
            'Structure_Type': report['structure']['type'],
            'Space_Group': report['structure']['space_group']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'inverse_design' / 'top_candidates.csv', index=False)
    
    print(f"  ✓ Saved top candidates table: top_candidates.csv")
    
    # Final summary
    print("\n" + "="*70)
    print("🎉 INVERSE DESIGN COMPLETE! 🎉")
    print("="*70)
    
    print(f"\nDiscovery Summary:")
    print(f"  Generations run: {len(best_history)}")
    print(f"  Total compositions evaluated: {len(results_df)}")
    print(f"  Top candidates identified: {len(top_candidates)}")
    
    print(f"\nTop 5 Candidates:")
    for i, report in enumerate(candidate_reports[:5], 1):
        print(f"  {i}. {report['composition']['formula']}")
        print(f"     ΔHd = {report['predictions']['stability_eV_per_atom']:.4f} ± {report['predictions']['uncertainty_eV_per_atom']:.4f} eV/atom")
        print(f"     τ = {report['geometric_analysis']['tolerance_factor']:.3f}")
    
    print(f"\nOutput files:")
    print(f"  Structure files: {structures_dir}/")
    print(f"  Summary table: {output_dir}/inverse_design/top_candidates.csv")
    print(f"  Full reports: {output_dir}/inverse_design/candidates_summary.json")
    print(f"  Evolution plot: {output_dir}/inverse_design_evolution.png")
    print("="*70 + "\n")
    
    return top_candidates, best_history, results_df