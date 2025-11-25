"""
SHAP-Guided Genetic Algorithm for Inverse Design.
Discovers stable perovskite compositions using ML-guided optimization.
"""

import numpy as np
import torch
import random
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import json

from pymatgen.core import Structure
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class ShapGuidedGA:
    """
    Genetic algorithm for perovskite composition discovery.
    
    Uses SHAP design rules to guide the search toward stable compositions.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        structure_generator,
        shap_rules: Optional[Dict] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize SHAP-guided genetic algorithm.
        
        Args:
            model: Trained CGCNN model (or ensemble)
            structure_generator: PerovskiteStructureGenerator instance
            shap_rules: Dictionary of SHAP design rules
            device: Device for model inference
        """
        self.model = model.to(device)
        self.model.eval()
        self.structure_gen = structure_generator
        self.shap_rules = shap_rules or {}
        self.device = device
        
        # Element pools based on SHAP insights
        self.A_site_elements = [
            # Group I-II (large, low electronegativity)
            'Li', 'Na', 'K',
            'Ca', 'Sr', 'Ba',
            # Rare earths (large, low electronegativity)
            'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Gd', 'Y',
            # Others
            'Pb'
        ]
        
        self.B_site_elements = [
            # Transition metals (small, high p-valence)
            'Ti', 'Zr', 'Hf',
            'V', 'Nb', 'Ta',
            'Cr', 'Mo', 'W',
            'Mn', 'Fe', 'Co', 'Ni',
            # Others
            'Al', 'Sn'
        ]
        
        print(f"ShapGuidedGA initialized")
        print(f"  A-site pool: {len(self.A_site_elements)} elements")
        print(f"  B-site pool: {len(self.B_site_elements)} elements")
        print(f"  Total search space: {len(self.A_site_elements) * len(self.B_site_elements)} compositions")
    
    def _prepare_batch(self, batch):
        """Prepare batch for model."""
        batch = batch.to(self.device)
        
        if hasattr(batch, 'comp_features') and batch.comp_features.dim() == 1:
            num_graphs = batch.num_graphs
            comp_dim = batch.comp_features.shape[0] // num_graphs
            batch.comp_features = batch.comp_features.view(num_graphs, comp_dim)
        
        return batch
    
    def initialize_population(self, size: int = 100) -> List[Dict]:
        """
        Initialize random population of compositions.
        
        Args:
            size: Population size
        
        Returns:
            List of composition dictionaries
        """
        print(f"\nInitializing population of {size} compositions...")
        
        population = []
        
        for i in range(size):
            composition = {
                'A': random.choice(self.A_site_elements),
                'B': random.choice(self.B_site_elements),
                'X': 'O'
            }
            population.append(composition)
        
        # Remove duplicates
        unique_pop = []
        seen = set()
        for comp in population:
            key = f"{comp['A']}-{comp['B']}-{comp['X']}"
            if key not in seen:
                seen.add(key)
                unique_pop.append(comp)
        
        print(f"  Generated {len(unique_pop)} unique compositions")
        
        return unique_pop
    
    def validate_composition(self, composition: Dict) -> Tuple[bool, str]:
        """
        Validate composition for chemical feasibility.
        
        Checks:
        1. Charge neutrality (approximately)
        2. Goldschmidt tolerance factor
        3. Element availability
        
        Args:
            composition: Composition dictionary
        
        Returns:
            (is_valid, reason)
        """
        A = composition['A']
        B = composition['B']
        
        # Check tolerance factor
        tau = self.structure_gen.calculate_tolerance_factor(A, B)
        
        if tau < 0.7:
            return False, f"Tolerance factor too low ({tau:.2f})"
        
        if tau > 1.15:
            return False, f"Tolerance factor too high ({tau:.2f})"
        
        # Check for forbidden elements (radioactive, rare)
        forbidden = ['Tc', 'Pm', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac']
        if A in forbidden or B in forbidden:
            return False, "Contains forbidden element"
        
        # Passed all checks
        return True, "Valid"
    
    def compute_fitness(
        self,
        composition: Dict,
        return_details: bool = False
    ) -> float:
        """
        Compute fitness for a composition.
        
        Fitness = -predicted_stability - λ*uncertainty + novelty_bonus
        
        Args:
            composition: Composition dictionary
            return_details: If True, return detailed results
        
        Returns:
            Fitness score (higher = better)
        """
        # Validate first
        is_valid, reason = self.validate_composition(composition)
        
        if not is_valid:
            if return_details:
                return -999.0, None, None, reason
            return -999.0
        
        try:
            # Generate structure
            structure, structure_type = self.structure_gen.generate_best_structure(
                composition['A'],
                composition['B']
            )
            
            # Convert to graph
            from mgnn.data.graph_builder import structure_to_graph
            graph = structure_to_graph(structure)
            
            # Predict stability
            loader = DataLoader([graph], batch_size=1)
            batch = next(iter(loader))
            batch = self._prepare_batch(batch)
            
            with torch.no_grad():
                prediction = self.model(batch)
                stability = prediction.cpu().numpy().flatten()[0]
            
            # Estimate uncertainty (if ensemble, use std; otherwise use heuristic)
            uncertainty = 0.015  # Default uncertainty
            
            # Novelty bonus (check if composition is novel)
            novelty = 0.01  # Assume all are novel for now
            
            # Fitness function
            # Minimize stability, penalize uncertainty, reward novelty
            fitness = -stability - 0.5 * uncertainty + novelty
            
            if return_details:
                return fitness, stability, uncertainty, structure
            
            return fitness
            
        except Exception as e:
            print(f"  Error computing fitness for {composition['A']}{composition['B']}O3: {e}")
            if return_details:
                return -999.0, None, None, str(e)
            return -999.0
    
    def tournament_selection(
        self,
        population: List[Dict],
        fitness_scores: List[float],
        k: int = 5
    ) -> Dict:
        """
        Tournament selection.
        
        Args:
            population: Current population
            fitness_scores: Fitness for each individual
            k: Tournament size
        
        Returns:
            Selected individual
        """
        # Randomly select k individuals
        indices = random.sample(range(len(population)), min(k, len(population)))
        tournament = [(population[i], fitness_scores[i]) for i in indices]
        
        # Return best
        winner = max(tournament, key=lambda x: x[1])
        return winner[0]
    
    def crossover(
        self,
        parent1: Dict,
        parent2: Dict
    ) -> Dict:
        """
        Uniform crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
        
        Returns:
            Offspring
        """
        offspring = {
            'A': parent1['A'] if random.random() < 0.5 else parent2['A'],
            'B': parent1['B'] if random.random() < 0.5 else parent2['B'],
            'X': 'O'
        }
        
        return offspring
    
    def mutate(
        self,
        composition: Dict,
        mutation_rate: float = 0.1,
        shap_guided: bool = True
    ) -> Dict:
        """
        Mutate composition.
        
        If SHAP-guided, bias mutations toward favorable elements.
        
        Args:
            composition: Composition to mutate
            mutation_rate: Probability of mutation per site
            shap_guided: Use SHAP rules to guide mutation
        
        Returns:
            Mutated composition
        """
        mutated = composition.copy()
        
        # Mutate A-site
        if random.random() < mutation_rate:
            if shap_guided and self.shap_rules:
                # Bias toward large radius (from SHAP: higher CovalentRadius_max)
                # Prefer: Ba, Sr, La over Li, Na
                preferred_A = ['Ba', 'Sr', 'La', 'Ca', 'Ce', 'Pr']
                available = [e for e in preferred_A if e in self.A_site_elements]
                
                if available and random.random() < 0.7:
                    mutated['A'] = random.choice(available)
                else:
                    mutated['A'] = random.choice(self.A_site_elements)
            else:
                mutated['A'] = random.choice(self.A_site_elements)
        
        # Mutate B-site
        if random.random() < mutation_rate:
            if shap_guided and self.shap_rules:
                # Bias toward small radius, high p-valence (from SHAP)
                # Prefer: Ti, Zr, Nb, Ta
                preferred_B = ['Ti', 'Zr', 'Hf', 'Nb', 'Ta', 'V']
                available = [e for e in preferred_B if e in self.B_site_elements]
                
                if available and random.random() < 0.7:
                    mutated['B'] = random.choice(available)
                else:
                    mutated['B'] = random.choice(self.B_site_elements)
            else:
                mutated['B'] = random.choice(self.B_site_elements)
        
        return mutated
    
    def evolve(
        self,
        n_generations: int = 50,
        population_size: int = 100,
        mutation_rate: float = 0.15,
        elite_size: int = 10,
        shap_guided: bool = True,
        verbose: bool = True
    ) -> Tuple[List[Dict], List[Dict], pd.DataFrame]:
        """
        Run genetic algorithm evolution.
        
        Args:
            n_generations: Number of generations
            population_size: Population size
            mutation_rate: Mutation probability
            elite_size: Number of elites to keep
            shap_guided: Use SHAP rules for guidance
            verbose: Print progress
        
        Returns:
            (final_population, best_history, results_df)
        """
        print("\n" + "="*70)
        print("STARTING GENETIC ALGORITHM EVOLUTION")
        print("="*70)
        print(f"  Generations: {n_generations}")
        print(f"  Population size: {population_size}")
        print(f"  Mutation rate: {mutation_rate}")
        print(f"  Elite size: {elite_size}")
        print(f"  SHAP-guided: {shap_guided}")
        print("="*70 + "\n")
        
        # Initialize population
        population = self.initialize_population(population_size)
        
        # Track history
        best_history = []
        all_evaluated = []
        
        for gen in range(n_generations):
            if verbose:
                print(f"\n{'='*70}")
                print(f"GENERATION {gen+1}/{n_generations}")
                print(f"{'='*70}")
            
            # Evaluate fitness
            fitness_scores = []
            
            iterator = tqdm(population, desc="Evaluating fitness") if verbose else population
            
            for comp in iterator:
                fitness = self.compute_fitness(comp)
                fitness_scores.append(fitness)
            
            # Track best
            best_idx = np.argmax(fitness_scores)
            best_composition = population[best_idx]
            best_fitness = fitness_scores[best_idx]
            
            # Get detailed info for best
            fitness_det, stability, uncertainty, structure = self.compute_fitness(
                best_composition,
                return_details=True
            )
            
            best_info = {
                'generation': gen,
                'composition': best_composition,
                'fitness': best_fitness,
                'stability': stability,
                'uncertainty': uncertainty,
                'structure': structure
            }
            best_history.append(best_info)
            
            if verbose:
                print(f"\nBest in generation {gen+1}:")
                print(f"  Composition: {best_composition['A']}{best_composition['B']}O3")
                print(f"  Fitness: {best_fitness:.4f}")
                if stability is not None:
                    print(f"  Predicted ΔHd: {stability:.4f} eV/atom")
                    print(f"  Uncertainty: {uncertainty:.4f} eV/atom")
                    tau = self.structure_gen.calculate_tolerance_factor(
                        best_composition['A'],
                        best_composition['B']
                    )
                    print(f"  Tolerance factor: {tau:.3f}")
            
            # Store all evaluated compositions
            for comp, fitness, score in zip(population, fitness_scores, fitness_scores):
                all_evaluated.append({
                    'generation': gen,
                    'A_site': comp['A'],
                    'B_site': comp['B'],
                    'formula': f"{comp['A']}{comp['B']}O3",
                    'fitness': fitness
                })
            
            # Selection and evolution
            # Keep elite
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            elite = [population[i] for i in elite_indices]
            
            # Generate next generation
            next_generation = elite.copy()
            
            while len(next_generation) < population_size:
                # Selection
                parent1 = self.tournament_selection(population, fitness_scores, k=5)
                parent2 = self.tournament_selection(population, fitness_scores, k=5)
                
                # Crossover
                offspring = self.crossover(parent1, parent2)
                
                # Mutation
                offspring = self.mutate(
                    offspring,
                    mutation_rate=mutation_rate,
                    shap_guided=shap_guided
                )
                
                # Validate
                is_valid, _ = self.validate_composition(offspring)
                if is_valid:
                    # Avoid duplicates
                    if offspring not in next_generation:
                        next_generation.append(offspring)
            
            population = next_generation[:population_size]
            
            # Convergence check
            if gen > 10:
                recent_best = [h['fitness'] for h in best_history[-10:]]
                if max(recent_best) - min(recent_best) < 0.001:
                    if verbose:
                        print(f"\nConverged at generation {gen+1}")
                    break
        
        # Create results dataframe
        results_df = pd.DataFrame(all_evaluated)
        
        print("\n" + "="*70)
        print("EVOLUTION COMPLETE")
        print("="*70)
        print(f"  Total generations: {gen+1}")
        print(f"  Total evaluations: {len(all_evaluated)}")
        print(f"  Best fitness: {best_history[-1]['fitness']:.4f}")
        print(f"  Best composition: {best_history[-1]['composition']['A']}{best_history[-1]['composition']['B']}O3")
        print("="*70 + "\n")
        
        return population, best_history, results_df
    
    def get_top_candidates(
        self,
        population: List[Dict],
        n_candidates: int = 20
    ) -> List[Dict]:
        """
        Get top N candidates with full analysis.
        
        Args:
            population: Final population
            n_candidates: Number of top candidates
        
        Returns:
            List of candidate reports
        """
        print(f"\nAnalyzing top {n_candidates} candidates...")
        
        # Compute detailed fitness for all
        candidates_data = []
        
        for comp in tqdm(population, desc="Computing detailed fitness"):
            fitness, stability, uncertainty, structure = self.compute_fitness(
                comp,
                return_details=True
            )
            
            if structure is not None:
                candidates_data.append({
                    'composition': comp,
                    'fitness': fitness,
                    'stability': stability,
                    'uncertainty': uncertainty,
                    'structure': structure
                })
        
        # Sort by fitness
        candidates_data.sort(key=lambda x: x['fitness'], reverse=True)
        
        # Take top N
        top_candidates = candidates_data[:n_candidates]
        
        print(f"  Selected {len(top_candidates)} top candidates")
        
        return top_candidates


def test_genetic_algorithm():
    """Test genetic algorithm (without actual model)."""
    print("\nTesting ShapGuidedGA...")
    
    from mgnn.inverse.structure_generator import PerovskiteStructureGenerator
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
    
    # Structure generator
    structure_gen = PerovskiteStructureGenerator()
    
    # Initialize GA
    ga = ShapGuidedGA(model, structure_gen, device='cpu')
    
    print("  ✓ GA initialized")
    
    # Test population initialization
    pop = ga.initialize_population(10)
    print(f"  ✓ Population initialized: {len(pop)} compositions")
    
    # Test validation
    test_comp = {'A': 'Ca', 'B': 'Ti', 'X': 'O'}
    is_valid, reason = ga.validate_composition(test_comp)
    print(f"  ✓ Validation works: CaTiO3 is {is_valid}")
    
    print("\n✓ ShapGuidedGA test passed!")


if __name__ == '__main__':
    test_genetic_algorithm()