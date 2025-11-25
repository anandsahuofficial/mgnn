"""
Exploratory Data Analysis for multinary perovskite oxides.
Pure matplotlib implementation - no seaborn, no plotly.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ExploratoryAnalysis:
    """
    Comprehensive exploratory data analysis for perovskite dataset.
    
    Uses pure matplotlib for all visualizations.
    """
    
    def __init__(self, df: pd.DataFrame, style: str = 'seaborn-v0_8-darkgrid', dpi: int = 300):
        """
        Initialize EDA analyzer.
        
        Args:
            df: DataFrame with perovskite data
            style: Matplotlib style
            dpi: Figure resolution
        """
        self.df = df
        self.dpi = dpi
        
        # Set plotting style - handle if style not available
        available_styles = plt.style.available
        if style in available_styles:
            plt.style.use(style)
        elif 'seaborn-v0_8' in available_styles:
            plt.style.use('seaborn-v0_8')
        else:
            # Use default style with custom settings
            plt.style.use('default')
        
        # Set default parameters
        mpl.rcParams['figure.dpi'] = dpi
        mpl.rcParams['savefig.dpi'] = dpi
        mpl.rcParams['font.size'] = 10
        mpl.rcParams['axes.labelsize'] = 12
        mpl.rcParams['axes.titlesize'] = 14
        mpl.rcParams['xtick.labelsize'] = 10
        mpl.rcParams['ytick.labelsize'] = 10
        mpl.rcParams['legend.fontsize'] = 10
        mpl.rcParams['figure.facecolor'] = 'white'
        mpl.rcParams['axes.facecolor'] = 'white'
        mpl.rcParams['axes.grid'] = True
        mpl.rcParams['grid.alpha'] = 0.3
        
        print(f"ExploratoryAnalysis initialized with {len(df)} compounds")
    
    def _compute_kde(self, data: np.ndarray, n_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Kernel Density Estimate using gaussian_kde.
        
        Args:
            data: 1D array of data points
            n_points: Number of points in output
        
        Returns:
            x values, density values
        """
        # Remove NaN values
        data_clean = data[~np.isnan(data)]
        
        if len(data_clean) < 2:
            return np.array([]), np.array([])
        
        # For large inputs gaussian_kde can be very slow (O(n^2) work).
        # Use a fast histogram + Gaussian smoothing approximation for large n.
        try:
            if len(data_clean) > 2000:
                bins = max(200, n_points * 3)
                counts, edges = np.histogram(data_clean, bins=bins, density=True)
                centers = (edges[:-1] + edges[1:]) / 2.0
                # Smooth the histogram to approximate a KDE
                smooth = gaussian_filter1d(counts, sigma=2.0)
                x = np.linspace(centers.min(), centers.max(), n_points)
                density = np.interp(x, centers, smooth)
                return x, density
            else:
                # Small datasets: fall back to gaussian_kde for higher fidelity
                kde = stats.gaussian_kde(data_clean)
                x_min, x_max = data_clean.min(), data_clean.max()
                x_range = x_max - x_min
                x = np.linspace(x_min - 0.1*x_range, x_max + 0.1*x_range, n_points)
                density = kde(x)
                return x, density
        except Exception:
            # Last-resort: histogram without smoothing
            counts, edges = np.histogram(data_clean, bins=n_points, density=True)
            centers = (edges[:-1] + edges[1:]) / 2.0
            return centers, counts

    def _downsample_for_plot(self, *arrays, max_points: int = 30000, random_state: Optional[int] = None):
        """Randomly downsample arrays (all must have same length).

        Returns the arrays subsampled to at most `max_points` elements.
        """
        if len(arrays) == 0:
            return arrays
        n = len(arrays[0])
        if any(len(a) != n for a in arrays):
            raise ValueError("All arrays must have the same length to downsample")
        if n <= max_points:
            return arrays
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=max_points, replace=False)
        return tuple(a[idx] for a in arrays)
    
    def plot_energy_distributions(self) -> Dict[str, plt.Figure]:
        """
        Plot distributions of formation and decomposition energies.
        Pure matplotlib implementation.
        
        Returns:
            Dictionary of figure objects
        """
        print("\nGenerating energy distribution plots...")
        
        figures = {}
        
        # Figure 1: Formation and decomposition energy distributions
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.patch.set_facecolor('white')
        
        # Formation energy histogram
        ax = axes[0, 0]
        n, bins, patches = ax.hist(self.df['dH_formation_ev'], bins=60, alpha=0.7, 
                                    color='steelblue', edgecolor='black', linewidth=0.5)
        mean_val = self.df['dH_formation_ev'].mean()
        median_val = self.df['dH_formation_ev'].median()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_val:.3f} eV')
        ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, 
                   label=f'Median: {median_val:.3f} eV')
        ax.set_xlabel('Formation Energy (eV)')
        ax.set_ylabel('Count')
        ax.set_title('Formation Energy Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Formation energy KDE
        ax = axes[0, 1]
        x_kde, density = self._compute_kde(self.df['dH_formation_ev'].values)
        if len(x_kde) > 0:
            ax.plot(x_kde, density, color='steelblue', linewidth=2)
            ax.fill_between(x_kde, 0, density, alpha=0.3, color='steelblue')
        ax.set_xlabel('Formation Energy (eV)')
        ax.set_ylabel('Density')
        ax.set_title('Formation Energy Density')
        ax.grid(True, alpha=0.3)
        
        # Decomposition energy histogram
        ax = axes[1, 0]
        n, bins, patches = ax.hist(self.df['dH_decomposition_ev'], bins=60, alpha=0.7,
                                    color='coral', edgecolor='black', linewidth=0.5)
        mean_val = self.df['dH_decomposition_ev'].mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_val:.3f} eV/atom')
        ax.axvline(0.025, color='green', linestyle=':', linewidth=2, 
                   label='Stable threshold (0.025)')
        ax.axvline(0.1, color='orange', linestyle=':', linewidth=2, 
                   label='Metastable threshold (0.1)')
        ax.set_xlabel('Decomposition Energy (eV/atom)')
        ax.set_ylabel('Count')
        ax.set_title('Decomposition Energy Distribution')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Decomposition energy KDE
        ax = axes[1, 1]
        x_kde, density = self._compute_kde(self.df['dH_decomposition_ev'].values)
        if len(x_kde) > 0:
            ax.plot(x_kde, density, color='coral', linewidth=2)
            ax.fill_between(x_kde, 0, density, alpha=0.3, color='coral')
        ax.axvline(0.025, color='green', linestyle=':', linewidth=2, alpha=0.7)
        ax.axvline(0.1, color='orange', linestyle=':', linewidth=2, alpha=0.7)
        ax.set_xlabel('Decomposition Energy (eV/atom)')
        ax.set_ylabel('Density')
        ax.set_title('Decomposition Energy Density')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        figures['energy_distributions'] = fig
        
        # Figure 2: Energy correlation
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('white')
        
        # Downsample scatter points for large datasets to speed up rendering
        x = self.df['dH_formation_ev'].values
        y = self.df['dH_decomposition_ev'].values
        c = self.df['tolerance_t'].values
        x_ds, y_ds, c_ds = self._downsample_for_plot(x, y, c)
        scatter = ax.scatter(x_ds, y_ds, c=c_ds, cmap='viridis', alpha=0.5,
                     s=10, edgecolors='none', rasterized=(len(x_ds) > 20000))
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Tolerance Factor (t)', rotation=270, labelpad=20)
        
        # Add stability regions
        ax.axhline(0.025, color='green', linestyle=':', linewidth=1.5, alpha=0.5, 
                   label='Stable')
        ax.axhline(0.1, color='orange', linestyle=':', linewidth=1.5, alpha=0.5, 
                   label='Metastable')
        
        ax.set_xlabel('Formation Energy (eV)')
        ax.set_ylabel('Decomposition Energy (eV/atom)')
        ax.set_title('Formation vs Decomposition Energy')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        figures['energy_correlation'] = fig
        
        print(f"  Generated {len(figures)} energy distribution figures")
        
        return figures
    
    def plot_stability_analysis(self) -> Dict[str, plt.Figure]:
        """
        Analyze and visualize stability categories.
        Pure matplotlib implementation.
        
        Returns:
            Dictionary of figure objects
        """
        print("\nGenerating stability analysis plots...")
        
        figures = {}
        
        # Classify stability
        df_copy = self.df.copy()
        df_copy['Stability'] = pd.cut(
            df_copy['dH_decomposition_ev'],
            bins=[-np.inf, 0.025, 0.1, np.inf],
            labels=['Stable', 'Metastable', 'Unstable']
        )
        
        # Figure 1: Stability distribution
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor('white')
        
        # Pie chart
        ax = axes[0]
        stability_counts = df_copy['Stability'].value_counts()
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        
        wedges, texts, autotexts = ax.pie(
            stability_counts.values,
            labels=stability_counts.index,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 12, 'weight': 'bold'}
        )
        
        ax.set_title('Stability Distribution')
        
        # Bar chart with counts
        ax = axes[1]
        bars = ax.bar(range(len(stability_counts)), stability_counts.values, 
                      color=colors, edgecolor='black', linewidth=1.5, alpha=0.8,
                      tick_label=stability_counts.index)
        
        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}',
                   ha='center', va='bottom', fontsize=11, weight='bold')
        
        ax.set_ylabel('Count')
        ax.set_title('Stability Category Counts')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        figures['stability_distribution'] = fig
        
        # Figure 2: Stability vs tolerance factors
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor('white')
        
        # Goldschmidt tolerance
        ax = axes[0]
        for stability, color in zip(['Stable', 'Metastable', 'Unstable'], colors):
            mask = df_copy['Stability'] == stability
            x = df_copy.loc[mask, 'tolerance_t'].values
            y = df_copy.loc[mask, 'dH_decomposition_ev'].values
            x_ds, y_ds = self._downsample_for_plot(x, y)
            ax.scatter(x_ds, y_ds, label=stability, alpha=0.5, s=10, color=color,
                       edgecolors='none', rasterized=(len(x_ds) > 20000))
        
        ax.axhline(0.025, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.axhline(0.1, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.axvline(0.825, color='blue', linestyle='--', linewidth=1, alpha=0.5, 
                   label='t range')
        ax.axvline(1.059, color='blue', linestyle='--', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Goldschmidt Tolerance Factor (t)')
        ax.set_ylabel('Decomposition Energy (eV/atom)')
        ax.set_title('Stability vs Goldschmidt Tolerance')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Bartel tolerance
        ax = axes[1]
        for stability, color in zip(['Stable', 'Metastable', 'Unstable'], colors):
            mask = df_copy['Stability'] == stability
            x = df_copy.loc[mask, 'tolerance_tau'].values
            y = df_copy.loc[mask, 'dH_decomposition_ev'].values
            x_ds, y_ds = self._downsample_for_plot(x, y)
            ax.scatter(x_ds, y_ds, label=stability, alpha=0.5, s=10, color=color,
                       edgecolors='none', rasterized=(len(x_ds) > 20000))
        
        ax.axhline(0.025, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.axhline(0.1, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.axvline(4.18, color='blue', linestyle='--', linewidth=1, alpha=0.5, 
                   label='tau threshold')
        
        ax.set_xlabel('Bartel Tolerance Factor (tau)')
        ax.set_ylabel('Decomposition Energy (eV/atom)')
        ax.set_title('Stability vs Bartel Tolerance')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        figures['stability_vs_tolerance'] = fig
        
        print(f"  Generated {len(figures)} stability analysis figures")
        
        return figures
    
    def plot_element_analysis(self, top_n: int = 20) -> Dict[str, plt.Figure]:
        """
        Analyze element distribution and frequency.
        Pure matplotlib implementation.
        
        Args:
            top_n: Number of top elements to show
        
        Returns:
            Dictionary of figure objects
        """
        print("\nGenerating element analysis plots...")
        
        figures = {}
        
        # Count element frequencies
        a_site_elements = pd.concat([self.df['Element_a1'], self.df['Element_a2']])
        b_site_elements = pd.concat([self.df['Element_b1'], self.df['Element_b2']])
        
        a_counts = a_site_elements.value_counts().head(top_n)
        b_counts = b_site_elements.value_counts().head(top_n)
        
        # Figure 1: Element frequency bars
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        fig.patch.set_facecolor('white')
        
        # A-site elements
        ax = axes[0]
        y_pos = np.arange(len(a_counts))
        bars = ax.barh(y_pos, a_counts.values, color='steelblue', 
                       edgecolor='black', linewidth=0.5, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(a_counts.index)
        ax.set_xlabel('Frequency')
        ax.set_title(f'Top {top_n} A-site Elements')
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
        
        # Add count labels
        for i, count in enumerate(a_counts.values):
            ax.text(count + max(a_counts.values)*0.01, i, f'{count:,}',
                   va='center', fontsize=9)
        
        # B-site elements
        ax = axes[1]
        y_pos = np.arange(len(b_counts))
        bars = ax.barh(y_pos, b_counts.values, color='coral',
                       edgecolor='black', linewidth=0.5, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(b_counts.index)
        ax.set_xlabel('Frequency')
        ax.set_title(f'Top {top_n} B-site Elements')
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
        
        # Add count labels
        for i, count in enumerate(b_counts.values):
            ax.text(count + max(b_counts.values)*0.01, i, f'{count:,}',
                   va='center', fontsize=9)
        
        plt.tight_layout()
        figures['element_frequency'] = fig
        
        # Figure 2: Element stability analysis
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor('white')
        
        # A1-site element stability
        ax = axes[0, 0]
        elem_stability_a1 = self.df.groupby('Element_a1')['dH_decomposition_ev'].agg(['mean', 'std', 'count'])
        elem_stability_a1 = elem_stability_a1.nlargest(15, 'count')
        
        x_pos = np.arange(len(elem_stability_a1))
        ax.errorbar(x_pos, elem_stability_a1['mean'],
                   yerr=elem_stability_a1['std'], fmt='o', capsize=5,
                   color='steelblue', markersize=8, linewidth=2)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(elem_stability_a1.index, rotation=45, ha='right')
        ax.axhline(0.025, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
        ax.axhline(0.1, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
        ax.set_ylabel('Mean Decomposition Energy (eV/atom)')
        ax.set_title('A1-site Element Mean Stability')
        ax.grid(True, alpha=0.3)
        
        # A2-site element stability
        ax = axes[0, 1]
        elem_stability_a2 = self.df.groupby('Element_a2')['dH_decomposition_ev'].agg(['mean', 'std', 'count'])
        elem_stability_a2 = elem_stability_a2.nlargest(15, 'count')
        
        x_pos = np.arange(len(elem_stability_a2))
        ax.errorbar(x_pos, elem_stability_a2['mean'],
                   yerr=elem_stability_a2['std'], fmt='o', capsize=5,
                   color='steelblue', markersize=8, linewidth=2)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(elem_stability_a2.index, rotation=45, ha='right')
        ax.axhline(0.025, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
        ax.axhline(0.1, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
        ax.set_ylabel('Mean Decomposition Energy (eV/atom)')
        ax.set_title('A2-site Element Mean Stability')
        ax.grid(True, alpha=0.3)
        
        # B1-site element stability
        ax = axes[1, 0]
        elem_stability_b1 = self.df.groupby('Element_b1')['dH_decomposition_ev'].agg(['mean', 'std', 'count'])
        elem_stability_b1 = elem_stability_b1.nlargest(15, 'count')
        
        x_pos = np.arange(len(elem_stability_b1))
        ax.errorbar(x_pos, elem_stability_b1['mean'],
                   yerr=elem_stability_b1['std'], fmt='o', capsize=5,
                   color='coral', markersize=8, linewidth=2)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(elem_stability_b1.index, rotation=45, ha='right')
        ax.axhline(0.025, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
        ax.axhline(0.1, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
        ax.set_ylabel('Mean Decomposition Energy (eV/atom)')
        ax.set_title('B1-site Element Mean Stability')
        ax.grid(True, alpha=0.3)
        
        # B2-site element stability
        ax = axes[1, 1]
        elem_stability_b2 = self.df.groupby('Element_b2')['dH_decomposition_ev'].agg(['mean', 'std', 'count'])
        elem_stability_b2 = elem_stability_b2.nlargest(15, 'count')
        
        x_pos = np.arange(len(elem_stability_b2))
        ax.errorbar(x_pos, elem_stability_b2['mean'],
                   yerr=elem_stability_b2['std'], fmt='o', capsize=5,
                   color='coral', markersize=8, linewidth=2)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(elem_stability_b2.index, rotation=45, ha='right')
        ax.axhline(0.025, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
        ax.axhline(0.1, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
        ax.set_ylabel('Mean Decomposition Energy (eV/atom)')
        ax.set_title('B2-site Element Mean Stability')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        figures['element_stability'] = fig
        
        print(f"  Generated {len(figures)} element analysis figures")
        
        return figures
    
    def plot_correlation_analysis(self, min_corr: float = 0.3) -> Dict[str, plt.Figure]:
        """
        Analyze correlations between properties.
        Pure matplotlib implementation.
        
        Args:
            min_corr: Minimum correlation to display
        
        Returns:
            Dictionary of figure objects
        """
        print("\nGenerating correlation analysis plots...")
        
        figures = {}
        
        # Select numeric columns
        numeric_cols = [
            'Oxidation_a1', 'Oxidation_a2', 'Oxidation_b1', 'Oxidation_b2',
            'tolerance_t', 'tolerance_tau',
            'dH_formation_ev', 'dH_decomposition_ev'
        ]
        
        corr_df = self.df[numeric_cols].copy()
        
        # Compute correlation matrix
        corr_matrix = corr_df.corr()
        
        # Figure 1: Correlation heatmap (pure matplotlib)
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('white')
        
        # Create heatmap
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        
        # Set ticks
        ax.set_xticks(np.arange(len(corr_matrix.columns)))
        ax.set_yticks(np.arange(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr_matrix.columns)
        
        # Add correlation values as text
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                if i < j:  # Upper triangle only
                    continue
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=9)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Correlation', rotation=270, labelpad=20)
        
        ax.set_title('Property Correlation Matrix', fontsize=14, weight='bold')
        plt.tight_layout()
        figures['correlation_heatmap'] = fig
        
        # Figure 2: Key property correlations
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.patch.set_facecolor('white')
        
        # Formation vs Decomposition
        ax = axes[0, 0]
        x = self.df['dH_formation_ev'].values
        y = self.df['dH_decomposition_ev'].values
        x_ds, y_ds = self._downsample_for_plot(x, y)
        ax.scatter(x_ds, y_ds, alpha=0.3, s=5, color='steelblue', edgecolors='none', rasterized=(len(x_ds) > 20000))
        corr = self.df['dH_formation_ev'].corr(self.df['dH_decomposition_ev'])
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
               fontsize=12, weight='bold', va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_xlabel('Formation Energy (eV)')
        ax.set_ylabel('Decomposition Energy (eV/atom)')
        ax.set_title('Formation vs Decomposition Energy')
        ax.grid(True, alpha=0.3)
        
        # Tolerance t vs Decomposition
        ax = axes[0, 1]
        x = self.df['tolerance_t'].values
        y = self.df['dH_decomposition_ev'].values
        x_ds, y_ds = self._downsample_for_plot(x, y)
        ax.scatter(x_ds, y_ds, alpha=0.3, s=5, color='coral', edgecolors='none', rasterized=(len(x_ds) > 20000))
        corr = self.df['tolerance_t'].corr(self.df['dH_decomposition_ev'])
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
               fontsize=12, weight='bold', va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_xlabel('Goldschmidt Tolerance (t)')
        ax.set_ylabel('Decomposition Energy (eV/atom)')
        ax.set_title('Tolerance vs Stability')
        ax.grid(True, alpha=0.3)
        
        # Tolerance tau vs Decomposition
        ax = axes[1, 0]
        x = self.df['tolerance_tau'].values
        y = self.df['dH_decomposition_ev'].values
        x_ds, y_ds = self._downsample_for_plot(x, y)
        ax.scatter(x_ds, y_ds, alpha=0.3, s=5, color='green', edgecolors='none', rasterized=(len(x_ds) > 20000))
        corr = self.df['tolerance_tau'].corr(self.df['dH_decomposition_ev'])
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
               fontsize=12, weight='bold', va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_xlabel('Bartel Tolerance (tau)')
        ax.set_ylabel('Decomposition Energy (eV/atom)')
        ax.set_title('Bartel Tolerance vs Stability')
        ax.grid(True, alpha=0.3)
        
        # Tolerance t vs tau
        ax = axes[1, 1]
        x = self.df['tolerance_t'].values
        y = self.df['tolerance_tau'].values
        x_ds, y_ds = self._downsample_for_plot(x, y)
        ax.scatter(x_ds, y_ds, alpha=0.3, s=5, color='purple', edgecolors='none', rasterized=(len(x_ds) > 20000))
        corr = self.df['tolerance_t'].corr(self.df['tolerance_tau'])
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
               fontsize=12, weight='bold', va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_xlabel('Goldschmidt Tolerance (t)')
        ax.set_ylabel('Bartel Tolerance (tau)')
        ax.set_title('Tolerance Factor Comparison')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        figures['key_correlations'] = fig
        
        print(f"  Generated {len(figures)} correlation figures")
        
        return figures
    
    def generate_summary_statistics(self) -> Dict:
        """
        Generate comprehensive summary statistics.
        
        Returns:
            Dictionary of statistics
        """
        print("\nComputing summary statistics...")
        
        stats_dict = {}
        
        # Basic statistics
        numeric_cols = [
            'dH_formation_ev', 'dH_decomposition_ev',
            'tolerance_t', 'tolerance_tau',
            'Oxidation_a1', 'Oxidation_a2', 'Oxidation_b1', 'Oxidation_b2'
        ]
        
        for col in numeric_cols:
            stats_dict[col] = {
                'mean': float(self.df[col].mean()),
                'std': float(self.df[col].std()),
                'min': float(self.df[col].min()),
                'max': float(self.df[col].max()),
                'median': float(self.df[col].median()),
                'q25': float(self.df[col].quantile(0.25)),
                'q75': float(self.df[col].quantile(0.75))
            }
        
        # Stability categories
        stability_counts = pd.cut(
            self.df['dH_decomposition_ev'],
            bins=[-np.inf, 0.025, 0.1, np.inf],
            labels=['Stable', 'Metastable', 'Unstable']
        ).value_counts()
        
        stats_dict['stability_distribution'] = {
            str(k): int(v) for k, v in stability_counts.items()
        }
        
        # Element statistics
        stats_dict['unique_elements'] = {
            'a_site': len(set(self.df['Element_a1']) | set(self.df['Element_a2'])),
            'b_site': len(set(self.df['Element_b1']) | set(self.df['Element_b2'])),
            'total': len(set(self.df['Element_a1']) | set(self.df['Element_a2']) |
                        set(self.df['Element_b1']) | set(self.df['Element_b2']))
        }
        
        print(f"  Computed statistics for {len(stats_dict)} categories")
        
        return stats_dict