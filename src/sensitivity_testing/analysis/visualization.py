"""
Visualization tools for sensitivity analysis results.
"""

import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Visualizer:
    """
    Visualization tools for sensitivity analysis.
    
    Generates plots for:
    - Sensitivity landscapes
    - Phase transition maps
    - Bug clustering
    - Coverage progress
    """
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = Path(output_dir) if output_dir else Path('./plots')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check matplotlib availability
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            self._has_matplotlib = True
        except ImportError:
            logger.warning("matplotlib not available. Visualization disabled.")
            self._has_matplotlib = False
    
    def plot_landscape(
        self,
        landscape: 'SensitivityLandscape',
        filename: str = 'landscape.png',
        title: str = 'Sensitivity Landscape'
    ) -> Optional[Path]:
        """Plot sensitivity landscape as heatmap."""
        if not self._has_matplotlib:
            return None
        
        import matplotlib.pyplot as plt
        
        grid = landscape.to_grid()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(
            grid.T,
            origin='lower',
            cmap='RdYlBu_r',
            aspect='auto',
            extent=[0, 1, 0, 1]
        )
        
        plt.colorbar(im, ax=ax, label='Lyapunov Exponent (λ)')
        
        ax.set_xlabel('Parameter 1')
        ax.set_ylabel('Parameter 2')
        ax.set_title(title)
        
        # Mark high-sensitivity regions
        high_sens = landscape.get_high_sensitivity_regions(threshold=0.1)
        for p in high_sens:
            if len(p.coordinates) >= 2:
                ax.plot(p.coordinates[0], p.coordinates[1], 'k*', markersize=10)
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_lyapunov_distribution(
        self,
        lyapunov_values: List[float],
        filename: str = 'lyapunov_dist.png'
    ) -> Optional[Path]:
        """Plot distribution of Lyapunov exponents."""
        if not self._has_matplotlib:
            return None
        
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(lyapunov_values, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--', label='λ=0 (stability boundary)')
        
        ax.set_xlabel('Lyapunov Exponent (λ)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Sensitivity Scores')
        ax.legend()
        
        # Add statistics
        stats_text = f'Mean: {np.mean(lyapunov_values):.3f}\n'
        stats_text += f'Std: {np.std(lyapunov_values):.3f}\n'
        stats_text += f'% Chaotic (λ>0): {100*sum(1 for l in lyapunov_values if l > 0)/len(lyapunov_values):.1f}%'
        
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_phase_transitions(
        self,
        parameter_values: List[float],
        metrics: List[float],
        transitions: List[int],
        filename: str = 'phase_transitions.png',
        parameter_name: str = 'Parameter'
    ) -> Optional[Path]:
        """Plot detected phase transitions."""
        if not self._has_matplotlib:
            return None
        
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(parameter_values, metrics, 'b-', linewidth=1, label='Behavior Metric')
        
        # Mark phase transitions
        for t in transitions:
            if t < len(parameter_values):
                ax.axvline(x=parameter_values[t], color='r', linestyle='--', alpha=0.7)
        
        ax.set_xlabel(parameter_name)
        ax.set_ylabel('Behavior Metric')
        ax.set_title(f'Phase Transitions in Compiler Behavior ({len(transitions)} detected)')
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_bug_rate(
        self,
        bugs_over_time: List[int],
        tests_over_time: List[int],
        filename: str = 'bug_rate.png'
    ) -> Optional[Path]:
        """Plot bug finding rate over time."""
        if not self._has_matplotlib:
            return None
        
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Cumulative bugs
        ax1.plot(range(len(bugs_over_time)), np.cumsum(bugs_over_time), 'b-', linewidth=2)
        ax1.set_xlabel('Tests Run')
        ax1.set_ylabel('Cumulative Bugs Found')
        ax1.set_title('Bug Discovery Progress')
        ax1.grid(True, alpha=0.3)
        
        # Bug rate (rolling window)
        window = 100
        if len(bugs_over_time) > window:
            rolling_rate = np.convolve(bugs_over_time, np.ones(window)/window, mode='valid')
            ax2.plot(range(len(rolling_rate)), rolling_rate, 'r-', linewidth=2)
        else:
            ax2.plot(range(len(bugs_over_time)), bugs_over_time, 'r-', linewidth=2)
        
        ax2.set_xlabel('Tests Run')
        ax2.set_ylabel(f'Bug Rate (per {window} tests)')
        ax2.set_title('Bug Finding Rate')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
