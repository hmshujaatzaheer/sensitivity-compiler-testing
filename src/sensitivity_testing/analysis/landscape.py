"""
Sensitivity Landscape Mapping.

Maps the sensitivity characteristics across the input space
to identify bug-prone regions.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class LandscapePoint:
    """A point in the sensitivity landscape."""
    coordinates: Tuple[float, ...]
    lyapunov_exponent: float
    phase_transitions: int
    sensitivity_score: float
    program_hash: str


class SensitivityLandscape:
    """
    Maps sensitivity characteristics across input space.
    
    Provides methods for:
    - Adding analysis results as landscape points
    - Querying high-sensitivity regions
    - Estimating coverage
    - Visualizing the landscape
    """
    
    def __init__(self, dimensions: int = 2):
        self.dimensions = dimensions
        self.points: List[LandscapePoint] = []
        self._grid = None
        self._grid_resolution = 100
    
    def add_point(self, result: 'AnalysisResult'):
        """Add an analysis result to the landscape."""
        # Extract coordinates from program parameters
        params = result.metadata.get('parameters', {})
        
        if params:
            coords = tuple(list(params.values())[:self.dimensions])
        else:
            # Use hash-based pseudo-coordinates
            hash_val = hash(result.program_path)
            coords = (
                (hash_val % 1000) / 1000,
                ((hash_val >> 10) % 1000) / 1000
            )
        
        point = LandscapePoint(
            coordinates=coords,
            lyapunov_exponent=result.lyapunov_exponent,
            phase_transitions=len(result.phase_transitions),
            sensitivity_score=result.sensitivity_score,
            program_hash=result.program_path
        )
        
        self.points.append(point)
        self._grid = None  # Invalidate grid
    
    def get_high_sensitivity_regions(
        self,
        threshold: float = 0.0,
        top_k: int = 10
    ) -> List[LandscapePoint]:
        """Get regions with highest sensitivity."""
        high_sens = [p for p in self.points if p.lyapunov_exponent > threshold]
        return sorted(high_sens, key=lambda p: -p.lyapunov_exponent)[:top_k]
    
    def coverage_estimate(self) -> float:
        """Estimate landscape coverage based on point distribution."""
        if len(self.points) < 2:
            return 0.0
        
        # Simple coverage estimate based on point spread
        coords = np.array([p.coordinates for p in self.points if len(p.coordinates) >= 2])
        
        if len(coords) < 2:
            return len(self.points) / 1000  # Rough estimate
        
        # Compute convex hull volume approximation
        try:
            from scipy.spatial import ConvexHull
            if len(coords) >= self.dimensions + 1:
                hull = ConvexHull(coords[:, :self.dimensions])
                # Normalize by expected total volume
                coverage = min(1.0, hull.volume / 1.0)
                return coverage
        except Exception:
            pass
        
        # Fallback: use standard deviation spread
        spread = np.std(coords, axis=0)
        coverage = min(1.0, np.prod(spread) * len(self.points) / 100)
        
        return float(coverage)
    
    def interpolate(self, coordinates: Tuple[float, ...]) -> float:
        """Interpolate sensitivity at given coordinates."""
        if not self.points:
            return 0.0
        
        # Find k nearest neighbors
        k = min(5, len(self.points))
        
        distances = []
        for p in self.points:
            if len(p.coordinates) >= len(coordinates):
                dist = np.sqrt(sum(
                    (a - b) ** 2 
                    for a, b in zip(coordinates, p.coordinates[:len(coordinates)])
                ))
                distances.append((dist, p.lyapunov_exponent))
        
        distances.sort(key=lambda x: x[0])
        nearest = distances[:k]
        
        # Inverse distance weighting
        if nearest[0][0] < 1e-10:
            return nearest[0][1]
        
        weights = [1.0 / (d + 1e-10) for d, _ in nearest]
        total_weight = sum(weights)
        
        interpolated = sum(w * v for (d, v), w in zip(nearest, weights)) / total_weight
        
        return float(interpolated)
    
    def to_grid(self, resolution: int = 100) -> np.ndarray:
        """Convert landscape to regular grid for visualization."""
        if self._grid is not None and self._grid_resolution == resolution:
            return self._grid
        
        self._grid_resolution = resolution
        
        # Create 2D grid
        grid = np.zeros((resolution, resolution))
        
        for i in range(resolution):
            for j in range(resolution):
                x = i / resolution
                y = j / resolution
                grid[i, j] = self.interpolate((x, y))
        
        self._grid = grid
        return grid
    
    def summary(self) -> Dict:
        """Get landscape summary statistics."""
        if not self.points:
            return {'num_points': 0}
        
        lyapunovs = [p.lyapunov_exponent for p in self.points]
        
        return {
            'num_points': len(self.points),
            'mean_lyapunov': float(np.mean(lyapunovs)),
            'max_lyapunov': float(np.max(lyapunovs)),
            'min_lyapunov': float(np.min(lyapunovs)),
            'std_lyapunov': float(np.std(lyapunovs)),
            'high_sensitivity_fraction': sum(1 for l in lyapunovs if l > 0) / len(lyapunovs),
            'coverage_estimate': self.coverage_estimate()
        }
