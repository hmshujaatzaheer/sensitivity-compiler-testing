"""
Metrics collection and performance tracking.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class MetricsCollector:
    """Collects and tracks framework metrics."""
    
    analyses_run: int = 0
    bugs_found: int = 0
    total_time_seconds: float = 0.0
    lyapunov_values: List[float] = field(default_factory=list)
    phase_transitions: int = 0
    
    def record_analysis(self, result: 'AnalysisResult'):
        """Record an analysis result."""
        self.analyses_run += 1
        self.total_time_seconds += result.analysis_time_seconds
        self.lyapunov_values.append(result.lyapunov_exponent)
        self.phase_transitions += len(result.phase_transitions)
    
    def record_bug(self, bug: Dict):
        """Record a found bug."""
        self.bugs_found += 1
    
    def summary(self) -> Dict:
        """Get metrics summary."""
        import numpy as np
        
        return {
            'analyses_run': self.analyses_run,
            'bugs_found': self.bugs_found,
            'total_time_seconds': self.total_time_seconds,
            'average_lyapunov': float(np.mean(self.lyapunov_values)) if self.lyapunov_values else 0,
            'max_lyapunov': float(np.max(self.lyapunov_values)) if self.lyapunov_values else 0,
            'phase_transitions': self.phase_transitions,
            'bugs_per_hour': self.bugs_found / max(self.total_time_seconds / 3600, 0.001)
        }
    
    def save(self, path: str):
        """Save metrics to file."""
        with open(path, 'w') as f:
            json.dump(self.summary(), f, indent=2)
