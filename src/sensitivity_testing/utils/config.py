"""
Configuration management for the sensitivity testing framework.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import json
import yaml


@dataclass
class FrameworkConfig:
    """Configuration for the sensitivity testing framework."""
    
    # Lyapunov algorithm parameters
    embedding_dimension: int = 5
    time_delay: int = 1
    min_neighbors: int = 5
    
    # Phase transition detection
    phase_detection_method: str = 'cusum'
    significance_level: float = 0.05
    
    # PAC learning parameters
    epsilon: float = 0.05
    delta: float = 0.01
    
    # Execution parameters
    execution_timeout: float = 30.0
    compilation_timeout: float = 60.0
    batch_size: int = 10
    
    # Output settings
    output_dir: Path = field(default_factory=lambda: Path('./results'))
    save_traces: bool = False
    verbose: bool = False
    
    @classmethod
    def from_file(cls, path: Path) -> 'FrameworkConfig':
        """Load configuration from file."""
        path = Path(path)
        
        if path.suffix == '.json':
            with open(path) as f:
                data = json.load(f)
        elif path.suffix in ['.yaml', '.yml']:
            with open(path) as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
        
        return cls(**data)
    
    def save(self, path: Path):
        """Save configuration to file."""
        path = Path(path)
        data = self.__dict__.copy()
        data['output_dir'] = str(data['output_dir'])
        
        if path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        elif path.suffix in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(data, f)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    
    name: str
    compilers: List[str]
    optimization_levels: List[str]
    generator: str
    budget_hours: float
    oracle: str
    seed: Optional[int] = None
    
    @classmethod
    def from_file(cls, path: Path) -> 'ExperimentConfig':
        """Load experiment config from file."""
        path = Path(path)
        
        with open(path) as f:
            if path.suffix == '.json':
                data = json.load(f)
            else:
                data = yaml.safe_load(f)
        
        return cls(**data)
