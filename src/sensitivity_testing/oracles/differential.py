"""
Differential testing oracle for compiler bug detection.

Compares outputs across multiple compilers to detect disagreements.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Bug:
    """Represents a detected bug."""
    bug_type: str
    description: str
    program_path: str
    compilers_disagree: List[str]
    outputs: Dict[str, str]
    severity: str = 'unknown'
    
    def to_dict(self) -> Dict:
        return {
            'bug_type': self.bug_type,
            'description': self.description,
            'program_path': self.program_path,
            'compilers_disagree': self.compilers_disagree,
            'severity': self.severity
        }


class BaseOracle:
    """Base class for bug detection oracles."""
    
    def check(self, program: 'TestProgram', results: Dict[str, 'CompilationResult']) -> Optional[Bug]:
        """Check if compilation results indicate a bug."""
        raise NotImplementedError


class DifferentialOracle(BaseOracle):
    """
    Differential testing oracle.
    
    Detects bugs by comparing outputs across multiple compilers.
    If compilers disagree on output, at least one has a bug.
    """
    
    def __init__(self, ignore_exit_codes: bool = False, ignore_whitespace: bool = True):
        self.ignore_exit_codes = ignore_exit_codes
        self.ignore_whitespace = ignore_whitespace
    
    def check(self, program: 'TestProgram', results: Dict[str, 'CompilationResult']) -> Optional[Bug]:
        """Check for differential bugs across compilers."""
        # Group results by output
        output_groups = {}
        
        for key, result in results.items():
            if not result.success:
                continue
            
            # Get output (would need to execute the compiled program)
            output = self._normalize_output(getattr(result, 'output', ''))
            
            if output not in output_groups:
                output_groups[output] = []
            output_groups[output].append(key)
        
        # Check for disagreement
        if len(output_groups) > 1:
            # Find minority group (likely the buggy compiler)
            groups = sorted(output_groups.items(), key=lambda x: len(x[1]))
            minority = groups[0]
            majority = groups[-1]
            
            return Bug(
                bug_type='differential',
                description=f"Output disagreement between {minority[1]} and {majority[1]}",
                program_path=str(program.path),
                compilers_disagree=minority[1],
                outputs={k: v for k, v in [(minority[0], minority[1]), (majority[0], majority[1])]},
                severity='high' if len(minority[1]) == 1 else 'medium'
            )
        
        return None
    
    def _normalize_output(self, output: str) -> str:
        """Normalize output for comparison."""
        if self.ignore_whitespace:
            return ' '.join(output.split())
        return output
