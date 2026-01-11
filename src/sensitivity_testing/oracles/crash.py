"""
Crash Detection Oracle.

Detects bugs through crashes, sanitizer reports, and abnormal terminations.
"""

from typing import Dict, Optional
from .differential import BaseOracle, Bug


class CrashOracle(BaseOracle):
    """
    Crash detection oracle.
    
    Detects compiler crashes, sanitizer violations, and runtime errors.
    """
    
    def __init__(self, check_sanitizers: bool = True):
        self.check_sanitizers = check_sanitizers
        
        # Sanitizer error patterns
        self.sanitizer_patterns = [
            'AddressSanitizer',
            'UndefinedBehaviorSanitizer',
            'MemorySanitizer',
            'ThreadSanitizer',
            'runtime error:',
            'SUMMARY:',
        ]
    
    def check(self, program: 'TestProgram', results: Dict[str, 'CompilationResult']) -> Optional[Bug]:
        """Check for crashes and sanitizer violations."""
        for key, result in results.items():
            # Check compilation crash
            if not result.success and result.return_code != 0:
                if 'internal compiler error' in result.stderr.lower():
                    return Bug(
                        bug_type='compiler_crash',
                        description=f"Internal compiler error in {key}",
                        program_path=str(program.path),
                        compilers_disagree=[key],
                        outputs={'stderr': result.stderr[:500]},
                        severity='critical'
                    )
            
            # Check sanitizer violations
            if self.check_sanitizers and result.stderr:
                for pattern in self.sanitizer_patterns:
                    if pattern in result.stderr:
                        return Bug(
                            bug_type='sanitizer_violation',
                            description=f"Sanitizer violation detected in {key}: {pattern}",
                            program_path=str(program.path),
                            compilers_disagree=[key],
                            outputs={'stderr': result.stderr[:500]},
                            severity='high'
                        )
        
        return None
