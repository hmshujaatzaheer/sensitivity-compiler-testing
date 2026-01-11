"""
Metamorphic Testing Oracle.

Detects bugs by checking metamorphic relations between inputs/outputs.
"""

from typing import Dict, Optional
from .differential import BaseOracle, Bug


class MetamorphicOracle(BaseOracle):
    """
    Metamorphic testing oracle.
    
    Checks that optimization levels produce equivalent outputs.
    """
    
    def check(self, program: 'TestProgram', results: Dict[str, 'CompilationResult']) -> Optional[Bug]:
        """Check for metamorphic relation violations."""
        # Compare outputs across optimization levels for same compiler
        compilers = {}
        
        for key, result in results.items():
            if not result.success:
                continue
            
            compiler = key.split('_')[0]
            if compiler not in compilers:
                compilers[compiler] = {}
            compilers[compiler][key] = result
        
        # Check each compiler's optimization levels produce same output
        for compiler, opts in compilers.items():
            outputs = set()
            for key, result in opts.items():
                output = getattr(result, 'output', '')
                outputs.add(output)
            
            if len(outputs) > 1:
                return Bug(
                    bug_type='metamorphic',
                    description=f"{compiler} produces different outputs at different optimization levels",
                    program_path=str(program.path),
                    compilers_disagree=[compiler],
                    outputs={},
                    severity='high'
                )
        
        return None
