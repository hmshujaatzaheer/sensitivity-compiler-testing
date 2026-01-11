"""
Test program representation and generation.

This module provides classes for representing and generating test programs
for compiler testing.
"""

import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
import logging
import hashlib
import re

logger = logging.getLogger(__name__)


@dataclass
class TestProgram:
    """
    Represents a test program for compiler testing.
    
    Attributes:
        path: Path to the source file
        source_code: Source code content
        name: Program name/identifier
        language: Programming language (c, cpp, etc.)
        metadata: Additional metadata
    """
    
    path: Path
    source_code: str = ""
    name: str = ""
    language: str = "c"
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        self.path = Path(self.path)
        if not self.name:
            self.name = self.path.stem
        if not self.source_code and self.path.exists():
            self.source_code = self.path.read_text()
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'TestProgram':
        """Load test program from file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Program file not found: {path}")
        
        source_code = path.read_text()
        
        # Detect language from extension
        ext = path.suffix.lower()
        lang_map = {'.c': 'c', '.cpp': 'cpp', '.cc': 'cpp', '.cxx': 'cpp'}
        language = lang_map.get(ext, 'c')
        
        return cls(
            path=path,
            source_code=source_code,
            name=path.stem,
            language=language
        )
    
    @classmethod
    def from_source(cls, source_code: str, name: str = "test", language: str = "c") -> 'TestProgram':
        """Create test program from source code string."""
        # Create temporary file
        suffix = '.c' if language == 'c' else '.cpp'
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix=suffix, delete=False, prefix=f"{name}_"
        ) as f:
            f.write(source_code)
            path = Path(f.name)
        
        return cls(
            path=path,
            source_code=source_code,
            name=name,
            language=language
        )
    
    def save(self, path: Union[str, Path]):
        """Save program to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.source_code)
        self.path = path
    
    def hash(self) -> str:
        """Get content hash of the program."""
        return hashlib.sha256(self.source_code.encode()).hexdigest()[:16]
    
    def line_count(self) -> int:
        """Get number of lines."""
        return len(self.source_code.splitlines())
    
    def extract_parameters(self) -> Dict[str, int]:
        """
        Extract numerical parameters from program.
        
        Looks for patterns like:
        - #define NAME VALUE
        - const int name = value;
        - for loops with numeric bounds
        
        Returns:
            Dictionary mapping parameter names to values
        """
        params = {}
        
        # Match #define NAME VALUE
        define_pattern = r'#define\s+(\w+)\s+(\d+)'
        for match in re.finditer(define_pattern, self.source_code):
            params[match.group(1)] = int(match.group(2))
        
        # Match const int name = value
        const_pattern = r'const\s+int\s+(\w+)\s*=\s*(\d+)'
        for match in re.finditer(const_pattern, self.source_code):
            params[match.group(1)] = int(match.group(2))
        
        # Match array sizes: type name[SIZE]
        array_pattern = r'\w+\s+\w+\[(\d+)\]'
        for i, match in enumerate(re.finditer(array_pattern, self.source_code)):
            params[f'ARRAY_SIZE_{i}'] = int(match.group(1))
        
        # Match for loop bounds: for(... i < N; ...)
        loop_pattern = r'for\s*\([^;]+;\s*\w+\s*<\s*(\d+)'
        for i, match in enumerate(re.finditer(loop_pattern, self.source_code)):
            params[f'LOOP_BOUND_{i}'] = int(match.group(1))
        
        return params
    
    def mutate_parameter(self, param_name: str, new_value: int) -> 'TestProgram':
        """
        Create a new program with a parameter changed.
        
        Args:
            param_name: Parameter to change
            new_value: New value
            
        Returns:
            New TestProgram with modified parameter
        """
        new_source = self.source_code
        
        # Try different patterns
        patterns = [
            (rf'(#define\s+{param_name}\s+)\d+', rf'\g<1>{new_value}'),
            (rf'(const\s+int\s+{param_name}\s*=\s*)\d+', rf'\g<1>{new_value}'),
        ]
        
        for pattern, replacement in patterns:
            new_source = re.sub(pattern, replacement, new_source)
        
        return TestProgram.from_source(
            new_source,
            name=f"{self.name}_{param_name}_{new_value}",
            language=self.language
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'path': str(self.path),
            'name': self.name,
            'language': self.language,
            'hash': self.hash(),
            'lines': self.line_count(),
            'parameters': self.extract_parameters(),
            'metadata': self.metadata
        }


class ProgramGenerator:
    """
    Base class for test program generators.
    
    Implementations include:
    - CsmithGenerator: Random C program generation
    - YARPGenGenerator: Intel's random program generator
    - TemplateGenerator: Template-based generation
    """
    
    def __init__(self, working_dir: Optional[Path] = None):
        self.working_dir = Path(working_dir) if working_dir else Path(tempfile.mkdtemp())
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self._counter = 0
    
    @classmethod
    def create(cls, generator_name: str, **kwargs) -> 'ProgramGenerator':
        """Factory method to create generators by name."""
        generators = {
            'csmith': CsmithGenerator,
            'yarpgen': YARPGenGenerator,
            'template': TemplateGenerator,
            'simple': SimpleGenerator,
        }
        
        if generator_name.lower() not in generators:
            raise ValueError(f"Unknown generator: {generator_name}. Available: {list(generators.keys())}")
        
        return generators[generator_name.lower()](**kwargs)
    
    def generate(self) -> TestProgram:
        """Generate a single test program."""
        raise NotImplementedError
    
    def generate_batch(self, size: int) -> List[TestProgram]:
        """Generate multiple test programs."""
        return [self.generate() for _ in range(size)]
    
    def _next_name(self) -> str:
        """Get next unique program name."""
        self._counter += 1
        return f"test_{self._counter:06d}"


class CsmithGenerator(ProgramGenerator):
    """
    Generate random C programs using Csmith.
    
    Csmith is a tool that generates random C programs that statically
    and dynamically conform to the C99 standard.
    
    Requires Csmith to be installed and in PATH.
    """
    
    def __init__(
        self,
        working_dir: Optional[Path] = None,
        csmith_path: str = "csmith",
        max_funcs: int = 10,
        max_block_depth: int = 3,
        timeout: int = 30
    ):
        super().__init__(working_dir)
        self.csmith_path = csmith_path
        self.max_funcs = max_funcs
        self.max_block_depth = max_block_depth
        self.timeout = timeout
        
        # Check Csmith availability
        self._available = shutil.which(csmith_path) is not None
        if not self._available:
            logger.warning("Csmith not found in PATH. Install from https://github.com/csmith-project/csmith")
    
    def generate(self) -> TestProgram:
        """Generate a random C program using Csmith."""
        if not self._available:
            return self._generate_fallback()
        
        name = self._next_name()
        output_path = self.working_dir / f"{name}.c"
        
        cmd = [
            self.csmith_path,
            f'--max-funcs', str(self.max_funcs),
            f'--max-block-depth', str(self.max_block_depth),
            '-o', str(output_path)
        ]
        
        try:
            subprocess.run(cmd, timeout=self.timeout, capture_output=True, check=True)
            return TestProgram.from_file(output_path)
        except Exception as e:
            logger.warning(f"Csmith generation failed: {e}")
            return self._generate_fallback()
    
    def _generate_fallback(self) -> TestProgram:
        """Generate simple random program if Csmith not available."""
        return SimpleGenerator(self.working_dir).generate()


class YARPGenGenerator(ProgramGenerator):
    """
    Generate random C/C++ programs using YARPGen.
    
    YARPGen (Yet Another Random Program Generator) is Intel's tool
    for generating random programs for compiler testing.
    
    Requires YARPGen to be installed and in PATH.
    """
    
    def __init__(
        self,
        working_dir: Optional[Path] = None,
        yarpgen_path: str = "yarpgen",
        language: str = "c",
        timeout: int = 30
    ):
        super().__init__(working_dir)
        self.yarpgen_path = yarpgen_path
        self.language = language
        self.timeout = timeout
        
        self._available = shutil.which(yarpgen_path) is not None
        if not self._available:
            logger.warning("YARPGen not found in PATH. Install from https://github.com/intel/yarpgen")
    
    def generate(self) -> TestProgram:
        """Generate a random program using YARPGen."""
        if not self._available:
            return SimpleGenerator(self.working_dir).generate()
        
        name = self._next_name()
        output_dir = self.working_dir / name
        output_dir.mkdir(exist_ok=True)
        
        cmd = [
            self.yarpgen_path,
            f'--std={self.language}99',
            '-d', str(output_dir)
        ]
        
        try:
            subprocess.run(cmd, timeout=self.timeout, capture_output=True, check=True)
            
            # YARPGen creates multiple files, find the main one
            for f in output_dir.glob('*.c'):
                if 'driver' in f.name or 'func' in f.name:
                    continue
                return TestProgram.from_file(f)
            
            # Fallback to first C file
            c_files = list(output_dir.glob('*.c'))
            if c_files:
                return TestProgram.from_file(c_files[0])
                
        except Exception as e:
            logger.warning(f"YARPGen generation failed: {e}")
        
        return SimpleGenerator(self.working_dir).generate()


class TemplateGenerator(ProgramGenerator):
    """
    Generate programs from templates with parameter substitution.
    
    Useful for systematically exploring parameter spaces.
    """
    
    def __init__(
        self,
        template: str = None,
        parameters: Dict[str, range] = None,
        working_dir: Optional[Path] = None
    ):
        super().__init__(working_dir)
        
        self.template = template or self._default_template()
        self.parameters = parameters or {'N': range(1, 100)}
        self._param_iterators = {k: iter(v) for k, v in self.parameters.items()}
    
    def _default_template(self) -> str:
        return '''
#include <stdio.h>

int main() {
    int sum = 0;
    int arr[{N}];
    
    for (int i = 0; i < {N}; i++) {
        arr[i] = i * 2 + 1;
    }
    
    for (int i = 0; i < {N}; i++) {
        sum += arr[i];
    }
    
    printf("%d\\n", sum);
    return 0;
}
'''
    
    def generate(self) -> TestProgram:
        """Generate program with next parameter values."""
        # Get next values for each parameter
        values = {}
        for param, iterator in self._param_iterators.items():
            try:
                values[param] = next(iterator)
            except StopIteration:
                # Reset iterator
                self._param_iterators[param] = iter(self.parameters[param])
                values[param] = next(self._param_iterators[param])
        
        # Substitute parameters
        source = self.template
        for param, value in values.items():
            source = source.replace(f'{{{param}}}', str(value))
        
        name = f"template_{self._counter}"
        return TestProgram.from_source(source, name=name)


class SimpleGenerator(ProgramGenerator):
    """
    Generate simple random C programs for testing.
    
    Useful when external generators are not available.
    """
    
    def __init__(self, working_dir: Optional[Path] = None, seed: int = None):
        super().__init__(working_dir)
        import random
        self.rng = random.Random(seed)
    
    def generate(self) -> TestProgram:
        """Generate a simple random C program."""
        name = self._next_name()
        
        # Random parameters
        n = self.rng.randint(10, 1000)
        ops = self.rng.randint(1, 5)
        
        operations = []
        for i in range(ops):
            op_type = self.rng.choice(['add', 'mul', 'shift', 'xor'])
            if op_type == 'add':
                operations.append(f"sum += arr[i % {n}];")
            elif op_type == 'mul':
                operations.append(f"sum *= (arr[i % {n}] | 1);")
            elif op_type == 'shift':
                operations.append(f"sum ^= (sum << {self.rng.randint(1, 5)});")
            else:
                operations.append(f"sum ^= arr[i % {n}];")
        
        source = f'''
#include <stdio.h>

int main() {{
    int sum = 0;
    int arr[{n}];
    
    // Initialize array
    for (int i = 0; i < {n}; i++) {{
        arr[i] = i * {self.rng.randint(1, 100)} + {self.rng.randint(1, 50)};
    }}
    
    // Compute
    for (int i = 0; i < {n * 10}; i++) {{
        {chr(10).join("        " + op for op in operations)}
    }}
    
    printf("%d\\n", sum);
    return 0;
}}
'''
        
        return TestProgram.from_source(source, name=name)
