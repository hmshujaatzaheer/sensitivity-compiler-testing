"""
Compiler abstraction layer for multi-compiler testing.

This module provides a unified interface for interacting with different
compilers (GCC, Clang, MSVC, ICC) and managing compilation processes.
"""

import subprocess
import shutil
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class CompilationResult:
    """Result of a compilation attempt."""
    
    success: bool
    executable_path: Optional[Path] = None
    return_code: int = 0
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None
    compilation_time_ms: float = 0.0
    compiler: str = ""
    optimization_level: str = ""
    warnings: List[str] = field(default_factory=list)
    
    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0
    
    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "executable_path": str(self.executable_path) if self.executable_path else None,
            "return_code": self.return_code,
            "error": self.error,
            "compilation_time_ms": self.compilation_time_ms,
            "compiler": self.compiler,
            "optimization_level": self.optimization_level,
            "warning_count": len(self.warnings)
        }


@dataclass
class CompilerInfo:
    """Information about an installed compiler."""
    
    name: str
    path: Path
    version: str
    supported_standards: List[str]
    supported_optimizations: List[str]
    
    @classmethod
    def detect(cls, compiler_name: str) -> Optional['CompilerInfo']:
        """Detect compiler information from system."""
        path = shutil.which(compiler_name)
        if not path:
            return None
        
        path = Path(path)
        
        # Get version
        try:
            result = subprocess.run(
                [str(path), '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            version_line = result.stdout.split('\n')[0]
        except Exception:
            version_line = "unknown"
        
        # Determine supported standards based on compiler
        if 'gcc' in compiler_name or 'g++' in compiler_name:
            standards = ['c89', 'c99', 'c11', 'c17', 'c2x', 'gnu89', 'gnu99', 'gnu11', 'gnu17']
            optimizations = ['-O0', '-O1', '-O2', '-O3', '-Os', '-Ofast', '-Og']
        elif 'clang' in compiler_name:
            standards = ['c89', 'c99', 'c11', 'c17', 'c2x', 'gnu89', 'gnu99', 'gnu11', 'gnu17']
            optimizations = ['-O0', '-O1', '-O2', '-O3', '-Os', '-Oz', '-Ofast']
        else:
            standards = ['c99', 'c11']
            optimizations = ['-O0', '-O1', '-O2', '-O3']
        
        return cls(
            name=compiler_name,
            path=path,
            version=version_line,
            supported_standards=standards,
            supported_optimizations=optimizations
        )


class CompilerManager:
    """
    Manages multiple compilers and provides unified compilation interface.
    
    Example:
        >>> manager = CompilerManager(['gcc', 'clang'])
        >>> result = manager.compile(program, 'gcc', '-O2')
        >>> if result.success:
        ...     print(f"Compiled to: {result.executable_path}")
    """
    
    def __init__(
        self,
        compilers: List[str],
        working_dir: Optional[Path] = None,
        timeout_seconds: int = 60
    ):
        """
        Initialize compiler manager.
        
        Args:
            compilers: List of compiler names to use
            working_dir: Directory for temporary files
            timeout_seconds: Compilation timeout
        """
        self.compilers: Dict[str, CompilerInfo] = {}
        self.working_dir = Path(working_dir) if working_dir else Path(tempfile.mkdtemp())
        self.timeout = timeout_seconds
        
        # Detect available compilers
        for compiler in compilers:
            info = CompilerInfo.detect(compiler)
            if info:
                self.compilers[compiler] = info
                logger.info(f"Detected {compiler}: {info.version}")
            else:
                logger.warning(f"Compiler not found: {compiler}")
        
        if not self.compilers:
            raise RuntimeError("No compilers available")
    
    def compile(
        self,
        program: 'TestProgram',
        compiler: str,
        optimization_level: str,
        extra_flags: Optional[List[str]] = None,
        standard: str = 'c11'
    ) -> CompilationResult:
        """
        Compile a test program with specified compiler and optimization.
        
        Args:
            program: TestProgram to compile
            compiler: Compiler name (e.g., 'gcc')
            optimization_level: Optimization flag (e.g., '-O2')
            extra_flags: Additional compiler flags
            standard: C standard to use
            
        Returns:
            CompilationResult with success status and paths
        """
        import time
        
        if compiler not in self.compilers:
            return CompilationResult(
                success=False,
                error=f"Compiler not available: {compiler}"
            )
        
        compiler_info = self.compilers[compiler]
        extra_flags = extra_flags or []
        
        # Generate unique output path
        program_hash = hashlib.md5(program.source_code.encode()).hexdigest()[:8]
        output_name = f"{program.name}_{compiler}_{optimization_level.replace('-', '')}_{program_hash}"
        output_path = self.working_dir / output_name
        
        # Build command
        cmd = [
            str(compiler_info.path),
            f'-std={standard}',
            optimization_level,
            '-o', str(output_path),
            str(program.path),
            '-lm',  # Math library
            *extra_flags
        ]
        
        # Add sanitizers for bug detection
        if '-fsanitize' not in ' '.join(extra_flags):
            cmd.extend(['-fsanitize=undefined', '-fno-sanitize-recover=all'])
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(self.working_dir)
            )
            
            compilation_time = (time.time() - start_time) * 1000
            
            # Parse warnings
            warnings = [
                line for line in result.stderr.split('\n')
                if 'warning:' in line.lower()
            ]
            
            if result.returncode == 0 and output_path.exists():
                return CompilationResult(
                    success=True,
                    executable_path=output_path,
                    return_code=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    compilation_time_ms=compilation_time,
                    compiler=compiler,
                    optimization_level=optimization_level,
                    warnings=warnings
                )
            else:
                return CompilationResult(
                    success=False,
                    return_code=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    error=result.stderr or "Compilation failed",
                    compilation_time_ms=compilation_time,
                    compiler=compiler,
                    optimization_level=optimization_level,
                    warnings=warnings
                )
                
        except subprocess.TimeoutExpired:
            return CompilationResult(
                success=False,
                error=f"Compilation timeout ({self.timeout}s)",
                compiler=compiler,
                optimization_level=optimization_level
            )
        except Exception as e:
            return CompilationResult(
                success=False,
                error=str(e),
                compiler=compiler,
                optimization_level=optimization_level
            )
    
    def compile_all_optimizations(
        self,
        program: 'TestProgram',
        compiler: str
    ) -> Dict[str, CompilationResult]:
        """Compile program with all optimization levels."""
        results = {}
        
        if compiler not in self.compilers:
            return results
        
        for opt in self.compilers[compiler].supported_optimizations:
            results[opt] = self.compile(program, compiler, opt)
        
        return results
    
    def get_assembly(
        self,
        program: 'TestProgram',
        compiler: str,
        optimization_level: str
    ) -> Optional[str]:
        """Get assembly output for analysis."""
        if compiler not in self.compilers:
            return None
        
        compiler_info = self.compilers[compiler]
        
        cmd = [
            str(compiler_info.path),
            '-S',  # Generate assembly
            optimization_level,
            '-o', '-',  # Output to stdout
            str(program.path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            if result.returncode == 0:
                return result.stdout
        except Exception as e:
            logger.warning(f"Assembly generation failed: {e}")
        
        return None
    
    def get_ir(
        self,
        program: 'TestProgram',
        compiler: str,
        optimization_level: str
    ) -> Optional[str]:
        """Get intermediate representation (LLVM IR for Clang)."""
        if compiler not in self.compilers:
            return None
        
        if 'clang' not in compiler:
            logger.warning("IR extraction only supported for Clang")
            return None
        
        compiler_info = self.compilers[compiler]
        
        cmd = [
            str(compiler_info.path),
            '-emit-llvm',
            '-S',
            optimization_level,
            '-o', '-',
            str(program.path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            if result.returncode == 0:
                return result.stdout
        except Exception as e:
            logger.warning(f"IR generation failed: {e}")
        
        return None
    
    @property
    def available_compilers(self) -> List[str]:
        """Get list of available compiler names."""
        return list(self.compilers.keys())
    
    def cleanup(self):
        """Clean up temporary files."""
        import shutil
        if self.working_dir.exists():
            shutil.rmtree(self.working_dir, ignore_errors=True)
