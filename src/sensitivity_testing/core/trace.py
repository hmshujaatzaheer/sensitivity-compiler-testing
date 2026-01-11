"""
Execution trace collection for compiler behavior analysis.

This module provides tools for collecting and analyzing execution traces
from compiled programs. Traces capture runtime behavior that can be
analyzed for sensitivity characteristics.
"""

import subprocess
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
import time
import struct

logger = logging.getLogger(__name__)


@dataclass
class ExecutionTrace:
    """
    Execution trace from a compiled program.
    
    Captures runtime behavior including:
    - Instruction counts
    - Memory access patterns
    - Branch decisions
    - Timing information
    """
    
    program_path: str
    """Path to the compiled executable."""
    
    exit_code: int
    """Program exit code."""
    
    execution_time_ms: float
    """Wall-clock execution time in milliseconds."""
    
    stdout: str
    """Standard output."""
    
    stderr: str
    """Standard error."""
    
    instruction_count: int = 0
    """Estimated instruction count (if available)."""
    
    memory_accesses: List[int] = field(default_factory=list)
    """Memory access pattern (addresses or hashes)."""
    
    branch_trace: List[int] = field(default_factory=list)
    """Branch taken/not-taken trace (1/0)."""
    
    timing_samples: List[float] = field(default_factory=list)
    """Fine-grained timing samples."""
    
    coverage_data: Optional[bytes] = None
    """Raw coverage data (if collected)."""
    
    def to_vector(self) -> np.ndarray:
        """
        Convert trace to numerical vector for analysis.
        
        Creates a fixed-length feature vector representing
        the execution trace, suitable for Lyapunov analysis.
        """
        features = []
        
        # Basic metrics
        features.append(self.exit_code)
        features.append(self.execution_time_ms)
        features.append(self.instruction_count)
        
        # Output hash (captures output behavior)
        output_hash = hash(self.stdout) % (2**32)
        features.append(output_hash)
        
        # Memory access statistics
        if self.memory_accesses:
            features.append(len(self.memory_accesses))
            features.append(np.mean(self.memory_accesses))
            features.append(np.std(self.memory_accesses))
        else:
            features.extend([0, 0, 0])
        
        # Branch statistics
        if self.branch_trace:
            features.append(len(self.branch_trace))
            features.append(np.mean(self.branch_trace))
            # Branch entropy
            p = np.mean(self.branch_trace)
            if 0 < p < 1:
                entropy = -p * np.log2(p) - (1-p) * np.log2(1-p)
            else:
                entropy = 0
            features.append(entropy)
        else:
            features.extend([0, 0, 0])
        
        # Timing statistics
        if self.timing_samples:
            features.append(np.mean(self.timing_samples))
            features.append(np.std(self.timing_samples))
            features.append(np.max(self.timing_samples) - np.min(self.timing_samples))
        else:
            features.extend([0, 0, 0])
        
        return np.array(features, dtype=np.float64)
    
    def to_dict(self) -> Dict:
        """Convert trace to dictionary."""
        return {
            'program_path': self.program_path,
            'exit_code': self.exit_code,
            'execution_time_ms': self.execution_time_ms,
            'stdout_length': len(self.stdout),
            'stderr_length': len(self.stderr),
            'instruction_count': self.instruction_count,
            'memory_access_count': len(self.memory_accesses),
            'branch_count': len(self.branch_trace),
            'timing_sample_count': len(self.timing_samples)
        }
    
    def compare(self, other: 'ExecutionTrace') -> Dict:
        """Compare two traces and return differences."""
        return {
            'exit_code_match': self.exit_code == other.exit_code,
            'output_match': self.stdout == other.stdout,
            'time_ratio': self.execution_time_ms / max(other.execution_time_ms, 0.001),
            'instruction_ratio': self.instruction_count / max(other.instruction_count, 1),
        }


class TraceCollector:
    """
    Collects execution traces from compiled programs.
    
    Supports multiple trace collection methods:
    - Basic execution (stdout, stderr, timing)
    - Instruction counting (via perf or simulation)
    - Coverage collection (via gcov/llvm-cov)
    - Memory profiling (via valgrind)
    
    Example:
        >>> collector = TraceCollector()
        >>> trace = collector.collect('/path/to/executable')
        >>> print(f"Execution time: {trace.execution_time_ms}ms")
    """
    
    def __init__(
        self,
        collect_coverage: bool = False,
        collect_memory: bool = False,
        use_perf: bool = False,
        working_dir: Optional[Path] = None
    ):
        """
        Initialize trace collector.
        
        Args:
            collect_coverage: Whether to collect code coverage
            collect_memory: Whether to profile memory accesses
            use_perf: Whether to use Linux perf for instruction counting
            working_dir: Working directory for temporary files
        """
        self.collect_coverage = collect_coverage
        self.collect_memory = collect_memory
        self.use_perf = use_perf
        self.working_dir = Path(working_dir) if working_dir else Path(tempfile.mkdtemp())
        
        # Check tool availability
        self._has_perf = self._check_tool('perf')
        self._has_valgrind = self._check_tool('valgrind')
        
        if use_perf and not self._has_perf:
            logger.warning("perf not available, falling back to basic timing")
        
    def collect(
        self,
        executable: Union[str, Path],
        args: List[str] = None,
        stdin_data: str = None,
        timeout: float = 30.0,
        env: Dict[str, str] = None
    ) -> ExecutionTrace:
        """
        Collect execution trace from a program.
        
        Args:
            executable: Path to executable
            args: Command-line arguments
            stdin_data: Data to pass to stdin
            timeout: Execution timeout in seconds
            env: Environment variables
            
        Returns:
            ExecutionTrace with collected data
        """
        executable = Path(executable)
        args = args or []
        
        if not executable.exists():
            raise FileNotFoundError(f"Executable not found: {executable}")
        
        # Ensure executable permission
        executable.chmod(executable.stat().st_mode | 0o111)
        
        # Collect basic trace
        trace = self._collect_basic(executable, args, stdin_data, timeout, env)
        
        # Optionally collect additional data
        if self.use_perf and self._has_perf:
            self._add_perf_data(trace, executable, args, stdin_data, timeout, env)
        
        if self.collect_memory and self._has_valgrind:
            self._add_memory_data(trace, executable, args, stdin_data, timeout, env)
        
        return trace
    
    def collect_differential(
        self,
        executables: List[Path],
        args: List[str] = None,
        stdin_data: str = None,
        timeout: float = 30.0
    ) -> Tuple[List[ExecutionTrace], List[Dict]]:
        """
        Collect traces from multiple executables and compare.
        
        Args:
            executables: List of executable paths
            args: Shared command-line arguments
            stdin_data: Data to pass to stdin
            timeout: Execution timeout
            
        Returns:
            Tuple of (traces, differences)
        """
        traces = []
        
        for exe in executables:
            try:
                trace = self.collect(exe, args, stdin_data, timeout)
                traces.append(trace)
            except Exception as e:
                logger.error(f"Failed to collect trace from {exe}: {e}")
                traces.append(None)
        
        # Compare adjacent traces
        differences = []
        for i in range(len(traces) - 1):
            if traces[i] and traces[i + 1]:
                diff = traces[i].compare(traces[i + 1])
                differences.append(diff)
            else:
                differences.append(None)
        
        return traces, differences
    
    def _collect_basic(
        self,
        executable: Path,
        args: List[str],
        stdin_data: Optional[str],
        timeout: float,
        env: Optional[Dict[str, str]]
    ) -> ExecutionTrace:
        """Collect basic execution trace."""
        cmd = [str(executable)] + args
        
        start_time = time.perf_counter()
        
        try:
            result = subprocess.run(
                cmd,
                input=stdin_data,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                cwd=str(self.working_dir)
            )
            
            execution_time = (time.perf_counter() - start_time) * 1000
            
            return ExecutionTrace(
                program_path=str(executable),
                exit_code=result.returncode,
                execution_time_ms=execution_time,
                stdout=result.stdout,
                stderr=result.stderr
            )
            
        except subprocess.TimeoutExpired:
            return ExecutionTrace(
                program_path=str(executable),
                exit_code=-1,
                execution_time_ms=timeout * 1000,
                stdout="",
                stderr="TIMEOUT"
            )
        except Exception as e:
            return ExecutionTrace(
                program_path=str(executable),
                exit_code=-2,
                execution_time_ms=0,
                stdout="",
                stderr=str(e)
            )
    
    def _add_perf_data(
        self,
        trace: ExecutionTrace,
        executable: Path,
        args: List[str],
        stdin_data: Optional[str],
        timeout: float,
        env: Optional[Dict[str, str]]
    ):
        """Add performance counter data using perf."""
        cmd = [
            'perf', 'stat', '-e', 'instructions,branches,branch-misses',
            '--', str(executable)
        ] + args
        
        try:
            result = subprocess.run(
                cmd,
                input=stdin_data,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env
            )
            
            # Parse perf output
            for line in result.stderr.split('\n'):
                if 'instructions' in line:
                    parts = line.strip().split()
                    if parts:
                        try:
                            trace.instruction_count = int(parts[0].replace(',', ''))
                        except ValueError:
                            pass
                            
        except Exception as e:
            logger.debug(f"perf collection failed: {e}")
    
    def _add_memory_data(
        self,
        trace: ExecutionTrace,
        executable: Path,
        args: List[str],
        stdin_data: Optional[str],
        timeout: float,
        env: Optional[Dict[str, str]]
    ):
        """Add memory profiling data using valgrind."""
        cmd = [
            'valgrind', '--tool=lackey', '--trace-mem=yes',
            str(executable)
        ] + args
        
        try:
            result = subprocess.run(
                cmd,
                input=stdin_data,
                capture_output=True,
                text=True,
                timeout=timeout * 2,  # Valgrind is slow
                env=env
            )
            
            # Parse memory accesses (simplified)
            accesses = []
            for line in result.stderr.split('\n'):
                if line.startswith(' L ') or line.startswith(' S '):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            addr = int(parts[1], 16)
                            accesses.append(addr % (2**16))  # Hash to manageable size
                        except ValueError:
                            pass
            
            trace.memory_accesses = accesses[:10000]  # Limit size
            
        except Exception as e:
            logger.debug(f"valgrind collection failed: {e}")
    
    def _check_tool(self, tool: str) -> bool:
        """Check if a tool is available."""
        import shutil
        return shutil.which(tool) is not None
    
    def generate_timing_samples(
        self,
        executable: Path,
        args: List[str] = None,
        num_samples: int = 100,
        timeout: float = 30.0
    ) -> List[float]:
        """
        Generate timing samples through repeated execution.
        
        Args:
            executable: Path to executable
            args: Command-line arguments
            num_samples: Number of timing samples
            timeout: Per-execution timeout
            
        Returns:
            List of execution times in milliseconds
        """
        args = args or []
        times = []
        
        for _ in range(num_samples):
            start = time.perf_counter()
            
            try:
                subprocess.run(
                    [str(executable)] + args,
                    capture_output=True,
                    timeout=timeout
                )
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
                
            except subprocess.TimeoutExpired:
                times.append(timeout * 1000)
            except Exception:
                pass
        
        return times


def compare_outputs(trace1: ExecutionTrace, trace2: ExecutionTrace) -> Dict:
    """
    Compare outputs of two execution traces.
    
    Returns:
        Dictionary with comparison results
    """
    return {
        'stdout_match': trace1.stdout == trace2.stdout,
        'stderr_match': trace1.stderr == trace2.stderr,
        'exit_code_match': trace1.exit_code == trace2.exit_code,
        'stdout_diff_ratio': _string_diff_ratio(trace1.stdout, trace2.stdout),
        'time_ratio': trace1.execution_time_ms / max(trace2.execution_time_ms, 0.001)
    }


def _string_diff_ratio(s1: str, s2: str) -> float:
    """Compute similarity ratio between two strings."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    
    from difflib import SequenceMatcher
    return SequenceMatcher(None, s1, s2).ratio()
