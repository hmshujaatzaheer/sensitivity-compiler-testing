"""
Main framework orchestration for sensitivity-theoretic compiler testing.

This module provides the SensitivityFramework class which coordinates
all components of the testing pipeline: trace collection, sensitivity
analysis, phase detection, and test prioritization.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time

from .core.compiler import CompilerManager, CompilationResult
from .core.trace import TraceCollector, ExecutionTrace
from .core.program import TestProgram, ProgramGenerator
from .algorithms.lyapunov import DiscreteLyapunov
from .algorithms.phase_transition import PhaseTransitionDetector
from .algorithms.sensitivity_oracle import SensitivityOracle
from .analysis.landscape import SensitivityLandscape
from .utils.config import FrameworkConfig
from .utils.metrics import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Results from analyzing a single test program."""
    
    program_path: str
    lyapunov_exponent: float
    phase_transitions: List[Dict]
    sensitivity_score: float
    bug_probability: float
    compilation_results: Dict[str, CompilationResult]
    traces: Dict[str, ExecutionTrace]
    analysis_time_seconds: float
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary for serialization."""
        return {
            "program_path": self.program_path,
            "lyapunov_exponent": self.lyapunov_exponent,
            "phase_transitions": self.phase_transitions,
            "sensitivity_score": self.sensitivity_score,
            "bug_probability": self.bug_probability,
            "analysis_time_seconds": self.analysis_time_seconds,
            "metadata": self.metadata
        }
    
    def is_bug_prone(self, threshold: float = 0.0) -> bool:
        """Check if program is in a bug-prone region."""
        return self.lyapunov_exponent > threshold


@dataclass
class CampaignResult:
    """Results from a testing campaign."""
    
    bugs_found: List[Dict]
    programs_tested: int
    total_time_hours: float
    coverage_achieved: float
    phase_transitions_discovered: List[Dict]
    sensitivity_landscape: Optional[SensitivityLandscape]
    
    @property
    def bugs_per_hour(self) -> float:
        """Calculate bug finding rate."""
        if self.total_time_hours == 0:
            return 0.0
        return len(self.bugs_found) / self.total_time_hours


class SensitivityFramework:
    """
    Main orchestration class for sensitivity-theoretic compiler testing.
    
    This framework coordinates:
    1. Test program generation/loading
    2. Multi-compiler compilation and execution
    3. Execution trace collection
    4. Sensitivity analysis (Lyapunov exponents)
    5. Phase transition detection
    6. Test prioritization with PAC bounds
    7. Bug detection via multiple oracles
    
    Example:
        >>> framework = SensitivityFramework(
        ...     compilers=['gcc', 'clang'],
        ...     optimization_levels=['-O0', '-O1', '-O2', '-O3']
        ... )
        >>> result = framework.analyze('test.c')
        >>> print(f"Sensitivity: {result.lyapunov_exponent}")
    """
    
    def __init__(
        self,
        compilers: List[str] = None,
        optimization_levels: List[str] = None,
        config: Optional[FrameworkConfig] = None,
        working_dir: Optional[Path] = None,
        num_workers: int = 4
    ):
        """
        Initialize the sensitivity testing framework.
        
        Args:
            compilers: List of compiler names (e.g., ['gcc', 'clang'])
            optimization_levels: List of optimization flags (e.g., ['-O0', '-O2'])
            config: Optional configuration object
            working_dir: Working directory for temporary files
            num_workers: Number of parallel workers for compilation
        """
        self.config = config or FrameworkConfig()
        self.compilers = compilers or ['gcc', 'clang']
        self.optimization_levels = optimization_levels or ['-O0', '-O1', '-O2', '-O3']
        self.working_dir = Path(working_dir) if working_dir else Path.cwd() / '.sct_work'
        self.num_workers = num_workers
        
        # Initialize components
        self.compiler_manager = CompilerManager(self.compilers)
        self.trace_collector = TraceCollector()
        self.lyapunov_analyzer = DiscreteLyapunov(
            embedding_dimension=self.config.embedding_dimension,
            time_delay=self.config.time_delay,
            min_neighbors=self.config.min_neighbors
        )
        self.phase_detector = PhaseTransitionDetector(
            method=self.config.phase_detection_method,
            significance_level=self.config.significance_level
        )
        self.sensitivity_oracle = SensitivityOracle(
            epsilon=self.config.epsilon,
            delta=self.config.delta
        )
        self.metrics = MetricsCollector()
        
        # Ensure working directory exists
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized SensitivityFramework with compilers: {self.compilers}")
    
    def analyze(
        self,
        program: Union[str, Path, TestProgram],
        collect_traces: bool = True,
        detect_phases: bool = True
    ) -> AnalysisResult:
        """
        Analyze a single test program for sensitivity characteristics.
        
        Args:
            program: Path to program or TestProgram object
            collect_traces: Whether to collect execution traces
            detect_phases: Whether to detect phase transitions
            
        Returns:
            AnalysisResult containing sensitivity metrics
        """
        start_time = time.time()
        
        # Load program
        if isinstance(program, (str, Path)):
            program = TestProgram.from_file(program)
        
        logger.info(f"Analyzing program: {program.path}")
        
        # Compile with all compiler/optimization combinations
        compilation_results = self._compile_all_variants(program)
        
        # Collect execution traces
        traces = {}
        if collect_traces:
            traces = self._collect_traces(program, compilation_results)
        
        # Compute Lyapunov exponent
        lyapunov_exponent = self._compute_lyapunov(traces)
        
        # Detect phase transitions
        phase_transitions = []
        if detect_phases:
            phase_transitions = self._detect_phase_transitions(program, traces)
        
        # Compute overall sensitivity score
        sensitivity_score = self._compute_sensitivity_score(
            lyapunov_exponent, phase_transitions
        )
        
        # Estimate bug probability
        bug_probability = self.sensitivity_oracle.estimate_bug_probability(
            sensitivity_score
        )
        
        analysis_time = time.time() - start_time
        
        result = AnalysisResult(
            program_path=str(program.path),
            lyapunov_exponent=lyapunov_exponent,
            phase_transitions=phase_transitions,
            sensitivity_score=sensitivity_score,
            bug_probability=bug_probability,
            compilation_results=compilation_results,
            traces=traces,
            analysis_time_seconds=analysis_time,
            metadata={
                "compilers": self.compilers,
                "optimization_levels": self.optimization_levels
            }
        )
        
        self.metrics.record_analysis(result)
        logger.info(f"Analysis complete: λ={lyapunov_exponent:.4f}, P(bug)={bug_probability:.4f}")
        
        return result
    
    def run_prioritized_testing(
        self,
        test_generator: Union[str, ProgramGenerator],
        budget_hours: float,
        oracle: 'BaseOracle',
        output_dir: Optional[Path] = None
    ) -> CampaignResult:
        """
        Run a sensitivity-guided testing campaign.
        
        Args:
            test_generator: Generator name ('csmith', 'yarpgen') or custom generator
            budget_hours: Time budget in hours
            oracle: Bug detection oracle to use
            output_dir: Directory for saving found bugs
            
        Returns:
            CampaignResult with found bugs and statistics
        """
        start_time = time.time()
        output_dir = Path(output_dir) if output_dir else self.working_dir / 'bugs'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting {budget_hours}h testing campaign with {test_generator}")
        
        # Initialize generator
        if isinstance(test_generator, str):
            generator = ProgramGenerator.create(test_generator)
        else:
            generator = test_generator
        
        bugs_found = []
        programs_tested = 0
        phase_transitions_discovered = []
        sensitivity_landscape = SensitivityLandscape()
        
        # Main testing loop
        budget_seconds = budget_hours * 3600
        while (time.time() - start_time) < budget_seconds:
            # Generate batch of test programs
            batch = generator.generate_batch(size=self.config.batch_size)
            
            # Analyze and prioritize
            analysis_results = []
            for program in batch:
                try:
                    result = self.analyze(program, detect_phases=False)
                    analysis_results.append((program, result))
                    sensitivity_landscape.add_point(result)
                except Exception as e:
                    logger.warning(f"Analysis failed for {program}: {e}")
            
            # Sort by sensitivity (highest first)
            analysis_results.sort(key=lambda x: x[1].sensitivity_score, reverse=True)
            
            # Test prioritized programs
            for program, analysis in analysis_results:
                programs_tested += 1
                
                # Run oracle to detect bugs
                bug = oracle.check(program, analysis.compilation_results)
                
                if bug:
                    bug['sensitivity_score'] = analysis.sensitivity_score
                    bug['lyapunov_exponent'] = analysis.lyapunov_exponent
                    bugs_found.append(bug)
                    
                    # Save bug-triggering program
                    bug_path = output_dir / f"bug_{len(bugs_found):04d}.c"
                    program.save(bug_path)
                    
                    logger.info(f"Bug #{len(bugs_found)} found! λ={analysis.lyapunov_exponent:.4f}")
                
                # Check time budget
                if (time.time() - start_time) >= budget_seconds:
                    break
            
            # Periodically detect phase transitions on high-sensitivity programs
            if programs_tested % 100 == 0:
                high_sensitivity = [r for _, r in analysis_results if r.lyapunov_exponent > 0]
                if high_sensitivity:
                    for result in high_sensitivity[:5]:
                        phases = self._detect_phase_transitions(
                            TestProgram.from_file(result.program_path),
                            result.traces
                        )
                        phase_transitions_discovered.extend(phases)
        
        total_time_hours = (time.time() - start_time) / 3600
        
        return CampaignResult(
            bugs_found=bugs_found,
            programs_tested=programs_tested,
            total_time_hours=total_time_hours,
            coverage_achieved=sensitivity_landscape.coverage_estimate(),
            phase_transitions_discovered=phase_transitions_discovered,
            sensitivity_landscape=sensitivity_landscape
        )
    
    def detect_phase_transitions(
        self,
        parameter_name: str,
        parameter_range: range,
        program_template: str
    ) -> List[Dict]:
        """
        Scan a parameter space to detect phase transitions.
        
        Args:
            parameter_name: Name of the parameter to vary
            parameter_range: Range of parameter values to test
            program_template: Program template with {parameter_name} placeholder
            
        Returns:
            List of detected phase transitions
        """
        return self.phase_detector.detect(
            parameter_name=parameter_name,
            parameter_range=parameter_range,
            program_template=program_template,
            compiler_manager=self.compiler_manager
        )
    
    def get_required_test_budget(
        self,
        target_coverage: float,
        sensitivity_scores: List[float]
    ) -> int:
        """
        Calculate required number of tests for target coverage.
        
        Uses PAC learning bounds to estimate the minimum test budget
        needed to achieve the specified coverage with high confidence.
        
        Args:
            target_coverage: Desired coverage (0.0 to 1.0)
            sensitivity_scores: List of sensitivity scores from analyzed programs
            
        Returns:
            Estimated number of tests required
        """
        return self.sensitivity_oracle.required_tests(
            sensitivity_scores=sensitivity_scores,
            target_coverage=target_coverage
        )
    
    def _compile_all_variants(
        self,
        program: TestProgram
    ) -> Dict[str, CompilationResult]:
        """Compile program with all compiler/optimization combinations."""
        results = {}
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {}
            
            for compiler in self.compilers:
                for opt_level in self.optimization_levels:
                    key = f"{compiler}_{opt_level}"
                    future = executor.submit(
                        self.compiler_manager.compile,
                        program,
                        compiler,
                        opt_level
                    )
                    futures[future] = key
            
            for future in as_completed(futures):
                key = futures[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    logger.error(f"Compilation failed for {key}: {e}")
                    results[key] = CompilationResult(success=False, error=str(e))
        
        return results
    
    def _collect_traces(
        self,
        program: TestProgram,
        compilation_results: Dict[str, CompilationResult]
    ) -> Dict[str, ExecutionTrace]:
        """Collect execution traces for all compiled variants."""
        traces = {}
        
        for key, result in compilation_results.items():
            if result.success and result.executable_path:
                try:
                    trace = self.trace_collector.collect(
                        result.executable_path,
                        timeout=self.config.execution_timeout
                    )
                    traces[key] = trace
                except Exception as e:
                    logger.warning(f"Trace collection failed for {key}: {e}")
        
        return traces
    
    def _compute_lyapunov(self, traces: Dict[str, ExecutionTrace]) -> float:
        """Compute Lyapunov exponent from execution traces."""
        if not traces:
            return 0.0
        
        trace_vectors = [t.to_vector() for t in traces.values()]
        return self.lyapunov_analyzer.compute(trace_vectors)
    
    def _detect_phase_transitions(
        self,
        program: TestProgram,
        traces: Dict[str, ExecutionTrace]
    ) -> List[Dict]:
        """Detect phase transitions in the program's behavior."""
        if not traces:
            return []
        
        return self.phase_detector.detect_from_traces(traces)
    
    def _compute_sensitivity_score(
        self,
        lyapunov: float,
        phase_transitions: List[Dict]
    ) -> float:
        """Compute overall sensitivity score combining multiple factors."""
        # Base score from Lyapunov exponent
        base_score = max(0, lyapunov)
        
        # Bonus for proximity to phase transitions
        phase_bonus = len(phase_transitions) * 0.1
        
        # Combined score (normalized to 0-1 range using sigmoid)
        import math
        raw_score = base_score + phase_bonus
        normalized = 1 / (1 + math.exp(-raw_score))
        
        return normalized
    
    def save_results(self, results: List[AnalysisResult], output_path: Path):
        """Save analysis results to JSON file."""
        data = [r.to_dict() for r in results]
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(results)} results to {output_path}")
    
    def load_results(self, input_path: Path) -> List[Dict]:
        """Load analysis results from JSON file."""
        with open(input_path, 'r') as f:
            return json.load(f)
