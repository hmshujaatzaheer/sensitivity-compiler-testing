"""
Command-line interface for sensitivity-theoretic compiler testing.

Usage:
    sct analyze program.c --compilers gcc,clang
    sct fuzz --generator csmith --budget 24h
    sct phase-detect --parameter loop-unroll --range 1:100
    sct visualize --input results.json --output landscape.png
"""

import click
import json
import logging
from pathlib import Path
from typing import List, Optional

from .framework import SensitivityFramework
from .utils.logging import setup_logging
from .utils.config import FrameworkConfig


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file')
@click.pass_context
def main(ctx, verbose, config):
    """
    Sensitivity-Theoretic Compiler Testing Framework.
    
    A novel approach to compiler testing using chaos theory and
    dynamical systems analysis to identify bug-prone regions.
    """
    ctx.ensure_object(dict)
    
    # Setup logging
    level = 'DEBUG' if verbose else 'INFO'
    setup_logging(level=level)
    
    # Load config if provided
    if config:
        ctx.obj['config'] = FrameworkConfig.from_file(Path(config))
    else:
        ctx.obj['config'] = FrameworkConfig()
    
    ctx.obj['verbose'] = verbose


@main.command()
@click.argument('program', type=click.Path(exists=True))
@click.option('--compilers', '-c', default='gcc,clang', help='Comma-separated list of compilers')
@click.option('--opt-levels', '-O', default='-O0,-O1,-O2,-O3', help='Optimization levels')
@click.option('--output', '-o', type=click.Path(), help='Output JSON file')
@click.pass_context
def analyze(ctx, program, compilers, opt_levels, output):
    """
    Analyze a single program for sensitivity characteristics.
    
    Example:
        sct analyze test.c --compilers gcc,clang --output results.json
    """
    config = ctx.obj['config']
    
    compiler_list = [c.strip() for c in compilers.split(',')]
    opt_list = [o.strip() for o in opt_levels.split(',')]
    
    click.echo(f"Analyzing {program}...")
    click.echo(f"Compilers: {compiler_list}")
    click.echo(f"Optimization levels: {opt_list}")
    
    try:
        framework = SensitivityFramework(
            compilers=compiler_list,
            optimization_levels=opt_list,
            config=config
        )
        
        result = framework.analyze(program)
        
        # Display results
        click.echo("\n" + "="*50)
        click.echo("ANALYSIS RESULTS")
        click.echo("="*50)
        click.echo(f"Lyapunov Exponent (λ): {result.lyapunov_exponent:.4f}")
        click.echo(f"Sensitivity Score: {result.sensitivity_score:.4f}")
        click.echo(f"Bug Probability: {result.bug_probability:.4f}")
        click.echo(f"Phase Transitions: {len(result.phase_transitions)}")
        click.echo(f"Analysis Time: {result.analysis_time_seconds:.2f}s")
        
        if result.is_bug_prone():
            click.secho("\n⚠️  HIGH SENSITIVITY DETECTED - Bug-prone region!", fg='red', bold=True)
        else:
            click.secho("\n✓ Low sensitivity - Stable region", fg='green')
        
        # Save results if output specified
        if output:
            with open(output, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            click.echo(f"\nResults saved to {output}")
            
    except Exception as e:
        click.secho(f"Error: {e}", fg='red')
        raise click.Abort()


@main.command()
@click.option('--generator', '-g', default='simple', 
              type=click.Choice(['csmith', 'yarpgen', 'simple', 'template']),
              help='Test program generator')
@click.option('--budget', '-b', default='1h', help='Time budget (e.g., 1h, 30m, 24h)')
@click.option('--compilers', '-c', default='gcc,clang', help='Compilers to test')
@click.option('--output', '-o', type=click.Path(), default='./bugs', help='Output directory')
@click.pass_context
def fuzz(ctx, generator, budget, compilers, output):
    """
    Run sensitivity-guided fuzzing campaign.
    
    Example:
        sct fuzz --generator csmith --budget 24h --output ./bugs
    """
    config = ctx.obj['config']
    
    # Parse budget
    budget_hours = _parse_budget(budget)
    compiler_list = [c.strip() for c in compilers.split(',')]
    
    click.echo(f"Starting fuzzing campaign...")
    click.echo(f"Generator: {generator}")
    click.echo(f"Budget: {budget_hours} hours")
    click.echo(f"Compilers: {compiler_list}")
    
    try:
        from .oracles import DifferentialOracle
        
        framework = SensitivityFramework(
            compilers=compiler_list,
            config=config
        )
        
        result = framework.run_prioritized_testing(
            test_generator=generator,
            budget_hours=budget_hours,
            oracle=DifferentialOracle(),
            output_dir=Path(output)
        )
        
        click.echo("\n" + "="*50)
        click.echo("CAMPAIGN RESULTS")
        click.echo("="*50)
        click.echo(f"Programs Tested: {result.programs_tested}")
        click.echo(f"Bugs Found: {len(result.bugs_found)}")
        click.echo(f"Bugs/Hour: {result.bugs_per_hour:.2f}")
        click.echo(f"Coverage Achieved: {result.coverage_achieved:.2%}")
        click.echo(f"Phase Transitions: {len(result.phase_transitions_discovered)}")
        
        if result.bugs_found:
            click.secho(f"\n🐛 Found {len(result.bugs_found)} bugs!", fg='yellow', bold=True)
            click.echo(f"Bug reports saved to {output}")
            
    except Exception as e:
        click.secho(f"Error: {e}", fg='red')
        raise click.Abort()


@main.command('phase-detect')
@click.option('--parameter', '-p', required=True, help='Parameter name to vary')
@click.option('--range', '-r', 'param_range', default='1:100', help='Parameter range (start:end)')
@click.option('--template', '-t', type=click.Path(exists=True), help='Program template file')
@click.option('--output', '-o', type=click.Path(), help='Output JSON file')
@click.pass_context
def phase_detect(ctx, parameter, param_range, template, output):
    """
    Detect phase transitions in compiler optimization behavior.
    
    Example:
        sct phase-detect --parameter loop_count --range 1:1000
    """
    # Parse range
    start, end = map(int, param_range.split(':'))
    
    click.echo(f"Scanning for phase transitions...")
    click.echo(f"Parameter: {parameter}")
    click.echo(f"Range: {start} to {end}")
    
    try:
        from .algorithms import PhaseTransitionDetector
        
        detector = PhaseTransitionDetector()
        
        # Use default template if not provided
        if template:
            template_code = Path(template).read_text()
        else:
            template_code = f'''
#include <stdio.h>
int main() {{
    int sum = 0;
    for (int i = 0; i < {{{parameter}}}; i++) sum += i;
    printf("%d\\n", sum);
    return 0;
}}
'''
        
        transitions = detector.detect(
            parameter_name=parameter,
            parameter_range=range(start, end),
            program_template=template_code
        )
        
        click.echo("\n" + "="*50)
        click.echo("PHASE TRANSITIONS DETECTED")
        click.echo("="*50)
        
        if transitions:
            for t in transitions:
                click.echo(f"\n  {parameter} = {t.parameter_value}")
                click.echo(f"    Type: {t.change_type.value}")
                click.echo(f"    Confidence: {t.confidence:.2%}")
                click.echo(f"    Magnitude: {t.magnitude:.4f}")
        else:
            click.echo("No phase transitions detected in the specified range.")
        
        if output:
            with open(output, 'w') as f:
                json.dump([t.to_dict() for t in transitions], f, indent=2)
            click.echo(f"\nResults saved to {output}")
            
    except Exception as e:
        click.secho(f"Error: {e}", fg='red')
        raise click.Abort()


@main.command()
@click.option('--input', '-i', 'input_file', required=True, type=click.Path(exists=True),
              help='Input JSON results file')
@click.option('--output', '-o', type=click.Path(), default='./plots', help='Output directory')
@click.option('--type', '-t', 'plot_type', default='landscape',
              type=click.Choice(['landscape', 'distribution', 'transitions', 'all']),
              help='Type of visualization')
@click.pass_context
def visualize(ctx, input_file, output, plot_type):
    """
    Generate visualizations from analysis results.
    
    Example:
        sct visualize --input results.json --output ./plots --type all
    """
    click.echo(f"Generating visualizations...")
    
    try:
        from .analysis import Visualizer, SensitivityLandscape
        
        # Load results
        with open(input_file) as f:
            data = json.load(f)
        
        viz = Visualizer(output_dir=Path(output))
        
        if plot_type in ['landscape', 'all']:
            # Create landscape from results
            landscape = SensitivityLandscape()
            # ... populate from data
            path = viz.plot_landscape(landscape)
            if path:
                click.echo(f"Landscape plot: {path}")
        
        if plot_type in ['distribution', 'all']:
            lyapunovs = [r.get('lyapunov_exponent', 0) for r in (data if isinstance(data, list) else [data])]
            path = viz.plot_lyapunov_distribution(lyapunovs)
            if path:
                click.echo(f"Distribution plot: {path}")
        
        click.echo(f"\nPlots saved to {output}")
        
    except Exception as e:
        click.secho(f"Error: {e}", fg='red')
        raise click.Abort()


@main.command()
def version():
    """Show version information."""
    from . import __version__
    click.echo(f"sensitivity-compiler-testing v{__version__}")


def _parse_budget(budget_str: str) -> float:
    """Parse budget string like '24h', '30m' to hours."""
    budget_str = budget_str.lower().strip()
    
    if budget_str.endswith('h'):
        return float(budget_str[:-1])
    elif budget_str.endswith('m'):
        return float(budget_str[:-1]) / 60
    elif budget_str.endswith('d'):
        return float(budget_str[:-1]) * 24
    else:
        return float(budget_str)


if __name__ == '__main__':
    main()
