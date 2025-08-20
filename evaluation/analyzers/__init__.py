"""
Model analysis tools for comprehensive evaluation.

This subpackage focuses on detecting and analyzing model behavior:
- Overfitting/underfitting detection through learning curves
- Comprehensive analysis pipeline integration
- Performance degradation analysis

Key modules:
- overfitting: Learning curve analysis and overfitting detection
- run_analysis: Main analysis pipeline coordination
"""

try:
    from .overfitting import OverfittingAnalyzer, integrate_overfitting_analysis
    from .run_analysis import run_comprehensive_analysis, run_visualizations, run_overfitting_analysis

    __all__ = [
        'OverfittingAnalyzer', 'integrate_overfitting_analysis',
        'run_comprehensive_analysis', 'run_visualizations', 'run_overfitting_analysis'
    ]
except ImportError:
    __all__ = []
