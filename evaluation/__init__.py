"""
Comprehensive model evaluation and analysis framework.

This package provides tools for thorough model evaluation including:
- Performance visualization (ROC curves, confusion matrices, metrics comparison)
- Overfitting/underfitting analysis with learning curves
- Automated report generation with timestamp management
- Integration with centralized directory management

Key subpackages:
- analyzers: Overfitting analysis and comprehensive evaluation runners
- visualizers: Performance plots and automated report generation

The evaluation pipeline creates:
1. Performance metrics and comparisons
2. Visual analysis (ROC, PR curves, confusion matrices)
3. Learning curves for overfitting detection
4. Final summary reports with timestamps
"""

# Import main evaluation functions
try:
    from .analyzers.run_analysis import run_comprehensive_analysis
    from .analyzers.overfitting import integrate_overfitting_analysis
    from .visualizers.plots import create_model_visualizations
    from .visualizers.reports import generate_final_summary_report

    __all__ = [
        'run_comprehensive_analysis',
        'create_model_visualizations',
        'generate_final_summary_report',
        'integrate_overfitting_analysis'
    ]
except ImportError:
    # Handle case where some dependencies might not be available
    __all__ = []
