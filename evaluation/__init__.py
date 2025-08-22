"""
Comprehensive model evaluation and analysis framework.

This package provides tools for thorough model evaluation including:
- Performance visualization (ROC curves, confusion matrices, metrics comparison)
- Overfitting/underfitting analysis with learning curves
- Automated report generation with timestamp management
- Integration with centralized directory management

Key modules:
- analyzer: Comprehensive evaluation runners and overfitting analysis
- visualization: Performance plots and automated report generation
"""

# Import main evaluation functions
try:
    from .analyzer import run_comprehensive_analysis, run_visualizations, run_overfitting_analysis
    from .visualization import ModelVisualizer, create_model_visualizations, generate_final_summary_report
    from .visualization.plots.learning_curves import OverfittingAnalyzer

    __all__ = [
        'run_comprehensive_analysis',
        'run_visualizations',
        'run_overfitting_analysis',
        'ModelVisualizer',
        'create_model_visualizations',
        'generate_final_summary_report',
        'OverfittingAnalyzer'
    ]
except ImportError as e:
    # Handle case where some dependencies might not be available
    print(f"Warning: Some evaluation components not available: {e}")
    __all__ = []