"""
Visualization tools for model performance analysis.

This subpackage provides comprehensive visualization capabilities:
- Performance metrics comparison charts
- ROC and Precision-Recall curves
- Confusion matrices with heatmaps
- Automated report generation with file management

Key modules:
- plots: Core visualization functions and ModelVisualizer class
- reports: Summary report generation with centralized file management
"""

try:
    from .plots import ModelVisualizer, create_model_visualizations
    from .reports import generate_final_summary_report

    __all__ = [
        'ModelVisualizer', 'create_model_visualizations',
        'generate_final_summary_report'
    ]
except ImportError:
    __all__ = []
