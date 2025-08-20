"""
Utility functions and shared infrastructure.

This package provides common utilities used across the project:
- Centralized directory management with timestamp organization
- Consistent model naming and display utilities
- File I/O helpers with proper error handling

Key modules:
- directory_management: Centralized file organization with timestamps
- name_display: Consistent model name mapping for displays

The directory management system creates organized output structure:
output/
  YYYYMMDD/
    results/         # JSON and pickle files
    evaluation_plots/ # All visualization outputs
"""

try:
    from .directory_management import (
        DirectoryManager, get_directory_manager,
        reset_directory_manager, save_results
    )
    from .name_display import get_model_names

    __all__ = [
        'DirectoryManager', 'get_directory_manager',
        'reset_directory_manager', 'save_results',
        'get_model_names'
    ]
except ImportError:
    __all__ = []