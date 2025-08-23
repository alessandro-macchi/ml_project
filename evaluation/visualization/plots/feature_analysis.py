import numpy as np
import matplotlib.pyplot as plt


def plot_misclassifications(visualizer_self, X_test, y_test, figsize=(16, 12), save_plots=False):
    """
    Analyze misclassified examples to understand model limitations

    Args:
        X_test: Test features
        y_test: Test labels
        figsize: Figure size for the plot
        save_plots (bool): Whether to save plots

    Returns:
        str: Path to saved plot or None
    """
    print("🔍 Creating Misclassification Analysis...")
    visualizer_self._save_enabled = save_plots

    # Convert data to arrays and extract feature names
    feature_names = _extract_feature_names(X_test)

    if hasattr(X_test, 'values'):
        X_test_array = X_test.values
    else:
        X_test_array = np.array(X_test)

    if hasattr(y_test, 'values'):
        y_test_array = y_test.values
    else:
        y_test_array = np.array(y_test)

    # Analyze each model
    n_models = len(visualizer_self.models)
    if n_models == 0:
        print("❌ No models available for misclassification analysis")
        return None

    # Create subplot grid
    cols = 2
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    if n_models == 1:
        axes = [axes]
    elif rows == 1 and n_models > 1:
        axes = list(axes)
    else:
        axes = axes.flatten() if n_models > 1 else [axes]

    model_keys = list(visualizer_self.models.keys())

    for i, (model_key, model) in enumerate(visualizer_self.models.items()):
        if i >= len(axes):  # Safety check
            break

        ax = axes[i]

        # Get predictions
        y_pred = model.predict(X_test)

        # Find misclassified examples
        misclassified_mask = (y_pred != y_test_array)

        if not np.any(misclassified_mask):
            ax.text(0.5, 0.5, 'No Misclassifications!',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=12, fontweight='bold')
            model_name = visualizer_self.model_names.get(model_key, model_key).replace(' (Custom)', '')
            ax.set_title(f'{model_name}', fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Get misclassified examples
        X_misclassified = X_test_array[misclassified_mask]
        y_true_misc = y_test_array[misclassified_mask]
        y_pred_misc = y_pred[misclassified_mask]

        # Feature importance analysis for misclassified examples
        feature_importance = _analyze_feature_patterns(
            visualizer_self, X_misclassified, y_true_misc, y_pred_misc, feature_names
        )

        # Plot feature importance for misclassifications
        top_features = feature_importance[:8]  # Top 8 features
        feature_names_short = [name[:15] + '...' if len(name) > 15 else name
                               for name, _ in top_features]
        importances = [imp for _, imp in top_features]

        if importances:  # Only plot if we have importance values
            bars = ax.barh(range(len(top_features)), importances, alpha=0.7)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(feature_names_short, fontsize=9)
            ax.set_xlabel('Misclassification Pattern Score', fontsize=10)

            # Add value labels
            max_importance = max(importances) if importances else 1
            for j, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.01 * max_importance, bar.get_y() + bar.get_height() / 2,
                        f'{width:.2f}', ha='left', va='center', fontsize=8)

            ax.grid(True, alpha=0.3, axis='x')

        model_name = visualizer_self.model_names.get(model_key, model_key).replace(' (Custom)', '')
        n_misc = len(X_misclassified)
        total = len(y_test_array)
        ax.set_title(f'{model_name}\n{n_misc}/{total} misclassified ({n_misc / total:.1%})',
                     fontweight='bold', fontsize=11)

    # Hide unused subplots
    for i in range(len(visualizer_self.models), len(axes)):
        if i < len(axes):
            axes[i].set_visible(False)

    plt.tight_layout()
    plt.suptitle('Misclassification Analysis - Feature Patterns',
                 fontsize=16, fontweight='bold', y=1.02)

    # Save plot
    saved_path = None
    if save_plots:
        saved_path = visualizer_self._save_figure('misclassification_analysis')

    plt.show()

    # Print summary statistics
    _print_misclassification_summary(visualizer_self, X_test_array, y_test_array)

    return saved_path


def _extract_feature_names(X_test):
    """
    Extract feature names from various data formats

    Args:
        X_test: Test features (pandas DataFrame, numpy array, etc.)

    Returns:
        list: List of feature names
    """
    # Try to get feature names from pandas DataFrame
    if hasattr(X_test, 'columns'):
        feature_names = X_test.columns.tolist()
        print(f"📊 Using DataFrame column names: {len(feature_names)} features")
        return feature_names

    # Try to get feature names from pandas Index
    elif hasattr(X_test, 'index') and hasattr(X_test.index, 'names'):
        if X_test.index.names and X_test.index.names[0] is not None:
            feature_names = list(X_test.index.names)
            print(f"📊 Using Index names: {len(feature_names)} features")
            return feature_names

    # Try to infer from shape and create meaningful names
    if hasattr(X_test, 'shape'):
        n_features = X_test.shape[1] if len(X_test.shape) > 1 else len(X_test)
    else:
        X_test_array = np.array(X_test)
        n_features = X_test_array.shape[1] if len(X_test_array.shape) > 1 else len(X_test_array)

    # Try common wine dataset feature names as fallback
    wine_feature_names = [
        'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
        'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
        'pH', 'sulphates', 'alcohol', 'quality'
    ]

    # If we have 11 or 12 features (typical for wine dataset), use wine names
    if n_features in [11, 12]:
        feature_names = wine_feature_names[:n_features]
        print(f"📊 Using wine dataset feature names: {len(feature_names)} features")
        return feature_names

    # Alternative common feature names for other datasets
    common_names = [
        'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5',
        'feature_6', 'feature_7', 'feature_8', 'feature_9', 'feature_10',
        'feature_11', 'feature_12', 'feature_13', 'feature_14', 'feature_15'
    ]

    if n_features <= len(common_names):
        feature_names = common_names[:n_features]
    else:
        feature_names = [f'feature_{i + 1}' for i in range(n_features)]

    print(f"📊 Using generated feature names: {len(feature_names)} features")
    return feature_names


def _analyze_feature_patterns(visualizer_self, X_misclassified, y_true, y_pred, feature_names):
    """Analyze feature patterns in misclassified examples"""
    feature_scores = []

    for i, feature_name in enumerate(feature_names):
        feature_values = X_misclassified[:, i]

        if len(feature_values) == 0:
            feature_scores.append((feature_name, 0.0))
            continue

        # Calculate various pattern indicators
        std_dev = np.std(feature_values)
        range_val = np.ptp(feature_values)  # peak-to-peak (max - min)
        mean_abs = np.mean(np.abs(feature_values))

        # Combine metrics for importance score
        # Higher variance and range often indicate problematic features
        importance_score = std_dev * 0.4 + range_val * 0.3 + mean_abs * 0.3

        feature_scores.append((feature_name, importance_score))

    # Sort by importance score (descending)
    feature_scores.sort(key=lambda x: x[1], reverse=True)

    return feature_scores


def _print_misclassification_summary(visualizer_self, X_test_array, y_test_array):
    """Print summary statistics for misclassifications"""
    print("\n📊 MISCLASSIFICATION SUMMARY")
    print("-" * 50)

    for model_key, model in visualizer_self.models.items():
        try:
            y_pred = model.predict(X_test_array)
            misclassified_mask = (y_pred != y_test_array)
            n_misc = np.sum(misclassified_mask)
            total = len(y_test_array)
            model_name = visualizer_self.model_names.get(model_key, model_key)

            print(f"{model_name}: {n_misc}/{total} ({n_misc / total:.1%}) misclassified")
        except Exception as e:
            print(f"{model_key}: Error computing misclassifications - {e}")