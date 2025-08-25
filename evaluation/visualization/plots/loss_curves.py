import matplotlib.pyplot as plt

def plot_loss_curves(visualizer_self, figsize=(12, 8), save_plots=False):
    print("📈 Creating Loss Curves Analysis...")
    visualizer_self._save_enabled = save_plots

    models_with_losses = {}
    for model_key, model in visualizer_self.models.items():
        if hasattr(model, 'losses') and model.losses:
            models_with_losses[model_key] = model.losses

    if not models_with_losses:
        print("❌ No models with loss tracking found")
        return None

    plt.figure(figsize=figsize)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, (model_key, losses) in enumerate(models_with_losses.items()):
        model_name = visualizer_self.model_names.get(model_key, model_key).replace(' (Custom)', '')
        color = colors[i % len(colors)]

        epochs = range(1, len(losses) + 1)
        plt.plot(epochs, losses, color=color, linewidth=2.5,
                 label=model_name, marker='o', markersize=2, alpha=0.8)

        final_loss = losses[-1]
        plt.annotate(f'{final_loss:.4f}',
                     xy=(len(losses), final_loss),
                     xytext=(5, 0), textcoords="offset points",
                     ha='left', va='center', fontsize=9,
                     color=color, fontweight='bold')

    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title('Training Loss Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.tight_layout()

    saved_path = None
    if save_plots:
        saved_path = visualizer_self._save_figure('loss_curves')

    plt.show()
    return saved_path