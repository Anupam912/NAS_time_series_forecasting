# visualization/visualize_results.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def save_plot(fig, filename, save_dir):
    """Helper function to save the plot as an image."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath)
    print(f"Plot saved at: {filepath}")

def plot_architecture_performance(results, save_dir=None):
    """
    Visualize the performance of different architectures.
    :param results: A list of dictionaries where each dictionary contains
                    'architecture', 'validation_loss' and other relevant info.
    :param save_dir: Directory to save the plots.
    """
    if not results:
        print("No results to plot.")
        return

    architectures = [r['architecture'] for r in results]
    validation_losses = [r['validation_loss'] for r in results]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=architectures, y=validation_losses)
    plt.title("Architecture Performance Comparison")
    plt.xlabel("Architecture")
    plt.ylabel("Validation Loss")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_dir:
        save_plot(plt.gcf(), 'architecture_performance.png', save_dir)
    
    plt.show()

def plot_hyperparameter_distribution(results, save_dir=None, top_k=5):
    """
    Visualize the distribution of hyperparameters for top-performing models.
    :param results: A list of dictionaries where each dictionary contains
                    'architecture', 'validation_loss' and hyperparameters.
    :param save_dir: Directory to save the plots.
    :param top_k: Number of top models to consider for hyperparameter distribution.
    """
    if not results:
        print("No results to plot.")
        return

    top_models = sorted(results, key=lambda x: x['validation_loss'])[:top_k]
    hyperparams = {k: [] for k in top_models[0].keys() if k != 'validation_loss' and k != 'architecture'}

    for model in top_models:
        for param in hyperparams.keys():
            hyperparams[param].append(model[param])

    plt.figure(figsize=(12, 8))
    for i, param in enumerate(hyperparams):
        plt.subplot(2, 3, i+1)
        sns.histplot(hyperparams[param], kde=True)
        plt.title(f"Distribution of {param}")

    plt.tight_layout()

    if save_dir:
        save_plot(plt.gcf(), 'hyperparameter_distribution.png', save_dir)
    
    plt.show()

def generate_comparison_report(results, save_dir=None):
    """
    Generate a comparison report of different architectures.
    :param results: A list of dictionaries where each dictionary contains
                    'architecture', 'validation_loss', 'learning_rate' and other relevant info.
    :param save_dir: Directory to save the report.
    """
    if not results:
        print("No results to generate report.")
        return

    df = pd.DataFrame(results)
    df = df.sort_values(by='validation_loss')  # Sort by validation loss

    # Print the report to the console
    print("\nArchitecture Comparison Report:")
    print(df[['architecture', 'validation_loss', 'learning_rate', 'num_layers']])

    # Save the report to a CSV file if a save directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        report_path = os.path.join(save_dir, 'architecture_comparison_report.csv')
        df.to_csv(report_path, index=False)
        print(f"\nComparison report saved at: {report_path}")
