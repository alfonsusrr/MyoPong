import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

def plot_vaf():
    # File paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "activations.npy")
    output_path = os.path.join(script_dir, "vaf_by_n_synergies.png")

    # Load activations
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    print(f"Loading {data_path}...")
    activations = np.load(data_path)
    print(f"Loaded data with shape: {activations.shape}")

    # Initialize PCA
    n_features = activations.shape[1]
    print(f"Fitting PCA with {n_features} components...")
    pca = PCA(n_components=n_features)
    pca.fit(activations)

    # Calculate VAF (Cumulative Explained Variance Ratio)
    vaf = np.cumsum(pca.explained_variance_ratio_)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 14})  # Increase default font size
    plt.plot(range(1, n_features + 1), vaf, marker='o', linestyle='-', markersize=2, label='VAF')
    
    # Add horizontal lines for common benchmarks
    plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.7, label='90% VAF')
    plt.axhline(y=0.95, color='g', linestyle='--', alpha=0.7, label='95% VAF')
    
    # Add vertical line for 120 components cutoff
    cutoff = 120
    if cutoff <= n_features:
        vaf_at_cutoff = vaf[cutoff - 1]
        plt.axvline(x=cutoff, color='b', linestyle='-.', alpha=0.8, label=f'Cutoff ({cutoff} comp)')
        plt.plot(cutoff, vaf_at_cutoff, 'ko') # Mark the point
        plt.annotate(f'VAF={vaf_at_cutoff:.4f} @ {cutoff}', 
                     xy=(cutoff, vaf_at_cutoff), 
                     xytext=(cutoff + 10, vaf_at_cutoff - 0.05),
                     arrowprops=dict(facecolor='blue', shrink=0.05, width=1, headwidth=5),
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.5),
                     fontsize=12)

    # Find number of components for 90% and 95% VAF
    n_90 = np.argmax(vaf >= 0.9) + 1
    n_95 = np.argmax(vaf >= 0.95) + 1
    
    plt.annotate(f'90% @ {n_90} components', xy=(n_90, 0.9), xytext=(n_90 + 10, 0.85),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                 fontsize=12)
    plt.annotate(f'95% @ {n_95} components', xy=(n_95, 0.95), xytext=(n_95 + 10, 0.91),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                 fontsize=12)

    plt.title('Variance Accounted For (VAF) by Number of PCA Components', fontsize=16)
    plt.xlabel('Number of Components', fontsize=14)
    plt.ylabel('Cumulative VAF', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.ylim(0, 1.05)
    plt.xlim(0, n_features + 1)
    plt.legend(fontsize=12)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    
    # Show the plot if in an interactive environment
    # plt.show()

if __name__ == "__main__":
    plot_vaf()

