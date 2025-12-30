import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_rewards(csv_path='merged_reward.csv', window_size=100):
    """
    Plots smoothed rewards with error bands from merged_reward.csv.
    Creates two figures: one for non-hierarchical models and one for hierarchical models.
    """
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run merge_csv.py first.")
        return

    # Read data
    df = pd.read_csv(csv_path)
    
    # Define model groups and their legend names
    non_h_models = {
        'ppo': 'PPO',
        'ppo_sarl': 'PPO w/ Synergy',
        'ppo_lattice': 'PPO w/ Lattice'
    }
    
    h_models = {
        'ppo_h': 'Hierarchical PPO',
        'ppo_h_sarl': 'Hierarchical PPO w/ Synergy',
        'ppo_h_lattice': 'Hierarchical PPO w/ Lattice'
    }

    # Plotting configuration for a more "publishable" look
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],
        'font.size': 20,
        'axes.labelsize': 22,
        'axes.titlesize': 24,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 16,
        'figure.titlesize': 26,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })
    
    # Colors for consistency
    # Using a vibrant but professional palette
    # Pairs: [Non-H, H] for each base color (Blue, Orange, Green)
    colors = [
        '#4c72b0', '#dd8452', '#55a868',      # Non-Hierarchical: Blue, Orange, Green
        '#254372', '#a85717', '#2b723a'       # Hierarchical: Darker Blue, Darker Orange, Darker Green
    ]

    def create_figure(models_dict, title, filename, plot_colors):
        fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
        
        # Add a very subtle gray background to the plot area
        ax.set_facecolor('#fcfcfc')
        
        for i, (col, label) in enumerate(models_dict.items()):
            if col not in df.columns:
                print(f"Warning: Column {col} not found in {csv_path}. Skipping.")
                continue
            
            # Use Step as X-axis and convert to Millions
            # Drop NaNs for the specific model column
            model_df = df[['Step', col]].dropna()
            steps = model_df['Step'] / 1e6
            data = model_df[col]
            
            # Calculate moving average and standard deviation
            smoothed = data.rolling(window=window_size, min_periods=1).mean()
            std = data.rolling(window=window_size, min_periods=1).std()
            
            # Plot moving average line
            ax.plot(steps, smoothed, label=label, color=plot_colors[i % len(plot_colors)], linewidth=2.0)
            
            # Plot error area (mean +/- std) with low transparency
            ax.fill_between(
                steps,
                smoothed - std,
                smoothed + std,
                color=plot_colors[i % len(plot_colors)],
                alpha=0.12
            )

        ax.set_title(title, pad=20, fontweight='bold')
        ax.set_xlabel('Training Steps (Millions)')
        ax.set_ylabel('Episode Reward')
        ax.legend(loc='upper left', frameon=True, shadow=False, framealpha=0.9)
        
        # Remove top and right spines for a cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Save figure in both PDF and PNG formats
        base_filename = os.path.splitext(filename)[0]
        plt.savefig(f"{base_filename}.pdf", bbox_inches='tight')
        plt.savefig(f"{base_filename}.png", bbox_inches='tight', dpi=300)
        print(f"Saved figures to {base_filename}.pdf and {base_filename}.png")
        # plt.show()

    # Create non-hierarchical figure
    create_figure(
        non_h_models, 
        'Non-Hierarchical Models Phase 1', 
        'reward_comparison_non_h.pdf',
        colors[:3]
    )
    
    # Create hierarchical figure
    create_figure(
        h_models, 
        'Hierarchical Models Phase 1', 
        'reward_comparison_h.pdf',
        colors[3:]
    )

    # Create combined figure
    create_figure(
        {**non_h_models, **h_models},
        'Phase 1 Training',
        'reward_comparison_all.pdf',
        colors
    )

if __name__ == "__main__":
    plot_rewards()

