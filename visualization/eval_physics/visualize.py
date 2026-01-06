import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.patches as mpatches

def plot_combined_physics(results):
    # Extract data
    difficulties = [res['difficulty'] for res in results]
    success_rates = [res['success_rate'] for res in results]
    success_stds = [res['success_std'] for res in results]
    paddle_rates = [res['paddle_rate'] for res in results]
    paddle_stds = [res['paddle_std'] for res in results]
    
    # Global plotting configuration
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 24,
        'axes.labelsize': 28,
        'axes.titlesize': 32,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 22,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })

    fig, ax = plt.subplots(figsize=(16, 10), dpi=100)
    ax.set_facecolor('#fcfcfc')
    
    x = np.arange(len(difficulties))
    width = 0.35
    
    # Blue for Success, Green for Paddle Hit
    color_success = '#4c72b0'
    color_paddle = '#55a868'
    
    # Plot bars
    rects1 = ax.bar(x - width/2, success_rates, width, yerr=success_stds, 
                    label='Success Rate', color=color_success, 
                    capsize=10, alpha=0.8, edgecolor='black', linewidth=2)
    
    rects2 = ax.bar(x + width/2, paddle_rates, width, yerr=paddle_stds, 
                    label='Paddle Hit Rate', color=color_paddle, 
                    capsize=10, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add numeric labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 8),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=18, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    
    ax.set_ylabel('Rate', fontweight='bold', labelpad=15)
    # ax.set_xlabel('Difficulty Level', fontweight='bold', labelpad=15)
    ax.set_title('Physics-Based HLP Performance across Difficulties', pad=30, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels([f'Difficulty {d}' for d in difficulties], fontweight='bold')
    
    ax.set_ylim(0, 1.2)
    
    # Legend
    ax.legend(loc='lower right', frameon=True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    
    output_filename = 'physics_eval_combined'
    plt.savefig(f"{output_filename}.png", bbox_inches='tight', dpi=300)
    plt.savefig(f"{output_filename}.pdf", bbox_inches='tight')
    print(f"Saved combined physics evaluation figures to {output_filename}.png and .pdf")

if __name__ == "__main__":
    # Hardcoded data from visualization/results/eval_physics/result.txt
    physics_results = [
        {
            'difficulty': 1,
            'success_rate': 1.000, 'success_std': 0.000,
            'paddle_rate': 1.000, 'paddle_std': 0.000
        },
        {
            'difficulty': 2,
            'success_rate': 0.975, 'success_std': 0.0078,
            'paddle_rate': 1.000, 'paddle_std': 0.000
        },
        {
            'difficulty': 3,
            'success_rate': 0.915, 'success_std': 0.0139,
            'paddle_rate': 1.000, 'paddle_std': 0.000
        }
    ]

    plot_combined_physics(physics_results)
