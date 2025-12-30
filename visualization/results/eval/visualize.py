import matplotlib.pyplot as plt
import numpy as np
import os
import re
import matplotlib.patches as mpatches

def parse_eval_results(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return []

    with open(file_path, 'r') as f:
        content = f.read()

    # Find each block of evaluation results
    # Each block starts with 'Model:' and ends with '====' or end of file
    blocks = re.split(r'Model:', content)
    results = {}
    
    for block in blocks[1:]:  # Skip the part before the first 'Model:'
        # Prepend 'Model:' back since re.split removes it
        block = 'Model:' + block
        
        model_match = re.search(r'Model: (.*)', block)
        score_match = re.search(r'score\s+:\s+([\d.]+)\s+\+/-\s+([\d.]+)', block)
        effort_match = re.search(r'effort\s+:\s+([\d.]+)\s+\+/-\s+([\d.]+)', block)
        
        if model_match and score_match and effort_match:
            model_path = model_match.group(1)
            score_val = float(score_match.group(1))
            score_std = float(score_match.group(2))
            effort_val = float(effort_match.group(1))
            effort_std = float(effort_match.group(2))
            
            # Map model path to name
            model_name = ""
            short_name = ""
            if 'run-ppo-h-lattice' in model_path:
                model_name = "Hierarchical PPO w/ Lattice"
                short_name = "ppo_h_lattice"
            elif 'run-ppo-h-sarl' in model_path:
                model_name = "Hierarchical PPO w/ Synergy"
                short_name = "ppo_h_sarl"
            elif 'run-ppo-h' in model_path:
                model_name = "Hierarchical PPO"
                short_name = "ppo_h"
            elif 'run-ppo-lattice' in model_path:
                model_name = "PPO w/ Lattice"
                short_name = "ppo_lattice"
            elif 'run-ppo-sarl' in model_path:
                model_name = "PPO w/ Synergy"
                short_name = "ppo_sarl"
            elif 'run-ppo' in model_path:
                model_name = "PPO"
                short_name = "ppo"
            
            if short_name:
                results[short_name] = {
                    'name': model_name,
                    'short_name': short_name,
                    'score': score_val,
                    'score_std': score_std,
                    'effort': effort_val,
                    'effort_std': effort_std
                }
            
    print(f"Found results for: {', '.join(results.keys())}")
    return list(results.values())

def plot_on_ax(ax1, results, title):
    if not results:
        ax1.text(0.5, 0.5, f"No results for {title}", ha='center')
        return

    # Define the desired order and color mapping
    order = [
        'ppo', 'ppo_sarl', 'ppo_lattice',
        'ppo_h', 'ppo_h_sarl', 'ppo_h_lattice'
    ]
    
    color_map = {
        'ppo': '#4c72b0',          # Blue
        'ppo_sarl': '#dd8452',     # Orange
        'ppo_lattice': '#55a868',  # Green
        'ppo_h': '#254372',        # Dark Blue
        'ppo_h_sarl': '#a85717',   # Dark Orange
        'ppo_h_lattice': '#2b723a' # Dark Green
    }
    
    # Sort results according to the order
    sorted_results = []
    plot_colors = []
    for sn in order:
        for res in results:
            if res['short_name'] == sn:
                sorted_results.append(res)
                plot_colors.append(color_map.get(sn, 'gray'))
                break
    
    names = [res['name'] for res in sorted_results]
    scores = [res['score'] for res in sorted_results]
    score_stds = [res['score_std'] for res in sorted_results]
    efforts = [res['effort'] for res in sorted_results]
    effort_stds = [res['effort_std'] for res in sorted_results]
    
    x = np.arange(len(names))
    width = 0.35
    
    # Add subtle background
    ax1.set_facecolor('#fcfcfc')
    
    # Plot scores on ax1
    rects1 = ax1.bar(x - width/2, scores, width, yerr=score_stds, label='Success Rate', 
                    color=plot_colors, capsize=5, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Create ax2 for effort
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, efforts, width, yerr=effort_stds, label='Effort', 
                    color=plot_colors, capsize=5, alpha=0.5, hatch='//', edgecolor='black', linewidth=1)
    
    # Add numeric labels on top of bars
    def autolabel(rects, ax, is_effort=False):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=15, fontweight='bold')

    autolabel(rects1, ax1)
    autolabel(rects2, ax2, is_effort=True)

    # Customize axes
    ax1.set_ylabel('Success Rate', fontweight='bold', color='black', labelpad=10)
    ax2.set_ylabel('Effort', fontweight='bold', color='black', labelpad=10)
    
    ax1.set_title(title, pad=25, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha='right')
    
    # Set y-limits with some padding
    max_score = max(scores) if scores else 0
    if max_score < 0.1:
        ax1.set_ylim(0, max_score * 1.6 if max_score > 0 else 0.1)
    elif max_score < 0.5:
        ax1.set_ylim(0, 0.65)
    else:
        ax1.set_ylim(0, 1.2)
        
    ax2.set_ylim(0, max(efforts + [0.05]) * 1.4)
    
    # Legend (only for the top plot usually, but we can put it on both or just one)
    score_patch = mpatches.Patch(color='gray', alpha=0.8, label='Success Rate', edgecolor='black')
    effort_patch = mpatches.Patch(color='gray', alpha=0.5, label='Effort', hatch='//', edgecolor='black')
    ax1.legend(handles=[score_patch, effort_patch], loc='upper left', frameon=True, fontsize=12)
    
    # Remove top spines
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

if __name__ == "__main__":
    # Load results
    results_p1 = parse_eval_results('phase_1.txt')
    results_p2 = parse_eval_results('phase_2.txt')

    # Global plotting configuration
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],
        'font.size': 18,
        'axes.labelsize': 22,
        'axes.titlesize': 26,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 16,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(18, 16), dpi=100)
    
    plot_on_ax(ax_top, results_p1, "Phase 1 Evaluation")
    plot_on_ax(ax_bottom, results_p2, "Phase 2 Evaluation")

    # Use h_pad to reduce space between subplots
    plt.tight_layout(pad=2.0, h_pad=1.5)
    
    # Save combined plot
    output_filename = 'eval_comparison_combined'
    plt.savefig(f"{output_filename}.pdf", bbox_inches='tight')
    plt.savefig(f"{output_filename}.png", bbox_inches='tight', dpi=300)
    print(f"Saved combined figures to {output_filename}.pdf and {output_filename}.png")
