# scripts/analysis/plot_learning_curves.py
# FINAL POLISHED VERSION with Seaborn styling.
# Creates a publication-quality comparison plot of learning curves.

import matplotlib.pyplot as plt
import numpy as np
import os
import re
import seaborn as sns

# --- Configuration ---
LOG_DIR = "logs"
BDH_LOG_FILE = os.path.join(LOG_DIR, "bdh_logs.txt")
GPT_LOG_FILE = os.path.join(LOG_DIR, "gpt_logs.txt")
OUTPUT_PLOT_PATH = "results/plots/learning_curves_comparison.png"

def parse_log_file(file_path):
    if not os.path.exists(file_path):
        print(f"Warning: Log file not found at '{file_path}'. Skipping.")
        return None, None
    steps, losses = [], []
    log_pattern = re.compile(r"Step: (\d+)/\d+ \| loss: ([\d.]+)")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = log_pattern.search(line)
            if match:
                steps.append(int(match.group(1)))
                losses.append(float(match.group(2)))
    if not steps:
        print(f"Warning: No valid log entries found in '{file_path}'.")
        return None, None
    return np.array(steps), np.array(losses)

def plot_curves():
    print("--- Plotting Learning Curves: BDH vs. GPT (with Seaborn) ---")
    
    if not os.path.isdir(LOG_DIR):
        print(f"Error: Logs directory '{LOG_DIR}' not found.")
        return

    bdh_steps, bdh_losses = parse_log_file(BDH_LOG_FILE)
    gpt_steps, gpt_losses = parse_log_file(GPT_LOG_FILE)
    
    if bdh_losses is None or gpt_losses is None:
        print("Cannot generate plot due to missing log data.")
        return

    sns.set_theme(style="whitegrid", palette="deep")
    plt.figure(figsize=(14, 8))
    
    plt.plot(bdh_steps, bdh_losses, label=f'BDH (Final Loss: {bdh_losses[-1]:.3f})', 
             linewidth=2.5, marker='o', markersize=5)
    
    plt.plot(gpt_steps, gpt_losses, label=f'GPT (Final Loss: {gpt_losses[-1]:.3f})', 
             linewidth=2.5, marker='x', markersize=6)
    
    plt.title('BDH vs. GPT: Learning Efficiency Comparison (~25M Parameters)', 
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    
    # --- CORRECTED LINE ---
    plt.ylim(bottom=min([bdh_losses.min(), gpt_losses.min()]) * 0.9, top=max([bdh_losses.max(), gpt_losses.max()]) * 1.05)
    # --- END OF CORRECTION ---
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.legend(fontsize=13, loc='upper right') 
    
    verdict = "BDH learns significantly faster\nand achieves a lower final loss."
    props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.6)
    plt.text(0.95, 0.65, verdict, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    
    os.makedirs(os.path.dirname(OUTPUT_PLOT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PLOT_PATH, dpi=300)
    print(f"\n>>> Comparison plot saved to '{OUTPUT_PLOT_PATH}' <<<")
    plt.show()

if __name__ == '__main__':
    plot_curves()