import matplotlib.pyplot as plt
import numpy as np

def plot_acc_prec_f1_metrics(results, classes):
    """
    Plot precision, F1, and accuracy per class with average lines.
    
    Args:
        results: Dictionary returned from evaluate_fairness()
        classes: List of class names
    """
    # Extract metrics
    precision = [results['precision'][c] for c in classes]
    f1 = [results['f1_score'][c] for c in classes]
    accuracy = [results['per_class_accuracy'][c] for c in classes]
    
    # Get averages
    avg_precision = results['macro_precision']
    avg_f1 = results['macro_f1']
    avg_accuracy = np.mean([a for a in accuracy if a is not None])
    
    # Set up the bar chart
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bars
    bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    bars2 = ax.bar(x, f1, width, label='F1 Score', alpha=0.8)
    bars3 = ax.bar(x + width, accuracy, width, label='Accuracy', alpha=0.8)
    
    # Add horizontal dotted lines for averages
    ax.axhline(y=avg_precision, color=bars1[0].get_facecolor(), 
               linestyle='--', linewidth=2, alpha=0.7,
               label=f'Avg Precision: {avg_precision:.2f}%')
    ax.axhline(y=avg_f1, color=bars2[0].get_facecolor(), 
               linestyle='--', linewidth=2, alpha=0.7,
               label=f'Avg F1: {avg_f1:.2f}%')
    ax.axhline(y=avg_accuracy, color=bars3[0].get_facecolor(), 
               linestyle='--', linewidth=2, alpha=0.7,
               label=f'Avg Accuracy: {avg_accuracy:.2f}%')
    
    # Customize plot
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.show()

def plot_bias_fairness_metrics(results, classes):
    """
    Plot demographic parity, TPR equal opportunity, and individual fairness metrics.
    
    Args:
        results: Dictionary returned from evaluate_fairness()
        classes: List of class names
    """
    # Extract metrics
    demographic_parity = [results['demographic_parity'][c] for c in classes]
    tpr = [results['TPR_equal_opportunity'][c] for c in classes]
    individual_fairness = [float(results['individual_fairness_proxy'][c]) for c in classes]
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # --- Subplot 1: Demographic Parity ---
    ax1 = axes[0]
    bars1 = ax1.bar(range(len(classes)), demographic_parity, alpha=0.8, color='#2E86AB')
    avg_dp = np.mean([d for d in demographic_parity if d is not None])
    ax1.axhline(y=avg_dp, color='red', linestyle='--', linewidth=2, alpha=0.7,
                label=f'Average: {avg_dp:.4f}')
    
    ax1.set_xlabel('Class', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Positive Prediction Rate', fontsize=11, fontweight='bold')
    ax1.set_title('Demographic Parity\n(Lower variance = more fair)', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(classes)))
    ax1.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(axis='y', alpha=0.3, linestyle=':')
    
    # --- Subplot 2: TPR Equal Opportunity ---
    ax2 = axes[1]
    bars2 = ax2.bar(range(len(classes)), tpr, alpha=0.8, color='#A23B72')
    avg_tpr = np.mean([t for t in tpr if t is not None])
    ax2.axhline(y=avg_tpr, color='red', linestyle='--', linewidth=2, alpha=0.7,
                label=f'Average: {avg_tpr:.4f}')
    
    ax2.set_xlabel('Class', fontsize=11, fontweight='bold')
    ax2.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    ax2.set_title('Equal Opportunity (TPR)\n(Higher & uniform = more fair)', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(classes)))
    ax2.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
    ax2.legend(loc='lower right', framealpha=0.9)
    ax2.grid(axis='y', alpha=0.3, linestyle=':')
    ax2.set_ylim(0, 1.05)
    
    # --- Subplot 3: Individual Fairness Proxy ---
    ax3 = axes[2]
    bars3 = ax3.bar(range(len(classes)), individual_fairness, alpha=0.8, color='#F18F01')
    avg_if = np.mean([f for f in individual_fairness if f is not None])
    ax3.axhline(y=avg_if, color='red', linestyle='--', linewidth=2, alpha=0.7,
                label=f'Average: {avg_if:.4f}')
    
    ax3.set_xlabel('Class', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Avg Pairwise Embedding Distance', fontsize=11, fontweight='bold')
    ax3.set_title('Individual Fairness Proxy\n(Lower variance = more fair)', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(len(classes)))
    ax3.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
    ax3.legend(loc='upper right', framealpha=0.9)
    ax3.grid(axis='y', alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("FAIRNESS METRICS SUMMARY")
    print("="*60)
    
    print(f"\nDemographic Parity:")
    print(f"  Mean: {avg_dp:.4f}")
    print(f"  Std Dev: {np.std(demographic_parity):.4f}")
    print(f"  Range: [{min(demographic_parity):.4f}, {max(demographic_parity):.4f}]")
    
    print(f"\nEqual Opportunity (TPR):")
    print(f"  Mean: {avg_tpr:.4f}")
    print(f"  Std Dev: {np.std(tpr):.4f}")
    print(f"  Range: [{min(tpr):.4f}, {max(tpr):.4f}]")
    
    print(f"\nIndividual Fairness Proxy:")
    print(f"  Mean: {avg_if:.4f}")
    print(f"  Std Dev: {np.std(individual_fairness):.4f}")
    print(f"  Range: [{min(individual_fairness):.4f}, {max(individual_fairness):.4f}]")
    print("="*60 + "\n")
