"""
Compare all results and create visualizations
Includes: t-SNE, improved plots, summary table
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import os
from config import *


def plot_tsne_comparison(raw_data, pca_data, test_labels, save_path):
    """
    Create t-SNE visualizations comparing Raw vs PCA feature spaces
    This helps answer: Which representation produces more separable clusters?
    """
    print("\nGenerating t-SNE visualizations...")
    
    # Sample for speed
    n_samples = min(2000, len(test_labels))
    np.random.seed(42)
    indices = np.random.choice(len(test_labels), n_samples, replace=False)
    
    # Select 10 classes for clearer visualization
    unique_classes = np.unique(test_labels[indices])
    selected_classes = np.sort(np.random.choice(unique_classes, 10, replace=False))
    
    # Create mask for selected classes
    mask = np.isin(test_labels[indices], selected_classes)
    raw_subset = raw_data[indices][mask]
    pca_subset = pca_data[indices][mask]
    labels_subset = test_labels[indices][mask]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # t-SNE for Raw features
    print("  Computing t-SNE for raw features (this may take a minute)...")
    tsne_raw = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    embedded_raw = tsne_raw.fit_transform(raw_subset)
    
    # t-SNE for PCA features
    print("  Computing t-SNE for PCA features...")
    tsne_pca = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    embedded_pca = tsne_pca.fit_transform(pca_subset)
    
    # Plot with consistent colors per class
    cmap = plt.cm.get_cmap('tab10')
    
    for i, cls in enumerate(selected_classes):
        cls_mask = labels_subset == cls
        axes[0].scatter(embedded_raw[cls_mask, 0], embedded_raw[cls_mask, 1],
                       c=[cmap(i)], label=f'Class {cls}', alpha=0.6, s=25, edgecolors='none')
        axes[1].scatter(embedded_pca[cls_mask, 0], embedded_pca[cls_mask, 1],
                       c=[cmap(i)], label=f'Class {cls}', alpha=0.6, s=25, edgecolors='none')
    
    axes[0].set_title('Raw Features (3072-D)', fontsize=13, fontweight='bold')
    axes[1].set_title(f'PCA Features ({PCA_COMPONENTS}-D)', fontsize=13, fontweight='bold')
    
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('t-SNE dimension 1', fontsize=10)
        ax.set_ylabel('t-SNE dimension 2', fontsize=10)
    
    # Shared legend
    handles, labels_legend = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc='center right', bbox_to_anchor=(1.08, 0.5), 
               fontsize=9, title='Classes', title_fontsize=10)
    
    plt.suptitle('t-SNE Visualization: Raw vs PCA Feature Spaces\n(10 randomly selected classes, 2000 samples)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ t-SNE comparison saved to '{save_path}'")


def create_improved_plots(all_results, save_path):
    """Create enhanced comparison visualizations (3 plots)"""
    
    fig = plt.figure(figsize=(16, 5))
    
    # Prepare data
    models = ['ResNet', 'Transformer', 'Autoencoder']
    raw_acc, pca_acc, raw_time, pca_time = [], [], [], []
    
    for model in models:
        model_data = all_results[all_results['Model'] == model]
        raw_row = model_data[model_data['Input_Type'] == 'Raw']
        pca_row = model_data[model_data['Input_Type'].str.contains('PCA')]
        
        raw_acc.append(raw_row['Accuracy_%'].values[0] if len(raw_row) > 0 else 0)
        pca_acc.append(pca_row['Accuracy_%'].values[0] if len(pca_row) > 0 else 0)
        raw_time.append(raw_row['Training_Time_sec'].values[0] if len(raw_row) > 0 else 0)
        pca_time.append(pca_row['Training_Time_sec'].values[0] if len(pca_row) > 0 else 0)
    
    # Colors
    raw_color = '#2E86AB'   # Steel blue
    pca_color = '#E94F37'   # Coral
    
    # ============================================================
    # Plot 1: Accuracy Comparison (left)
    # ============================================================
    ax1 = fig.add_subplot(1, 3, 1)
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, raw_acc, width, label='Raw (3072-D)', 
                    color=raw_color, edgecolor='black', linewidth=0.8)
    bars2 = ax1.bar(x + width/2, pca_acc, width, label=f'PCA ({PCA_COMPONENTS}-D)', 
                    color=pca_color, edgecolor='black', linewidth=0.8)
    
    ax1.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('(a) Classification Accuracy', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=11)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_ylim(0, max(raw_acc + pca_acc) * 1.18)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 4), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 4), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
    
    # ============================================================
    # Plot 2: Training Time Comparison (middle)
    # ============================================================
    ax2 = fig.add_subplot(1, 3, 2)
    bars3 = ax2.bar(x - width/2, raw_time, width, label='Raw (3072-D)', 
                    color=raw_color, edgecolor='black', linewidth=0.8)
    bars4 = ax2.bar(x + width/2, pca_time, width, label=f'PCA ({PCA_COMPONENTS}-D)', 
                    color=pca_color, edgecolor='black', linewidth=0.8)
    
    ax2.set_ylabel('Training Time (seconds)', fontsize=11, fontweight='bold')
    ax2.set_title('(b) Training Time', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=11)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    
    # ============================================================
    # Plot 3: PCA Impact Summary (right)
    # ============================================================
    ax3 = fig.add_subplot(1, 3, 3)
    
    acc_delta = [pca - raw for pca, raw in zip(pca_acc, raw_acc)]
    time_reduction = [(raw - pca) / raw * 100 if raw > 0 else 0 for raw, pca in zip(raw_time, pca_time)]
    
    y_pos = np.arange(len(models))
    bar_height = 0.35
    
    # Accuracy change bars
    colors_acc = ['#27AE60' if d >= 0 else '#E74C3C' for d in acc_delta]
    bars_acc = ax3.barh(y_pos - bar_height/2, acc_delta, bar_height, 
                        label='Accuracy Change (%)', color=colors_acc, edgecolor='black', linewidth=0.8)
    
    # Time reduction bars
    bars_time = ax3.barh(y_pos + bar_height/2, time_reduction, bar_height, 
                         label='Time Reduction (%)', color='#3498DB', alpha=0.7, edgecolor='black', linewidth=0.8)
    
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(models, fontsize=11)
    ax3.set_xlabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax3.set_title('(c) PCA Impact Summary', fontsize=12, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(axis='x', alpha=0.3, linestyle='--')
    ax3.set_axisbelow(True)
    
    # Add value labels
    for i, (acc, time_red) in enumerate(zip(acc_delta, time_reduction)):
        # Accuracy label
        x_pos = acc + (1 if acc >= 0 else -1)
        ha = 'left' if acc >= 0 else 'right'
        ax3.text(x_pos, i - bar_height/2, f'{acc:+.1f}%', va='center', ha=ha, fontsize=9, fontweight='bold')
        # Time label
        ax3.text(time_red + 1, i + bar_height/2, f'{time_red:.0f}%', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Comparison plots saved to '{save_path}'")


def create_summary_table_figure(all_results, save_path):
    """Create a publication-ready summary table as an image"""
    
    models = ['ResNet', 'Transformer', 'Autoencoder']
    
    # Build table data
    table_data = []
    for model in models:
        model_data = all_results[all_results['Model'] == model]
        raw_row = model_data[model_data['Input_Type'] == 'Raw']
        pca_row = model_data[model_data['Input_Type'].str.contains('PCA')]
        
        if len(raw_row) > 0 and len(pca_row) > 0:
            raw_acc = raw_row['Accuracy_%'].values[0]
            pca_acc = pca_row['Accuracy_%'].values[0]
            raw_time = raw_row['Training_Time_sec'].values[0]
            pca_time = pca_row['Training_Time_sec'].values[0]
            
            table_data.append([
                model,
                f'{raw_acc:.2f}%',
                f'{pca_acc:.2f}%',
                f'{pca_acc - raw_acc:+.2f}%',
                f'{raw_time:.1f}s',
                f'{pca_time:.1f}s',
                f'{raw_time/pca_time:.2f}x' if pca_time > 0 else 'N/A'
            ])
    
    columns = ['Model', 'Raw Acc', 'PCA Acc', 'Δ Accuracy', 'Raw Time', 'PCA Time', 'Speedup']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.axis('off')
    
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        colColours=['#34495E'] * len(columns)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)
    
    # Style the table
    for i in range(len(columns)):
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_height(0.15)
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ECF0F1')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')
    
    plt.title('Summary: PCA Impact on CIFAR-100 Classification', 
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Summary table saved to '{save_path}'")


def main():
    print("\n" + "="*60)
    print("Generating Comparison Analysis")
    print("="*60 + "\n")
    
    os.makedirs('../results', exist_ok=True)
    
    # Load all results
    resnet_df = pd.read_csv('../results/resnet_results.csv')
    transformer_df = pd.read_csv('../results/transformer_results.csv')
    autoencoder_df = pd.read_csv('../results/autoencoder_results.csv')
    
    # Combine all results
    all_results = pd.concat([resnet_df, transformer_df, autoencoder_df], ignore_index=True)
    all_results.to_csv('../results/all_results.csv', index=False)
    print("✅ Combined results saved to '../results/all_results.csv'")
    
    # ============================================================
    # 1. Create improved comparison plots
    # ============================================================
    print("\n[1/4] Creating comparison plots...")
    create_improved_plots(all_results, '../results/comparison_plots.png')
    
    # ============================================================
    # 2. Create summary table figure
    # ============================================================
    print("\n[2/4] Creating summary table...")
    create_summary_table_figure(all_results, '../results/summary_table.png')
    
    # ============================================================
    # 3. t-SNE Visualization
    # ============================================================
    print("\n[3/4] Creating t-SNE visualization...")
    try:
        raw_data = np.load('../data/cifar100_raw.npz')
        pca_data = np.load(f'../data/cifar100_pca{PCA_COMPONENTS}.npz')
        
        plot_tsne_comparison(
            raw_data['test_features'],
            pca_data['test_features'],
            raw_data['test_labels'],
            '../results/tsne_comparison.png'
        )
    except Exception as e:
        print(f"  ⚠️ Could not generate t-SNE: {e}")
    
    # ============================================================
    # 4. Print and save summary
    # ============================================================
    print("\n[4/4] Generating summary report...")
    
    models = ['ResNet', 'Transformer', 'Autoencoder']
    raw_accuracies, pca_accuracies = [], []
    raw_times, pca_times = [], []
    
    for model in models:
        model_data = all_results[all_results['Model'] == model]
        raw_row = model_data[model_data['Input_Type'] == 'Raw']
        pca_row = model_data[model_data['Input_Type'].str.contains('PCA')]
        
        raw_accuracies.append(raw_row['Accuracy_%'].values[0] if len(raw_row) > 0 else 0)
        pca_accuracies.append(pca_row['Accuracy_%'].values[0] if len(pca_row) > 0 else 0)
        raw_times.append(raw_row['Training_Time_sec'].values[0] if len(raw_row) > 0 else 0)
        pca_times.append(pca_row['Training_Time_sec'].values[0] if len(pca_row) > 0 else 0)
    
    accuracy_delta = [pca - raw for pca, raw in zip(pca_accuracies, raw_accuracies)]
    time_speedup = [raw / pca if pca > 0 else 0 for raw, pca in zip(raw_times, pca_times)]
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Model':<15} {'Raw Acc':<12} {'PCA Acc':<12} {'Δ Acc':<12} {'Raw Time':<12} {'PCA Time':<12} {'Speedup':<10}")
    print("-"*80)
    
    for i, model in enumerate(models):
        print(f"{model:<15} {raw_accuracies[i]:<12.2f} {pca_accuracies[i]:<12.2f} "
              f"{accuracy_delta[i]:+11.2f} {raw_times[i]:<12.1f} {pca_times[i]:<12.1f} {time_speedup[i]:<10.2f}x")
    
    print("-"*80)
    
    # Save insights
    with open('../results/insights.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("EXPERIMENT INSIGHTS: PCA Impact on CIFAR-100 Classification\n")
        f.write("="*70 + "\n\n")
        
        f.write("RESEARCH QUESTION:\n")
        f.write("-"*70 + "\n")
        f.write("How does PCA dimensionality reduction (3072-D → 512-D) affect the\n")
        f.write("classification performance of ResNet, Transformer, and Autoencoder\n")
        f.write("architectures on CIFAR-100?\n\n")
        
        f.write("KEY FINDINGS:\n")
        f.write("-"*70 + "\n")
        
        for i, model in enumerate(models):
            f.write(f"\n{i+1}. {model}:\n")
            f.write(f"   Raw accuracy:     {raw_accuracies[i]:.2f}%\n")
            f.write(f"   PCA accuracy:     {pca_accuracies[i]:.2f}%\n")
            f.write(f"   Accuracy change:  {accuracy_delta[i]:+.2f}%\n")
            f.write(f"   Training speedup: {time_speedup[i]:.2f}x\n")
            
            if accuracy_delta[i] > 0:
                f.write(f"   → PCA IMPROVED accuracy while being {time_speedup[i]:.1f}x faster\n")
            elif accuracy_delta[i] > -2:
                f.write(f"   → PCA maintained accuracy (< 2% loss) with {time_speedup[i]:.1f}x speedup\n")
            elif accuracy_delta[i] > -5:
                f.write(f"   → Moderate accuracy trade-off for {time_speedup[i]:.1f}x speedup\n")
            else:
                f.write(f"   → Significant accuracy loss; PCA may remove important features\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("CONCLUSIONS:\n")
        f.write("-"*70 + "\n")
        
        best_raw = models[np.argmax(raw_accuracies)]
        best_pca = models[np.argmax(pca_accuracies)]
        
        f.write(f"• Best model on raw data: {best_raw} ({max(raw_accuracies):.2f}%)\n")
        f.write(f"• Best model with PCA:    {best_pca} ({max(pca_accuracies):.2f}%)\n")
        f.write(f"• Average accuracy change with PCA: {np.mean(accuracy_delta):+.2f}%\n")
        f.write(f"• Average training speedup with PCA: {np.mean(time_speedup):.2f}x\n\n")
        
        f.write("t-SNE INTERPRETATION:\n")
        f.write("-"*70 + "\n")
        f.write("The t-SNE visualization shows how well-separated the class clusters are\n")
        f.write("in each feature space. More distinct, tight clusters indicate better\n")
        f.write("separability and potentially easier classification.\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("GENERATED FILES:\n")
        f.write("-"*70 + "\n")
        f.write("• comparison_plots.png - Accuracy, time, and PCA impact charts\n")
        f.write("• summary_table.png    - Summary table as image\n")
        f.write("• tsne_comparison.png  - t-SNE visualization (Raw vs PCA)\n")
        f.write("• all_results.csv      - Combined numerical results\n")
        f.write("• insights.txt         - This report\n")
        f.write("="*70 + "\n")
    
    print("\n✅ Insights saved to '../results/insights.txt'")
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print("\nGenerated files in ../results/:")
    print("  • comparison_plots.png  - Accuracy, time & PCA impact")
    print("  • summary_table.png     - Summary table")
    print("  • tsne_comparison.png   - t-SNE visualization")
    print("  • all_results.csv       - Combined results")
    print("  • insights.txt          - Analysis report")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()