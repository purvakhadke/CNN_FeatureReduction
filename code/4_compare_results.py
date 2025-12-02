"""
Compare all results and create visualizations
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


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
    
    # Save combined results
    all_results.to_csv('../results/all_results.csv', index=False)
    print("✅ Combined results saved to '../results/all_results.csv'")
    
    # ============================================================
    # Create comparison plots
    # ============================================================
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Prepare data
    models = ['ResNet', 'Transformer', 'Autoencoder']
    raw_accuracies = []
    pca_accuracies = []
    raw_times = []
    pca_times = []
    
    for model in models:
        model_data = all_results[all_results['Model'] == model]
        
        raw_row = model_data[model_data['Input_Type'] == 'Raw']
        pca_row = model_data[model_data['Input_Type'].str.contains('PCA')]
        
        raw_accuracies.append(raw_row['Accuracy_%'].values[0] if len(raw_row) > 0 else 0)
        pca_accuracies.append(pca_row['Accuracy_%'].values[0] if len(pca_row) > 0 else 0)
        raw_times.append(raw_row['Training_Time_sec'].values[0] if len(raw_row) > 0 else 0)
        pca_times.append(pca_row['Training_Time_sec'].values[0] if len(pca_row) > 0 else 0)
    
    # Plot 1: Accuracy Comparison
    x = np.arange(len(models))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, raw_accuracies, width, label='Raw (3072-D)', alpha=0.8, color='steelblue')
    axes[0, 0].bar(x + width/2, pca_accuracies, width, label='PCA (512-D)', alpha=0.8, color='coral')
    axes[0, 0].set_ylabel('Test Accuracy (%)')
    axes[0, 0].set_title('Classification Accuracy: Raw vs PCA', fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models)
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Training Time Comparison
    axes[0, 1].bar(x - width/2, raw_times, width, label='Raw (3072-D)', alpha=0.8, color='steelblue')
    axes[0, 1].bar(x + width/2, pca_times, width, label='PCA (512-D)', alpha=0.8, color='coral')
    axes[0, 1].set_ylabel('Training Time (seconds)')
    axes[0, 1].set_title('Training Time: Raw vs PCA', fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models)
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Accuracy Delta (PCA - Raw)
    accuracy_delta = [pca - raw for pca, raw in zip(pca_accuracies, raw_accuracies)]
    colors = ['green' if d > 0 else 'red' for d in accuracy_delta]
    
    axes[1, 0].bar(models, accuracy_delta, alpha=0.8, color=colors)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 0].set_ylabel('Accuracy Change (%)')
    axes[1, 0].set_title('Impact of PCA on Accuracy (PCA - Raw)', fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: Time Speedup (Raw / PCA)
    time_speedup = [raw / pca if pca > 0 else 0 for raw, pca in zip(raw_times, pca_times)]
    
    axes[1, 1].bar(models, time_speedup, alpha=0.8, color='purple')
    axes[1, 1].axhline(y=1, color='black', linestyle='--', linewidth=1, label='No speedup')
    axes[1, 1].set_ylabel('Speedup Factor (Raw/PCA)')
    axes[1, 1].set_title('Training Speedup with PCA', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/comparison_plots.png', dpi=300, bbox_inches='tight')
    print("✅ Comparison plots saved to '../results/comparison_plots.png'")
    
    # ============================================================
    # Create summary table
    # ============================================================
    
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Model':<15} {'Input':<12} {'Accuracy':<12} {'Time (s)':<12} {'Δ Acc':<12}")
    print("-"*70)
    
    for model in models:
        model_data = all_results[all_results['Model'] == model]
        
        raw_row = model_data[model_data['Input_Type'] == 'Raw']
        pca_row = model_data[model_data['Input_Type'].str.contains('PCA')]
        
        if len(raw_row) > 0:
            print(f"{model:<15} {'Raw':<12} {raw_row['Accuracy_%'].values[0]:<12.2f} "
                  f"{raw_row['Training_Time_sec'].values[0]:<12.1f} {'-':<12}")
        
        if len(pca_row) > 0:
            delta = pca_row['Accuracy_%'].values[0] - raw_row['Accuracy_%'].values[0] if len(raw_row) > 0 else 0
            print(f"{model:<15} {'PCA':<12} {pca_row['Accuracy_%'].values[0]:<12.2f} "
                  f"{pca_row['Training_Time_sec'].values[0]:<12.1f} {delta:+.2f}{'%':<11}")
        
        print("-"*70)
    
    # ============================================================
    # Generate insights
    # ============================================================
    
    with open('../results/insights.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("EXPERIMENT INSIGHTS: PCA Impact on CIFAR-100 Classification\n")
        f.write("="*70 + "\n\n")
        
        f.write("Research Question:\n")
        f.write("How does PCA dimensionality reduction affect the performance of different\n")
        f.write("classifier architectures (ResNet, Transformer, Autoencoder) on CIFAR-100?\n\n")
        
        f.write("Key Findings:\n")
        f.write("-"*70 + "\n")
        
        for i, model in enumerate(models):
            f.write(f"\n{i+1}. {model}:\n")
            f.write(f"   Raw accuracy: {raw_accuracies[i]:.2f}%\n")
            f.write(f"   PCA accuracy: {pca_accuracies[i]:.2f}%\n")
            f.write(f"   Accuracy change: {accuracy_delta[i]:+.2f}%\n")
            f.write(f"   Training speedup: {time_speedup[i]:.2f}x\n")
            
            if accuracy_delta[i] > 0:
                f.write(f"   → PCA IMPROVED accuracy while being {time_speedup[i]:.1f}x faster\n")
            elif accuracy_delta[i] < -5:
                f.write(f"   → PCA significantly HURT accuracy (trade-off for {time_speedup[i]:.1f}x speedup)\n")
            else:
                f.write(f"   → PCA maintained similar accuracy with {time_speedup[i]:.1f}x speedup\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("Conclusions:\n")
        f.write("-"*70 + "\n")
        
        best_raw = models[np.argmax(raw_accuracies)]
        best_pca = models[np.argmax(pca_accuracies)]
        
        f.write(f"- Best model on raw data: {best_raw} ({max(raw_accuracies):.2f}%)\n")
        f.write(f"- Best model with PCA: {best_pca} ({max(pca_accuracies):.2f}%)\n")
        f.write(f"- Average accuracy change with PCA: {np.mean(accuracy_delta):+.2f}%\n")
        f.write(f"- Average training speedup with PCA: {np.mean(time_speedup):.2f}x\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print("\n✅ Insights saved to '../results/insights.txt'")
    print("\n" + "="*70)
    print("Analysis complete! Check ../results/ for all outputs.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()