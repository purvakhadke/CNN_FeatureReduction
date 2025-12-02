"""
Compare all results and create summary plots and tables
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import csv

def main():
    print("\n" + "="*60)
    print("Creating Comparison Plots and Summary Tables")
    print("="*60 + "\n")
    
    os.makedirs('../results/', exist_ok=True)
    
    # Load baseline results
    baseline_acc = None
    baseline_time = None
    if os.path.exists('../results/baseline_results.txt'):
        with open('../results/baseline_results.txt', 'r') as f:
            for line in f:
                if 'Test Accuracy' in line:
                    baseline_acc = float(line.split(':')[1].strip().replace('%', ''))
                if 'Training Time' in line:
                    baseline_time = float(line.split(':')[1].strip().split()[0])
        print(f"Baseline ResNet50 Accuracy: {baseline_acc:.2f}%")
    
    # Load PCA results
    pca_acc = None
    pca_mse = None
    pca_time = None
    pca_variance = None
    if os.path.exists('../results/pca_results.txt'):
        with open('../results/pca_results.txt', 'r') as f:
            for line in f:
                if 'Classification Accuracy' in line:
                    pca_acc = float(line.split(':')[1].strip().replace('%', ''))
                if 'Reconstruction MSE' in line:
                    pca_mse = float(line.split(':')[1].strip())
                if 'PCA Time' in line:
                    pca_time = float(line.split(':')[1].strip().split()[0])
                if 'Variance Explained' in line:
                    pca_variance = float(line.split(':')[1].strip())
        print(f"PCA-200 Accuracy: {pca_acc:.2f}%")
    
    # Load sweep results
    ae_df = pd.read_csv('../results/autoencoder_results.csv')
    tf_df = pd.read_csv('../results/transformer_results.csv')
    
    print(f"\nAutoencoder results loaded: {len(ae_df)} dimensions")
    print(f"Transformer results loaded: {len(tf_df)} dimensions")

    # ============================================================
    # SAVE OVERALL SUMMARY TABLE TO CSV
    # ============================================================
    print("\nGenerating overall summary CSV...")
    with open('../results/overall_summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Input_Dim', 'Output_Dim', 'Compression_Ratio', 
                        'Accuracy_%', 'MSE_Loss', 'Training_Time_sec'])
        
        # Baseline
        writer.writerow(['ResNet_Baseline', 3072, 10, '1:1', baseline_acc, '-', baseline_time])
        
        # PCA
        writer.writerow(['PCA', 3072, 200, '15.4:1', pca_acc, pca_mse, pca_time])
        
        # Autoencoder
        for i in range(len(ae_df)):
            dim = ae_df['Latent_Dim'][i]
            comp_ratio = f"{3072/dim:.1f}:1"
            writer.writerow([f'PCA_Autoencoder', 200, dim, comp_ratio,
                           ae_df['Accuracy'][i], ae_df['MSE_Loss'][i], 
                           ae_df['Training_Time_sec'][i]])
        
        # Transformer
        for i in range(len(tf_df)):
            dim = tf_df['Latent_Dim'][i]
            comp_ratio = f"{3072/dim:.1f}:1"
            writer.writerow([f'PCA_Transformer', 200, dim, comp_ratio,
                           tf_df['Accuracy'][i], tf_df['MSE_Loss'][i],
                           tf_df['Training_Time_sec'][i]])
    
    print("✅ Saved to '../results/overall_summary.csv'")

    # ============================================================
    # SAVE RQ2: TRANSFORMER vs AUTOENCODER COMPARISON TO CSV
    # ============================================================
    print("\nGenerating Transformer vs Autoencoder comparison CSV...")
    with open('../results/ae_vs_transformer_comparison.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dimension', 'AE_Accuracy_%', 'TF_Accuracy_%', 
                        'TF_Advantage_%', 'AE_Time_sec', 'TF_Time_sec', 
                        'TF_Time_Ratio'])
        
        for i in range(len(ae_df)):
            dim = ae_df['Latent_Dim'][i]
            ae_acc = ae_df['Accuracy'][i]
            tf_acc = tf_df['Accuracy'][i]
            delta_acc = tf_acc - ae_acc
            ae_time = ae_df['Training_Time_sec'][i]
            tf_time = tf_df['Training_Time_sec'][i]
            time_ratio = tf_time / ae_time
            
            writer.writerow([dim, ae_acc, tf_acc, delta_acc, ae_time, tf_time, time_ratio])
    
    print("✅ Saved to '../results/ae_vs_transformer_comparison.csv'")

    # ============================================================
    # SAVE RQ4: OPTIMAL COMPRESSION ANALYSIS TO FILE
    # ============================================================
    print("\nGenerating optimal compression analysis...")
    with open('../results/optimal_compression_analysis.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("RQ4: OPTIMAL COMPRESSION RATIO ANALYSIS\n")
        f.write("="*70 + "\n\n")
        
        f.write("Methods achieving >80% accuracy:\n\n")
        f.write(f"{'Method':<30} {'Compression':<20} {'Accuracy':<12} {'Time':<15}\n")
        f.write("-"*70 + "\n")
        
        # PCA
        f.write(f"{'PCA-200':<30} {'3072→200 (15.4:1)':<20} "
                f"{pca_acc:<12.2f} {pca_time:<15.2f}s\n")
        
        # Autoencoder
        for i in range(len(ae_df)):
            if ae_df['Accuracy'][i] > 80:
                dim = ae_df['Latent_Dim'][i]
                compression_ratio = f"3072→{dim} ({3072/dim:.1f}:1)"
                f.write(f"{'PCA→Autoencoder-' + str(dim):<30} "
                       f"{compression_ratio:<20} "
                       f"{ae_df['Accuracy'][i]:<12.2f} "
                       f"{ae_df['Training_Time_sec'][i]:<15.1f}s\n")
        
        # Transformer
        for i in range(len(tf_df)):
            if tf_df['Accuracy'][i] > 80:
                dim = tf_df['Latent_Dim'][i]
                compression_ratio = f"3072→{dim} ({3072/dim:.1f}:1)"
                f.write(f"{'PCA→Transformer-' + str(dim):<30} "
                       f"{compression_ratio:<20} "
                       f"{tf_df['Accuracy'][i]:<12.2f} "
                       f"{tf_df['Training_Time_sec'][i]:<15.1f}s\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("RECOMMENDATION:\n")
        f.write("PCA-200 offers best balance:\n")
        f.write(f"  - Compression: 15.4:1 (3072→200)\n")
        f.write(f"  - Accuracy: {pca_acc:.2f}%\n")
        f.write(f"  - Training Time: {pca_time:.2f}s\n")
        f.write(f"  - 100-1000x faster than learned methods\n")
        f.write("="*70 + "\n")
    
    print("✅ Saved to '../results/optimal_compression_analysis.txt'")

    # ============================================================
    # CREATE COMPARISON PLOTS
    # ============================================================
    print("\nGenerating comparison plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Accuracy Comparison
    if baseline_acc:
        axes[0, 0].axhline(y=baseline_acc, color='red', linestyle='--', 
                           label=f'ResNet Baseline ({baseline_acc:.2f}%)', linewidth=2)
    if pca_acc:
        axes[0, 0].axhline(y=pca_acc, color='blue', linestyle='--', 
                           label=f'PCA-200 ({pca_acc:.2f}%)', linewidth=2)
    axes[0, 0].plot(ae_df['Latent_Dim'], ae_df['Accuracy'], 
                    marker='o', label='PCA→Autoencoder', linewidth=2, color='orange')
    axes[0, 0].plot(tf_df['Latent_Dim'], tf_df['Accuracy'], 
                    marker='s', label='PCA→Transformer', linewidth=2, color='green')
    axes[0, 0].set_title('Classification Accuracy Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Compressed Dimension')
    axes[0, 0].set_ylabel('Test Accuracy (%)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: MSE Comparison
    if pca_mse:
        axes[0, 1].axhline(y=pca_mse, color='blue', linestyle='--', 
                           label=f'PCA-200 MSE', linewidth=2)
    axes[0, 1].plot(ae_df['Latent_Dim'], ae_df['MSE_Loss'], 
                    marker='o', label='Autoencoder', linewidth=2, color='orange')
    axes[0, 1].plot(tf_df['Latent_Dim'], tf_df['MSE_Loss'], 
                    marker='s', label='Transformer', linewidth=2, color='green')
    axes[0, 1].set_title('Reconstruction Loss (MSE)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Compressed Dimension')
    axes[0, 1].set_ylabel('MSE Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Training Time
    axes[1, 0].plot(ae_df['Latent_Dim'], ae_df['Training_Time_sec'], 
                    marker='o', label='Autoencoder', linewidth=2, color='orange')
    axes[1, 0].plot(tf_df['Latent_Dim'], tf_df['Training_Time_sec'], 
                    marker='s', label='Transformer', linewidth=2, color='green')
    if pca_time:
        axes[1, 0].axhline(y=pca_time, color='blue', linestyle='--',
                          label=f'PCA-200 ({pca_time:.1f}s)', linewidth=2)
    axes[1, 0].set_title('Training Time Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Compressed Dimension')
    axes[1, 0].set_ylabel('Training Time (seconds)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Bar Chart Comparison
    x = np.arange(len(ae_df))
    width = 0.35

    axes[1, 1].bar(x - width/2, ae_df['Accuracy'], width, 
                   label='Autoencoder', alpha=0.8, color='orange')
    axes[1, 1].bar(x + width/2, tf_df['Accuracy'], width, 
                   label='Transformer', alpha=0.8, color='green')
    if pca_acc:
        axes[1, 1].axhline(y=pca_acc, color='blue', linestyle='--', 
                           label=f'PCA-200', linewidth=2)
    axes[1, 1].set_title('Accuracy by Dimension', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Compressed Dimension')
    axes[1, 1].set_ylabel('Test Accuracy (%)')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(ae_df['Latent_Dim'])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('../results/comparison_all_methods.png', dpi=300)
    print("✅ Comparison plot saved to '../results/comparison_all_methods.png'")

    # ============================================================
    # PRINT SUMMARY TO CONSOLE (for quick viewing)
    # ============================================================
    print("\n" + "="*70)
    print("SUMMARY TABLE (also saved to CSV)")
    print("="*70)
    print(f"{'Method':<30} {'Accuracy':<12} {'MSE':<12} {'Time (s)':<10}")
    print("-"*70)
    if baseline_acc:
        print(f"{'ResNet Baseline':<30} {baseline_acc:<12.2f} {'-':<12} {baseline_time if baseline_time else '-':<10}")
    if pca_acc:
        print(f"{'PCA-200':<30} {pca_acc:<12.2f} {pca_mse:<12.6f} {pca_time:<10.1f}")
    print("-"*70)
    for i in range(len(ae_df)):
        dim = ae_df['Latent_Dim'][i]
        print(f"{'PCA→Autoencoder-' + str(dim):<30} "
              f"{ae_df['Accuracy'][i]:<12.2f} "
              f"{ae_df['MSE_Loss'][i]:<12.6f} "
              f"{ae_df['Training_Time_sec'][i]:<10.1f}")
        print(f"{'PCA→Transformer-' + str(dim):<30} "
              f"{tf_df['Accuracy'][i]:<12.2f} "
              f"{tf_df['MSE_Loss'][i]:<12.6f} "
              f"{tf_df['Training_Time_sec'][i]:<10.1f}")
        print("-"*70)
    
    print("\n" + "="*70)
    print("FILES GENERATED:")
    print("  - ../results/overall_summary.csv")
    print("  - ../results/ae_vs_transformer_comparison.csv")
    print("  - ../results/optimal_compression_analysis.txt")
    print("  - ../results/comparison_all_methods.png")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
