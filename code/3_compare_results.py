import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Load results
    pca_df = pd.read_csv('../results/pca_results.csv')
    ae_df = pd.read_csv('../results/autoencoder_results.csv')
    tf_df = pd.read_csv('../results/transformer_results.csv')

    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: MSE Comparison
    axes[0, 0].plot(pca_df['Latent_Dim'], pca_df['MSE_Loss'], marker='^', label='PCA', linewidth=2)
    axes[0, 0].plot(ae_df['Latent_Dim'], ae_df['MSE_Loss'], marker='o', label='Autoencoder', linewidth=2)
    axes[0, 0].plot(tf_df['Latent_Dim'], tf_df['MSE_Loss'], marker='s', label='Transformer', linewidth=2)
    axes[0, 0].set_title('Reconstruction Loss (MSE) Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Latent Dimension')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Accuracy Comparison
    axes[0, 1].plot(pca_df['Latent_Dim'], pca_df['Accuracy'], marker='^', label='PCA', linewidth=2)
    axes[0, 1].plot(ae_df['Latent_Dim'], ae_df['Accuracy'], marker='o', label='Autoencoder', linewidth=2)
    axes[0, 1].plot(tf_df['Latent_Dim'], tf_df['Accuracy'], marker='s', label='Transformer', linewidth=2)
    axes[0, 1].set_title('Classification Accuracy Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Latent Dimension')
    axes[0, 1].set_ylabel('Test Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Training Time Comparison
    axes[1, 0].plot(pca_df['Latent_Dim'], pca_df['Training_Time_sec'], marker='^', label='PCA', linewidth=2)
    axes[1, 0].plot(ae_df['Latent_Dim'], ae_df['Training_Time_sec'], marker='o', label='Autoencoder', linewidth=2)
    axes[1, 0].plot(tf_df['Latent_Dim'], tf_df['Training_Time_sec'], marker='s', label='Transformer', linewidth=2)
    axes[1, 0].set_title('Training Time Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Latent Dimension')
    axes[1, 0].set_ylabel('Training Time (seconds)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: 3-Way Accuracy Comparison (Bar Chart)
    x = np.arange(len(pca_df))
    width = 0.25

    axes[1, 1].bar(x - width, pca_df['Accuracy'], width, label='PCA', alpha=0.8)
    axes[1, 1].bar(x, ae_df['Accuracy'], width, label='Autoencoder', alpha=0.8)
    axes[1, 1].bar(x + width, tf_df['Accuracy'], width, label='Transformer', alpha=0.8)
    axes[1, 1].set_title('Accuracy Comparison by Dimension', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Latent Dimension')
    axes[1, 1].set_ylabel('Test Accuracy (%)')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(pca_df['Latent_Dim'])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('../results/comparison_all_methods.png', dpi=300)
    print("\n✅ Comparison plots saved to '../results/comparison_all_methods.png'")

    # Print summary table
    print("\n" + "="*70)
    print("COMPARISON SUMMARY TABLE")
    print("="*70)
    print(f"{'Dim':<8} {'Method':<15} {'MSE Loss':<12} {'Accuracy':<12} {'Time (s)':<10}")
    print("-"*70)
    for i in range(len(pca_df)):
        print(f"{pca_df['Latent_Dim'][i]:<8} {'PCA':<15} {pca_df['MSE_Loss'][i]:<12.4f} {pca_df['Accuracy'][i]:<12.2f} {pca_df['Training_Time_sec'][i]:<10.1f}")
        print(f"{ae_df['Latent_Dim'][i]:<8} {'Autoencoder':<15} {ae_df['MSE_Loss'][i]:<12.4f} {ae_df['Accuracy'][i]:<12.2f} {ae_df['Training_Time_sec'][i]:<10.1f}")
        print(f"{tf_df['Latent_Dim'][i]:<8} {'Transformer':<15} {tf_df['MSE_Loss'][i]:<12.4f} {tf_df['Accuracy'][i]:<12.2f} {tf_df['Training_Time_sec'][i]:<10.1f}")
        print("-"*70)


if __name__ == "__main__":
    main()