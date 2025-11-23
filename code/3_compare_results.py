import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Load results
    ae_df = pd.read_csv('../results/autoencoder_results.csv')
    tf_df = pd.read_csv('../results/transformer_results.csv')

    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: MSE Comparison
    axes[0, 0].plot(ae_df['Latent_Dim'], ae_df['MSE_Loss'], marker='o', label='Autoencoder', linewidth=2)
    axes[0, 0].plot(tf_df['Latent_Dim'], tf_df['MSE_Loss'], marker='s', label='Transformer', linewidth=2)
    axes[0, 0].set_title('Reconstruction Loss (MSE) Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Latent Dimension')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Accuracy Comparison
    axes[0, 1].plot(ae_df['Latent_Dim'], ae_df['Accuracy'], marker='o', label='Autoencoder', linewidth=2)
    axes[0, 1].plot(tf_df['Latent_Dim'], tf_df['Accuracy'], marker='s', label='Transformer', linewidth=2)
    axes[0, 1].set_title('Classification Accuracy Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Latent Dimension')
    axes[0, 1].set_ylabel('Test Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Training Time Comparison
    axes[1, 0].plot(ae_df['Latent_Dim'], ae_df['Training_Time_sec'], marker='o', label='Autoencoder', linewidth=2)
    axes[1, 0].plot(tf_df['Latent_Dim'], tf_df['Training_Time_sec'], marker='s', label='Transformer', linewidth=2)
    axes[1, 0].set_title('Training Time Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Latent Dimension')
    axes[1, 0].set_ylabel('Training Time (seconds)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Accuracy Difference (Transformer - Autoencoder)
    acc_diff = tf_df['Accuracy'].values - ae_df['Accuracy'].values
    axes[1, 1].bar(range(len(ae_df)), acc_diff, color=['green' if x > 0 else 'red' for x in acc_diff])
    axes[1, 1].set_title('Accuracy Advantage: Transformer vs Autoencoder', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Latent Dimension')
    axes[1, 1].set_ylabel('Accuracy Difference (%)')
    axes[1, 1].set_xticks(range(len(ae_df)))
    axes[1, 1].set_xticklabels(ae_df['Latent_Dim'])
    axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('../results/comparison_all_methods.png', dpi=300)
    print("\n✅ Comparison plots saved to '../results/comparison_all_methods.png'")

    # Print summary table
    print("\n" + "="*60)
    print("COMPARISON SUMMARY TABLE")
    print("="*60)
    print(f"{'Dim':<8} {'Method':<15} {'MSE Loss':<12} {'Accuracy':<12} {'Time (s)':<10}")
    print("-"*60)
    for i in range(len(ae_df)):
        print(f"{ae_df['Latent_Dim'][i]:<8} {'Autoencoder':<15} {ae_df['MSE_Loss'][i]:<12.4f} {ae_df['Accuracy'][i]:<12.2f} {ae_df['Training_Time_sec'][i]:<10.1f}")
        print(f"{tf_df['Latent_Dim'][i]:<8} {'Transformer':<15} {tf_df['MSE_Loss'][i]:<12.4f} {tf_df['Accuracy'][i]:<12.2f} {tf_df['Training_Time_sec'][i]:<10.1f}")
        print("-"*60)


if __name__ == "__main__":
    main()
