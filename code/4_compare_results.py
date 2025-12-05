# plots and tables and any other diagrams we might need, ADD it HERE
# make sure to run this file independently if the data is already loaded in results (if you are only making plots and tables w that same data, so you domt have to regenerate data and wait for code to run)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import os
from config import *

# used llm to help create nicer looking plots w setting colors/fonts
COLORS = {
    'Raw':  '#2C3E50',
    'PCA':  '#C0392B',
    'UMAP': '#2980B9',
    'AE':   '#27AE60'
}

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


def plot_tsne_comparison(save_path):
    # Create t-SNE visuals comparing all 4 feature spaces (Raw, PCA, UMAP, Autoencoder)
    # PAY ATTENCTION TO UMAP, that i think is most important bc we can actually see some possible issues w tbe weird clumps/patches 
    
    # Load all data
    raw_data = np.load('../data/cifar100_raw.npz')
    pca_data = np.load(f'../data/cifar100_pca{REDUCED_DIM}.npz')
    umap_data = np.load(f'../data/cifar100_umap{REDUCED_DIM}.npz')
    ae_data = np.load(f'../data/cifar100_ae{REDUCED_DIM}.npz')
    
    test_labels = raw_data['test_labels']
    
    n_samples = min(TSNE_SAMPLE_SIZE, len(test_labels))
    np.random.seed(RANDOM_SEED)
    indices = np.random.choice(len(test_labels), n_samples, replace=False)
    
    # Use all classes
    labels_subset = test_labels[indices]
    selected_classes = np.unique(labels_subset)
    
    # Prepare data dictionary
    data_dict = {
        'Raw (3072-D)': raw_data['test_features'][indices],
        f'PCA ({REDUCED_DIM}-D)': pca_data['test_features'][indices],
        f'UMAP ({REDUCED_DIM}-D)': umap_data['test_features'][indices],
        f'AE ({REDUCED_DIM}-D)': ae_data['test_features'][indices]
    }
    
    # Create 2x2 subplot with clean style
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    axes = axes.flatten()
    
    # Use a colormap with enough colors for 100 classes
    cmap = plt.cm.get_cmap('tab20')
    
    for idx, (name, features) in enumerate(data_dict.items()):
        print(f"  Computing t-SNE for {name}...")
        tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, 
                   n_iter=TSNE_N_ITER, random_state=RANDOM_SEED)
        embedded = tsne.fit_transform(features)
        
        # Plot all classes
        for i, cls in enumerate(selected_classes):
            cls_mask = labels_subset == cls
            color_idx = i % 20  # Cycle through 20 colors
            axes[idx].scatter(embedded[cls_mask, 0], embedded[cls_mask, 1],
                            c=[cmap(color_idx)], alpha=0.6, s=10, edgecolors='none')
        
        axes[idx].set_title(name, fontsize=13, fontweight='bold', pad=10)
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
        axes[idx].set_xlabel('t-SNE dim 1', fontsize=10, color='#666666')
        axes[idx].set_ylabel('t-SNE dim 2', fontsize=10, color='#666666')
        
        # Clean up spines
        for spine in axes[idx].spines.values():
            spine.set_visible(False)
    
    plt.suptitle(f't-SNE Visualization: Comparing Dimensionality Reduction Methods\n(All {len(selected_classes)} classes, {n_samples} samples)',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  t-SNE comparison saved to '{save_path}'")


def create_comparison_plots(all_results, save_path):

    models = ['ResNet', 'Transformer', 'Autoencoder']
    input_types = ['Raw', f'PCA-{REDUCED_DIM}', f'UMAP-{REDUCED_DIM}', f'AE-{REDUCED_DIM}']
    input_labels = ['Raw', 'PCA', 'UMAP', 'AE']
    
    colors = [COLORS['Raw'], COLORS['PCA'], COLORS['UMAP'], COLORS['AE']]
    
    fig = plt.figure(figsize=(18, 10), facecolor='white')
    
    # Plot 1: Accuracy by Model
    ax1 = fig.add_subplot(2, 3, 1)
    x = np.arange(len(models))
    width = 0.18
    
    for i, (input_type, label) in enumerate(zip(input_types, input_labels)):
        accuracies = []
        for model in models:
            row = all_results[(all_results['Model'] == model) & (all_results['Input_Type'] == input_type)]
            acc = row['Accuracy_%'].values[0] if len(row) > 0 else 0
            accuracies.append(acc)
        
        bars = ax1.bar(x + i*width - 1.5*width, accuracies, width, label=label,
                       color=colors[i], edgecolor='white', linewidth=0.5)
    
    ax1.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax1.set_title('Classification Accuracy', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=11)
    ax1.legend(loc='upper right', fontsize=9, frameon=False)
    ax1.set_ylim(0, ax1.get_ylim()[1] * 1.1)
    ax1.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)
    
    # Plot 2: Training Time by Model
    ax2 = fig.add_subplot(2, 3, 2)
    
    for i, (input_type, label) in enumerate(zip(input_types, input_labels)):
        times = []
        for model in models:
            row = all_results[(all_results['Model'] == model) & (all_results['Input_Type'] == input_type)]
            t = row['Training_Time_sec'].values[0] if len(row) > 0 else 0
            times.append(t)
        
        ax2.bar(x + i*width - 1.5*width, times, width, label=label,
                color=colors[i], edgecolor='white', linewidth=0.5)
    
    ax2.set_ylabel('Training Time (seconds)', fontsize=11)
    ax2.set_title('Training Time', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=11)
    ax2.legend(loc='upper right', fontsize=9, frameon=False)
    ax2.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.set_axisbelow(True)
    
    # Plots 3-5: Impact Summary for Each Classifier
    for plot_idx, model in enumerate(models):
        ax = fig.add_subplot(2, 3, plot_idx + 4)  # positions 4, 5, 6 (second row)
        
        # Get Raw baseline for this specific model
        raw_row = all_results[(all_results['Model'] == model) & 
                             (all_results['Input_Type'] == 'Raw')]
        raw_time = raw_row['Training_Time_sec'].values[0]
        raw_acc = raw_row['Accuracy_%'].values[0]
        
        # Calculate impact for PCA, UMAP, AE (skip Raw)
        method_labels = []
        time_reductions = []
        accuracy_changes = []
        
        for input_type, label in zip(input_types[1:], input_labels[1:]):
            row = all_results[(all_results['Model'] == model) & 
                            (all_results['Input_Type'] == input_type)]
            
            if len(row) > 0:
                time = row['Training_Time_sec'].values[0]
                acc = row['Accuracy_%'].values[0]
                
                # Calculate percentage time reduction
                time_reduction = ((raw_time - time) / raw_time) * 100
                # Calculate accuracy change
                accuracy_change = acc - raw_acc
                
                method_labels.append(label)
                time_reductions.append(time_reduction)
                accuracy_changes.append(accuracy_change)
        
        # Create horizontal bar chart
        y_pos = np.arange(len(method_labels))
        height = 0.35
        
        # Plot accuracy change bars (red)
        bars1 = ax.barh(y_pos - height/2, accuracy_changes, height, 
                       label='Accuracy Change (%)', color='#E74C3C', alpha=0.8,
                       edgecolor='white', linewidth=0.5)
        
        # Plot time reduction bars (blue)
        bars2 = ax.barh(y_pos + height/2, time_reductions, height,
                       label='Time Reduction (%)', color='#3498DB', alpha=0.8,
                       edgecolor='white', linewidth=0.5)
        
        # Add value labels on bars
        for i, (y, acc, time) in enumerate(zip(y_pos, accuracy_changes, time_reductions)):
            # Accuracy change label
            if acc < -1:
                ax.text(acc - 1, y - height/2, f'{acc:.1f}%', 
                       ha='right', va='center', fontsize=9, fontweight='bold', color='white')
            else:
                ax.text(acc + 1, y - height/2, f'{acc:.1f}%', 
                       ha='left', va='center', fontsize=9, fontweight='bold')
            
            # Time reduction label
            ax.text(time + 1, y + height/2, f'{time:.0f}%',
                   ha='left', va='center', fontsize=9, fontweight='bold')
        
        # Add vertical line at x=0
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(method_labels, fontsize=10)
        ax.set_xlabel('Percentage (%)', fontsize=10)
        
        ax.set_title(f'{model} Impact', fontsize=12, fontweight='bold', pad=10)
        
        if plot_idx == 0:  # Only show legend on first plot
            ax.legend(loc='upper right', fontsize=8, frameon=False)
        
        ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def create_summary_table(all_results, save_path):    
    models = ['ResNet', 'Transformer', 'Autoencoder']
    input_types = ['Raw', f'PCA-{REDUCED_DIM}', f'UMAP-{REDUCED_DIM}', f'AE-{REDUCED_DIM}']
    
    # Build table data
    table_data = []
    for model in models:
        row = [model]
        for input_type in input_types:
            data = all_results[(all_results['Model'] == model) & (all_results['Input_Type'] == input_type)]
            if len(data) > 0:
                acc = data['Accuracy_%'].values[0]
                row.append(f'{acc:.2f}%')
            else:
                row.append('-')
        table_data.append(row)
    
    # Add average row
    avg_row = ['Average']
    for input_type in input_types:
        data = all_results[all_results['Input_Type'] == input_type]
        if len(data) > 0:
            avg_row.append(f'{data["Accuracy_%"].mean():.2f}%')
        else:
            avg_row.append('-')
    table_data.append(avg_row)
    
    columns = ['Model', 'Raw', 'PCA', 'UMAP', 'AE']
    
    # Create figure with clean styling
    fig, ax = plt.subplots(figsize=(10, 3), facecolor='white')
    ax.axis('off')
    
    # Header colors matching our scheme
    header_colors = ['#2D3E50', COLORS['Raw'], COLORS['PCA'], COLORS['UMAP'], COLORS['AE']]
    
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        colColours=header_colors
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.4, 2.0)
    
    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style cells
    for i in range(1, len(table_data) + 1):
        for j in range(len(columns)):
            table[(i, j)].set_edgecolor('#DDDDDD')
            if i == len(table_data):  # Average row
                table[(i, j)].set_facecolor('#F5F5F5')
                table[(i, j)].set_text_props(weight='bold')
            elif i % 2 == 0:
                table[(i, j)].set_facecolor('#FAFAFA')
            else:
                table[(i, j)].set_facecolor('white')
    
    plt.title('Classification Accuracy Summary (%)',
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    print("Generating Results")
    
    os.makedirs('../results', exist_ok=True)
    
    # Load all results
    resnet_df = pd.read_csv('../results/resnet_results.csv')
    transformer_df = pd.read_csv('../results/transformer_results.csv')
    autoencoder_df = pd.read_csv('../results/autoencoder_results.csv')
    
    # Combine all results
    all_results = pd.concat([resnet_df, transformer_df, autoencoder_df], ignore_index=True)
    all_results.to_csv('../results/all_results.csv', index=False)
    
    # 1. Create comparison plots
    create_comparison_plots(all_results, '../results/comparison_plots.png')
    
    # 2. Create summary table
    print("\n[2/4] Creating summary table...")
    create_summary_table(all_results, '../results/summary_table.png')
    
    # 3. t-SNE Visualization
    # if input("Do you want to regenerate TSNE chart (takes a min): Yes or No").upper() == "YES":
    #     plot_tsne_comparison('../results/tsne_comparison.png')
    plot_tsne_comparison('../results/tsne_comparison.png')
    

if __name__ == "__main__":
    main()