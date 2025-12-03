"""
Main Pipeline: CIFAR-100 Classification with Dimensionality Reduction Comparison
Compares: Raw vs PCA vs UMAP vs Autoencoder reduction
Classifiers: ResNet, Transformer, Autoencoder
"""

scripts = [
    ('0_prepare_data.py', 'Prepare Data (Raw + PCA + UMAP + Autoencoder)'),
    ('1_resnet_classifier.py', 'Train ResNet Classifiers (4 input types)'),
    ('2_transformer_classifier.py', 'Train Transformer Classifiers (4 input types)'),
    ('3_autoencoder_classifier.py', 'Train Autoencoder Classifiers (4 input types)'),
    ('4_compare_results.py', 'Generate Comparison Analysis')
]

print("="*60)
print("CIFAR-100 Classification Experiment")
print("Comparing: Raw vs PCA vs UMAP vs Autoencoder")
print("="*60)

for i, (script, desc) in enumerate(scripts, 1):
    print(f"\n[{i}/{len(scripts)}] {desc}")
    print("-"*60)
    with open(script) as f:
        exec(f.read())

print("\n" + "="*60)
print("EXPERIMENT COMPLETE!")
print("Check ../results/ for outputs")
print("="*60) 