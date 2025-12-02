"""
Master pipeline to run all experiments
"""
print("="*60)
print("CNN Feature Dimensionality Reduction Pipeline")
print("="*60 + "\n")

scripts = [
    ('0_baseline_resnet.py', 'Step A: ResNet Baseline'),
    ('1_save_raw_images.py', 'Save Raw Images (3072-D)'),
    ('2_pca_only.py', 'Step B: PCA Reduction (3072→200)'),
    ('3_autoencoder_sweep.py', 'Step C: PCA→Autoencoder'),
    ('4_transformer_sweep.py', 'Step D: PCA→Transformer'),
    ('5_compare_results.py', 'Generate Comparison Plots')
]

for i, (script, desc) in enumerate(scripts, 1):
    print(f"\n[{i}/{len(scripts)}] {desc}...")
    print("-"*60)
    with open(script) as f:
        exec(f.read())

print("\n" + "="*60)
print("✅ Pipeline Complete! Check ../results/ folder.")
print("="*60 + "\n")
