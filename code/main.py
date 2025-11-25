print("MAKE SURE YOU'RE IN THE code DIRECTORY")
print("\n" + "="*60)
print("Running CNN Feature Dimensionality Reduction Pipeline")
print("="*60 + "\n")

scripts = [
    ('0_feature_extractor.py', 'Extracting ResNet50 features'),
    ('00_pca_sweep.py', 'PCA baseline'),
    ('1_autoencoder_sweep.py', 'Autoencoder sweep'),
    ('2_transformer_sweep.py', 'Transformer sweep'),
    ('3_compare_results.py', 'Comparison plots')
]

for i, (script, desc) in enumerate(scripts, 1):
    print(f"[{i}/5] {desc}...")
    with open(script) as f:
        exec(f.read())
    print(f"✅ {script} completed\n")

print("="*60)
print("✅ Pipeline complete! Check results/ folder.")
print("="*60 + "\n")
