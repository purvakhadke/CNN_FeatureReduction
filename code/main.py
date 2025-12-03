"""
Main Pipeline: CIFAR-100 Classification with/without PCA
Compares ResNet, Transformer, and Autoencoder classifiers
"""

scripts = [
    ('0_prepare_data.py', 'Prepare Data (Raw + PCA)'),
    ('1_resnet_classifier.py', 'Train ResNet Classifiers'),
    ('2_transformer_classifier.py', 'Train Transformer Classifiers'),
    ('3_autoencoder_classifier.py', 'Train Autoencoder Classifiers'),
    ('4_compare_results.py', 'Generate Comparison Analysis')
]

for i, (script, desc) in enumerate(scripts, 1):
    with open(script) as f:
        print(f"\nProcessing {script}")
        exec(f.read())
    