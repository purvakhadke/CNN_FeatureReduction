# config for randome variables that were going to use thru the files

NUM_CLASSES = 100   

# PCA had over 90% high varience at this dimmentions so this is our target dimension for all reduction methods (PCA, UMAP, Autoencoder)
REDUCED_DIM = 512

# hyper parambertes for training classigier
BATCH_SIZE = 128
LEARNING_RATE = 0.001
RANDOM_SEED = 42

# Epochs for transfomer and autoencoder classifiers
EPOCHS = 30 
# resnet needs less epoch bc its pretrainfed
RESNET_EPOCH = 10


# Transformer classifier
TRANSFORMER_DIM = 128
TRANSFORMER_HEADS = 8
TRANSFORMER_LAYERS = 4
TRANSFORMER_DROPOUT = 0.1


# TSNE visualizatioon
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000
TSNE_SAMPLE_SIZE = 10000 #2000 # 1000

# data input for the experiments
# changed so we only have 1 REDUCED_DIM since it was said we can just choose this once bc it gave high PCA variece
INPUT_FILES = {
    'Raw': '../data/cifar100_raw.npz',
    f'PCA-{REDUCED_DIM}': f'../data/cifar100_pca{REDUCED_DIM}.npz',
    f'UMAP-{REDUCED_DIM}': f'../data/cifar100_umap{REDUCED_DIM}.npz',
    f'AE-{REDUCED_DIM}': f'../data/cifar100_ae{REDUCED_DIM}.npz'
}