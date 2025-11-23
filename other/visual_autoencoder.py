import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {DEVICE}")

BATCH_SIZE = 64
EPOCHS = 20       # Higher epochs needed to learn to draw images
LEARNING_RATE = 1e-3
SAMPLE_LIMIT = 2000  # We only use 2000 images
LATENT_DIMS = [1024, 256, 64, 32] # The sweep sizes

# --- 1. Data Loading (Raw Images) ---
# We do NOT resize to 224 here. We keep them at 32x32 to make training fast and clear.
transform = transforms.Compose([
    transforms.ToTensor(),
])

full_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Select first 2000 images
indices = list(range(SAMPLE_LIMIT))
subset_data = Subset(full_dataset, indices)
loader = DataLoader(subset_data, batch_size=BATCH_SIZE, shuffle=True)

print(f"Loaded {len(subset_data)} images.")

# --- 2. The Image Autoencoder ---
class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(ConvolutionalAutoencoder, self).__init__()
        
        # ENCODER: Compresses Image (3 RGB channels) -> Latent Vector
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 16x16 -> 8x8
            nn.ReLU(),
            nn.Flatten(),
            # 32 filters * 8 * 8 pixels = 2048 size flat vector
            nn.Linear(32 * 8 * 8, latent_dim)          # Compress to Latent Dim
        )
        
        # DECODER: Latent Vector -> Reconstructs Image
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (32, 8, 8)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),  # 16x16 -> 32x32
            nn.Sigmoid() # Squish output between 0 and 1 (pixel colors)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# --- 3. Helper to Save Images ---
def save_comparison_images(model, dataset, latent_dim):
    # """Takes 5 random images, passes them through the model, and saves the Before/After."""
    model.eval()
    
    # Get a small batch of images
    data_iter = iter(DataLoader(dataset, batch_size=5, shuffle=False))
    images, _ = next(data_iter)
    images = images.to(DEVICE)
    
    with torch.no_grad():
        reconstructed = model(images)
    
    # Move to CPU for plotting
    images = images.cpu()
    reconstructed = reconstructed.cpu()
    
    # Plotting
    plt.figure(figsize=(10, 4))
    for i in range(5):
        # Original (Top Row)
        ax = plt.subplot(2, 5, i + 1)
        # plt.imshow(np.transpose(images[i], (1, 2, 0)))
        plt.imshow(np.transpose(images[i], (1, 2, 0)), interpolation='nearest') 
        ax.set_title("Original")
        plt.axis("off")
        
        # Reconstructed (Bottom Row)
        ax = plt.subplot(2, 5, i + 1 + 5)
        # plt.imshow(np.transpose(reconstructed[i], (1, 2, 0)))
        plt.imshow(np.transpose(reconstructed[i], (1, 2, 0)), interpolation='nearest')
        ax.set_title(f"Dim {latent_dim}")
        plt.axis("off")
        
    plt.tight_layout()
    filename = f"../results/visual_result_{latent_dim}.png"
    plt.savefig(filename)
    print(f"Saved image visualization to {filename}")
    plt.close()

# --- 4. Main Loop ---
def main():
    for dim in LATENT_DIMS:
        print(f"\n--- Training Autoencoder for Latent Dimension: {dim} ---")
        model = ConvolutionalAutoencoder(latent_dim=dim).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Train
        model.train()
        for epoch in range(EPOCHS):
            total_loss = 0
            for img, _ in loader:
                img = img.to(DEVICE)
                optimizer.zero_grad()
                output = model(img)
                loss = criterion(output, img)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch+1) % 5 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(loader):.4f}")
                
        # Visualize Results
        save_comparison_images(model, subset_data, dim)

if __name__ == "__main__":
    main()