import csv 
SAMPLE_SIZE = 2000  # Change to None for full dataset



def save_results_csv(filename, dims, mse_losses, accuracies, times):
    """Save numerical results to CSV for later comparison."""
    import os
    os.makedirs('results', exist_ok=True)
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Latent_Dim', 'MSE_Loss', 'Accuracy', 'Training_Time_sec'])
        for d, mse, acc, t in zip(dims, mse_losses, accuracies, times):
            writer.writerow([d, mse, acc, t])
    
    print(f"\n--- Results saved to {filename} ---")
