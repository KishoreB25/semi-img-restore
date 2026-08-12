import numpy as np
import os
import matplotlib.pyplot as plt

def inspect_dataset(base_dir):
    train_gt_dir = os.path.join(base_dir, "train", "train", "GT")
    train_noisy_dir = os.path.join(base_dir, "train", "train", "NoisyLR")
    test_noisy_dir = os.path.join(base_dir, "Test_NoisyLR", "NoisyLR")
    
    gt_files = sorted(os.listdir(train_gt_dir))
    noisy_files = sorted(os.listdir(train_noisy_dir))
    test_files = sorted(os.listdir(test_noisy_dir))
    
    print(f"Number of Train GT files: {len(gt_files)}")
    print(f"Number of Train NoisyLR files: {len(noisy_files)}")
    print(f"Number of Test NoisyLR files: {len(test_files)}")
    
    # Inspect a few GT files
    print("\n--- GT Statistics ---")
    gt_mins, gt_maxs, gt_means, gt_stds = [], [], [], []
    for f in gt_files[:10]:
        data = np.load(os.path.join(train_gt_dir, f))
        gt_mins.append(data.min())
        gt_maxs.append(data.max())
        gt_means.append(data.mean())
        gt_stds.append(data.std())
    print(f"Shape: {np.load(os.path.join(train_gt_dir, gt_files[0])).shape}")
    print(f"Dtype: {np.load(os.path.join(train_gt_dir, gt_files[0])).dtype}")
    print(f"Min range: {min(gt_mins):.4f} to {max(gt_mins):.4f}")
    print(f"Max range: {min(gt_maxs):.4f} to {max(gt_maxs):.4f}")
    print(f"Mean range: {min(gt_means):.4f} to {max(gt_means):.4f}")
    print(f"Std range: {min(gt_stds):.4f} to {max(gt_stds):.4f}")

    # Inspect a few NoisyLR files
    print("\n--- Train NoisyLR Statistics ---")
    n_mins, n_maxs, n_means, n_stds = [], [], [], []
    for f in noisy_files[:10]:
        data = np.load(os.path.join(train_noisy_dir, f))
        n_mins.append(data.min())
        n_maxs.append(data.max())
        n_means.append(data.mean())
        n_stds.append(data.std())
    print(f"Shape: {np.load(os.path.join(train_noisy_dir, noisy_files[0])).shape}")
    print(f"Dtype: {np.load(os.path.join(train_noisy_dir, noisy_files[0])).dtype}")
    print(f"Min range: {min(n_mins):.4f} to {max(n_mins):.4f}")
    print(f"Max range: {min(n_maxs):.4f} to {max(n_maxs):.4f}")
    print(f"Mean range: {min(n_means):.4f} to {max(n_means):.4f}")
    print(f"Std range: {min(n_stds):.4f} to {max(n_stds):.4f}")

    # Inspect Test NoisyLR files
    print("\n--- Test NoisyLR Statistics ---")
    t_mins, t_maxs, t_means, t_stds = [], [], [], []
    for f in test_files[:10]:
        data = np.load(os.path.join(test_noisy_dir, f))
        t_mins.append(data.min())
        t_maxs.append(data.max())
        t_means.append(data.mean())
        t_stds.append(data.std())
    print(f"Shape: {np.load(os.path.join(test_noisy_dir, test_files[0])).shape}")
    print(f"Dtype: {np.load(os.path.join(test_noisy_dir, test_files[0])).dtype}")
    print(f"Min range: {min(t_mins):.4f} to {max(t_mins):.4f}")
    print(f"Max range: {min(t_maxs):.4f} to {max(t_maxs):.4f}")
    print(f"Mean range: {min(t_means):.4f} to {max(t_means):.4f}")
    print(f"Std range: {min(t_stds):.4f} to {max(t_stds):.4f}")

    # Visualizations
    vis_dir = os.path.join(base_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    print("\nGenerating visualizations...")
    for i in range(3):
        gt_data = np.load(os.path.join(train_gt_dir, gt_files[i]))
        noisy_data = np.load(os.path.join(train_noisy_dir, noisy_files[i]))
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # GT Image
        axes[0, 0].imshow(gt_data, cmap='gray', vmin=0, vmax=1)
        axes[0, 0].set_title(f'GT {gt_files[i]} (256x256)')
        axes[0, 0].axis('off')
        
        # NoisyLR Image
        # Note: NoisyLR has values outside [0,1], so we let matplotlib scale or set explicitly
        axes[0, 1].imshow(noisy_data, cmap='gray')
        axes[0, 1].set_title(f'NoisyLR {noisy_files[i]} (128x128)')
        axes[0, 1].axis('off')
        
        # GT Histogram
        axes[1, 0].hist(gt_data.ravel(), bins=50, color='blue', alpha=0.7)
        axes[1, 0].set_title('GT Histogram')
        axes[1, 0].set_xlim([0, 1])
        
        # NoisyLR Histogram
        axes[1, 1].hist(noisy_data.ravel(), bins=50, color='orange', alpha=0.7)
        axes[1, 1].set_title('NoisyLR Histogram')
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'pair_vis_{i}.png'))
        plt.close()
    
    print(f"Visualizations saved to {vis_dir}")

if __name__ == "__main__":
    inspect_dataset(r"d:\semi-img-restore")
