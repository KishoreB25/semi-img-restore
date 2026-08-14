import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import yaml

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from degradation import SyntheticDegradationEngine, add_gaussian_noise, add_speckle_noise, bicubic_downsample

def main():
    base_dir = r"d:\semi-img-restore"
    results_dir = os.path.join(base_dir, "results", "phase05_synthetic")
    samples_dir = os.path.join(results_dir, "generated_samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    # Load configuration
    config_path = os.path.join(base_dir, "configs", "synthetic_degradation.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    deg_cfg = config['degradation']
    engine = SyntheticDegradationEngine(
        min_g=deg_cfg['gaussian']['min_sigma'],
        max_g=deg_cfg['gaussian']['max_sigma'],
        min_s=deg_cfg['speckle']['min_sigma'],
        max_s=deg_cfg['speckle']['max_sigma']
    )
    
    # Let's perform unit-like checks
    print("Running operator-level tests...")
    test_img = torch.ones((1, 256, 256))
    
    # 1. Downsample output shape
    ds_img = bicubic_downsample(test_img)
    assert ds_img.shape == (1, 128, 128), f"Expected shape (1, 128, 128), got {ds_img.shape}"
    print("  - Downsample shape check passed.")
    
    # 2. Gaussian-only output changes statistics
    g_img = add_gaussian_noise(test_img, 0.05)
    assert g_img.shape == (1, 256, 256)
    assert abs(g_img.mean() - 1.0) < 0.02
    assert abs(g_img.std() - 0.05) < 0.01
    print("  - Gaussian-only noise check passed.")
    
    # 3. Speckle-only intensity dependence
    zero_img = torch.zeros((1, 256, 256))
    s_zero = add_speckle_noise(zero_img, 0.15)
    assert torch.all(s_zero == 0.0), "Speckle noise on zero image must be zero"
    
    one_img = torch.ones((1, 256, 256))
    s_one = add_speckle_noise(one_img, 0.15)
    assert abs(s_one.std() - 0.15) < 0.03
    print("  - Speckle intensity dependence check passed.")
    
    # 4. Mixed operators preserve expected final dimensions
    deg_img, _ = engine.degrade(test_img)
    assert deg_img.shape == (1, 128, 128)
    print("  - Mixed operators final dimension check passed.")
    
    # 5. Random order reproducibility under fixed seed
    img_a, meta_a = engine.degrade(test_img, seed=42)
    img_b, meta_b = engine.degrade(test_img, seed=42)
    assert torch.allclose(img_a, img_b), "Outputs with same seed must be equal"
    assert meta_a['order'] == meta_b['order'], "Orders with same seed must be equal"
    print("  - Seed reproducibility check passed.")
    
    print("\nStarting distribution comparisons on train subset...")
    train_gt_dir = os.path.join(base_dir, "train", "train", "GT")
    train_noisy_dir = os.path.join(base_dir, "train", "train", "NoisyLR")
    
    fnames = sorted(os.listdir(train_gt_dir))[:100]
    
    real_stats = []
    synth_stats = []
    
    for idx, fname in enumerate(fnames):
        gt = np.load(os.path.join(train_gt_dir, fname)).astype(np.float32)
        real_noisy = np.load(os.path.join(train_noisy_dir, fname)).astype(np.float32)
        
        gt_t = torch.from_numpy(gt)[None, :, :]  # 1x256x256
        
        # Degrade using engine
        synth_noisy_t, meta = engine.degrade(gt_t, seed=42 + idx)
        synth_noisy = synth_noisy_t.squeeze().numpy()
        
        # Compute statistics helper
        def get_stat_dict(data, name):
            return {
                'filename': fname,
                'min': float(data.min()),
                'max': float(data.max()),
                'mean': float(data.mean()),
                'std': float(data.std()),
                'p01': float(np.percentile(data, 1)),
                'p05': float(np.percentile(data, 5)),
                'p50': float(np.percentile(data, 50)),
                'p95': float(np.percentile(data, 95)),
                'p99': float(np.percentile(data, 99))
            }
            
        real_stats.append(get_stat_dict(real_noisy, 'real'))
        synth_stats.append(get_stat_dict(synth_noisy, 'synthetic'))
        
        # Save a few sample visualizations (first 5)
        if idx < 5:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(gt, cmap='gray', vmin=0, vmax=1)
            axes[0].set_title(f"GT: {fname}")
            axes[0].axis('off')
            
            axes[1].imshow(real_noisy, cmap='gray')
            axes[1].set_title("Real NoisyLR")
            axes[1].axis('off')
            
            axes[2].imshow(synth_noisy, cmap='gray')
            axes[2].set_title(f"Synth NoisyLR\nOrder: {meta['order']}")
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(samples_dir, f"compare_{fname.replace('.npy', '')}.png"))
            plt.close()
            
    df_real = pd.DataFrame(real_stats)
    df_synth = pd.DataFrame(synth_stats)
    
    # Save statistics.csv containing detailed per-image stats for synthetic
    df_synth.to_csv(os.path.join(results_dir, "statistics.csv"), index=False)
    
    # Create comparison.csv of global averages
    summary_data = []
    for col in ['min', 'max', 'mean', 'std', 'p01', 'p05', 'p50', 'p95', 'p99']:
        summary_data.append({
            'Metric': col,
            'Real_Average': df_real[col].mean(),
            'Synthetic_Average': df_synth[col].mean(),
            'Diff': df_synth[col].mean() - df_real[col].mean()
        })
    df_comp = pd.DataFrame(summary_data)
    df_comp.to_csv(os.path.join(results_dir, "comparison.csv"), index=False)
    
    print("\n--- Global Distribution Comparison ---")
    print(df_comp.to_string(index=False))
    print(f"\nAll results saved to {results_dir}")

if __name__ == "__main__":
    main()
