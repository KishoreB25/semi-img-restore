import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

def main():
    base_dir = r"d:\semi-img-restore"
    
    train_gt_dir = os.path.join(base_dir, "train", "train", "GT")
    train_noisy_dir = os.path.join(base_dir, "train", "train", "NoisyLR")
    test_noisy_dir = os.path.join(base_dir, "Test_NoisyLR", "NoisyLR")
    
    report_dir = os.path.join(base_dir, "reports", "phase00_dataset_audit")
    os.makedirs(report_dir, exist_ok=True)
    
    # 1. Discover files
    train_gt_files = sorted(os.listdir(train_gt_dir))
    train_noisy_files = sorted(os.listdir(train_noisy_dir))
    test_noisy_files = sorted(os.listdir(test_noisy_dir))
    
    # 2. Verify pairing
    assert len(train_gt_files) == len(train_noisy_files), f"Mismatch in train counts: GT {len(train_gt_files)}, Noisy {len(train_noisy_files)}"
    assert train_gt_files == train_noisy_files, "Train file names do not match between GT and NoisyLR."
    
    # Check counts
    assert len(train_gt_files) == 3200, f"Expected 3200 GT files, got {len(train_gt_files)}"
    assert len(test_noisy_files) == 400, f"Expected 400 test files, got {len(test_noisy_files)}"
    
    stats_records = []
    
    # 3 & 4. Validate arrays and compute statistics
    def process_file(filepath, role):
        data = np.load(filepath)
        
        # validate array
        assert data.ndim == 2, f"{role} {filepath} ndim is {data.ndim} not 2"
        assert data.dtype == np.float32, f"{role} {filepath} dtype is {data.dtype} not float32"
        assert not np.isnan(data).any(), f"{role} {filepath} contains NaN"
        assert not np.isinf(data).any(), f"{role} {filepath} contains Inf"
        
        if role == 'GT':
            assert data.shape == (256, 256), f"GT {filepath} shape is {data.shape}"
            assert data.min() >= 0.0 and data.max() <= 1.0, f"GT {filepath} out of [0,1] range"
        else:
            assert data.shape == (128, 128), f"NoisyLR {filepath} shape is {data.shape}"
            
        record = {
            'filename': os.path.basename(filepath),
            'role': role,
            'shape': str(data.shape),
            'dtype': str(data.dtype),
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
        return record, data
    
    print("Processing Train pairs...")
    train_gt_arrays = []
    train_noisy_arrays = []
    for f in train_gt_files:
        gt_path = os.path.join(train_gt_dir, f)
        noisy_path = os.path.join(train_noisy_dir, f)
        
        gt_rec, gt_data = process_file(gt_path, 'GT')
        noisy_rec, noisy_data = process_file(noisy_path, 'Train_NoisyLR')
        
        stats_records.append(gt_rec)
        stats_records.append(noisy_rec)
        
        train_gt_arrays.append(gt_data)
        train_noisy_arrays.append(noisy_data)
        
    print("Processing Test files...")
    for f in test_noisy_files:
        path = os.path.join(test_noisy_dir, f)
        rec, _ = process_file(path, 'Test_NoisyLR')
        stats_records.append(rec)
        
    # Save statistics
    df = pd.DataFrame(stats_records)
    df.to_csv(os.path.join(report_dir, "statistics.csv"), index=False)
    
    # Compute Global Statistics
    global_stats = {
        'GT': df[df['role'] == 'GT'][['min', 'max', 'mean', 'std', 'p01', 'p99']].mean().to_dict(),
        'Train_NoisyLR': df[df['role'] == 'Train_NoisyLR'][['min', 'max', 'mean', 'std', 'p01', 'p99']].mean().to_dict(),
        'Test_NoisyLR': df[df['role'] == 'Test_NoisyLR'][['min', 'max', 'mean', 'std', 'p01', 'p99']].mean().to_dict()
    }
    
    # 5. Visual inspection
    print("Generating visualizations...")
    np.random.seed(42)
    sample_indices = np.random.choice(len(train_gt_files), 20, replace=False)
    
    for i, idx in enumerate(sample_indices):
        gt_data = train_gt_arrays[idx]
        noisy_data = train_noisy_arrays[idx]
        filename = train_gt_files[idx]
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(gt_data, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f"GT: {filename}")
        axes[0].axis('off')
        
        axes[1].imshow(noisy_data, cmap='gray')
        axes[1].set_title(f"NoisyLR: {filename}")
        axes[1].axis('off')
        
        axes[2].hist(gt_data.ravel(), bins=50, color='blue', alpha=0.7)
        axes[2].set_title("GT Histogram")
        axes[2].set_xlim([0, 1])
        
        axes[3].hist(noisy_data.ravel(), bins=50, color='orange', alpha=0.7)
        axes[3].set_title("NoisyLR Histogram")
        
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, f"sample_pairs_{i:02d}.png"))
        plt.close()

    # 6. Save audit manifest
    manifest = {
        "timestamp": datetime.datetime.now().isoformat(),
        "files_found": {
            "train_gt": len(train_gt_files),
            "train_noisy": len(train_noisy_files),
            "test_noisy": len(test_noisy_files)
        },
        "pairing_valid": True,
        "shapes_valid": True,
        "dtypes_valid": True,
        "global_statistics": global_stats,
        "visualizations_saved": 20
    }
    
    with open(os.path.join(report_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=4)
        
    print("Dataset audit complete. All checks passed.")

if __name__ == "__main__":
    main()
