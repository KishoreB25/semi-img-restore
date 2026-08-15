import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class RestorationDataset(Dataset):
    def __init__(self, gt_dir=None, noisy_dir=None, filenames=None, transform=None,
                 synthetic_ratio=0.0, min_g=0.00, max_g=0.10, min_s=0.10, max_s=0.20):
        """
        Args:
            gt_dir: Directory containing GT .npy files (can be None for testing).
            noisy_dir: Directory containing NoisyLR .npy files.
            filenames: List of filenames to include in this dataset.
            transform: Optional transform to be applied on a sample.
            synthetic_ratio: Ratio of synthetic data to real data (e.g. 1.0 = 1:1).
            min_g, max_g: Gaussian noise parameter range.
            min_s, max_s: Speckle noise parameter range.
        """
        self.gt_dir = gt_dir
        self.noisy_dir = noisy_dir
        self.filenames = filenames
        self.transform = transform
        self.synthetic_ratio = float(synthetic_ratio)
        
        if self.filenames is None:
            self.filenames = sorted(os.listdir(noisy_dir))
            
        if self.synthetic_ratio > 0 and self.gt_dir is not None:
            import sys
            import os
            # Ensure src is in path so we can import degradation
            src_dir = os.path.dirname(os.path.abspath(__file__))
            if src_dir not in sys.path:
                sys.path.append(src_dir)
            from degradation import SyntheticDegradationEngine
            self.degradation_engine = SyntheticDegradationEngine(
                min_g=min_g, max_g=max_g, min_s=min_s, max_s=max_s
            )
        else:
            self.degradation_engine = None

    def __len__(self):
        if self.synthetic_ratio > 0 and self.gt_dir is not None:
            return int(len(self.filenames) * (1.0 + self.synthetic_ratio))
        return len(self.filenames)

    def __getitem__(self, idx):
        num_real = len(self.filenames)
        is_synthetic = idx >= num_real
        
        fname = self.filenames[idx % num_real]
        
        if self.gt_dir is not None:
            gt_path = os.path.join(self.gt_dir, fname)
            gt = np.load(gt_path).astype(np.float32)
            if gt.ndim == 2:
                gt = gt[None, :, :]
            gt_tensor = torch.from_numpy(gt)
            
            if is_synthetic:
                # Generate synthetic noisy LR from gt_tensor
                # We can use None seed for random training augmentation
                noisy_tensor, _ = self.degradation_engine.degrade(gt_tensor)
            else:
                noisy_path = os.path.join(self.noisy_dir, fname)
                noisy = np.load(noisy_path).astype(np.float32)
                if noisy.ndim == 2:
                    noisy = noisy[None, :, :]
                noisy_tensor = torch.from_numpy(noisy)
                
            if self.transform:
                noisy_tensor, gt_tensor = self.transform(noisy_tensor, gt_tensor)
                
            return noisy_tensor, gt_tensor, fname
            
        # No GT dir: test mode, only real
        noisy_path = os.path.join(self.noisy_dir, fname)
        noisy = np.load(noisy_path).astype(np.float32)
        if noisy.ndim == 2:
            noisy = noisy[None, :, :]
        noisy_tensor = torch.from_numpy(noisy)
        return noisy_tensor, fname

def get_dataloaders(base_dir, val_split=0.1, batch_size=16, seed=42, synthetic_ratio=0.0, degradation_cfg=None):
    train_gt_dir = os.path.join(base_dir, "train", "train", "GT")
    train_noisy_dir = os.path.join(base_dir, "train", "train", "NoisyLR")
    
    config_dir = os.path.join(base_dir, "configs")
    split_file = os.path.join(config_dir, "split.yaml")
    
    all_files = sorted(os.listdir(train_gt_dir))
    
    if os.path.exists(split_file):
        import yaml
        with open(split_file, "r") as f:
            split_data = yaml.safe_load(f)
        train_files = split_data['train']
        val_files = split_data['val']
    else:
        # Deterministic split
        np.random.seed(seed)
        indices = np.random.permutation(len(all_files))
        val_size = int(len(all_files) * val_split)
        
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]
        
        train_files = [all_files[i] for i in train_idx]
        val_files = [all_files[i] for i in val_idx]
        
        import yaml
        os.makedirs(config_dir, exist_ok=True)
        with open(split_file, "w") as f:
            yaml.dump({'train': train_files, 'val': val_files}, f)
            
    # Set default degradation params
    min_g, max_g = 0.00, 0.10
    min_s, max_s = 0.10, 0.20
    if degradation_cfg is not None:
        min_g = degradation_cfg.get('gaussian', {}).get('min_sigma', min_g)
        max_g = degradation_cfg.get('gaussian', {}).get('max_sigma', max_g)
        min_s = degradation_cfg.get('speckle', {}).get('min_sigma', min_s)
        max_s = degradation_cfg.get('speckle', {}).get('max_sigma', max_s)
        
    train_dataset = RestorationDataset(
        gt_dir=train_gt_dir, noisy_dir=train_noisy_dir, filenames=train_files,
        synthetic_ratio=synthetic_ratio, min_g=min_g, max_g=max_g, min_s=min_s, max_s=max_s
    )
    # Validation dataset remains REAL only
    val_dataset = RestorationDataset(
        gt_dir=train_gt_dir, noisy_dir=train_noisy_dir, filenames=val_files,
        synthetic_ratio=0.0
    )
    
    # Use 0 num_workers if windows/debugging, or 4 for fast loading
    num_workers = 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader

def get_test_dataloader(test_noisy_dir, batch_size=16):
    test_dataset = RestorationDataset(noisy_dir=test_noisy_dir)
    num_workers = 4 if torch.cuda.is_available() else 0
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
