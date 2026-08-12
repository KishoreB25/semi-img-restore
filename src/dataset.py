import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class RestorationDataset(Dataset):
    def __init__(self, gt_dir=None, noisy_dir=None, filenames=None, transform=None):
        """
        Args:
            gt_dir: Directory containing GT .npy files (can be None for testing).
            noisy_dir: Directory containing NoisyLR .npy files.
            filenames: List of filenames to include in this dataset.
            transform: Optional transform to be applied on a sample.
        """
        self.gt_dir = gt_dir
        self.noisy_dir = noisy_dir
        self.filenames = filenames
        self.transform = transform
        
        if self.filenames is None:
            self.filenames = sorted(os.listdir(noisy_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        noisy_path = os.path.join(self.noisy_dir, fname)
        noisy = np.load(noisy_path).astype(np.float32)
        
        # Add channel dimension if needed -> 1xHxW
        if noisy.ndim == 2:
            noisy = noisy[None, :, :]
            
        noisy_tensor = torch.from_numpy(noisy)
        
        if self.gt_dir is not None:
            gt_path = os.path.join(self.gt_dir, fname)
            gt = np.load(gt_path).astype(np.float32)
            if gt.ndim == 2:
                gt = gt[None, :, :]
            gt_tensor = torch.from_numpy(gt)
            
            if self.transform:
                noisy_tensor, gt_tensor = self.transform(noisy_tensor, gt_tensor)
                
            return noisy_tensor, gt_tensor, fname
        
        return noisy_tensor, fname

def get_dataloaders(base_dir, val_split=0.1, batch_size=16, seed=42):
    train_gt_dir = os.path.join(base_dir, "train", "train", "GT")
    train_noisy_dir = os.path.join(base_dir, "train", "train", "NoisyLR")
    
    all_files = sorted(os.listdir(train_gt_dir))
    
    # Deterministic split
    np.random.seed(seed)
    indices = np.random.permutation(len(all_files))
    val_size = int(len(all_files) * val_split)
    
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    train_files = [all_files[i] for i in train_idx]
    val_files = [all_files[i] for i in val_idx]
    
    train_dataset = RestorationDataset(gt_dir=train_gt_dir, noisy_dir=train_noisy_dir, filenames=train_files)
    val_dataset = RestorationDataset(gt_dir=train_gt_dir, noisy_dir=train_noisy_dir, filenames=val_files)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader

def get_test_dataloader(test_noisy_dir, batch_size=16):
    test_dataset = RestorationDataset(noisy_dir=test_noisy_dir)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
