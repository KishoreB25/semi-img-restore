import os
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import get_dataloaders
from resunet import ResUNet
from metrics import Evaluator

def save_visuals(noisy_np, pred_np, gt_np, fname, save_dir, epoch):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(noisy_np, cmap='gray')
    axes[0].set_title(f"NoisyLR")
    axes[0].axis('off')
    
    axes[1].imshow(pred_np, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f"ResUNet (Epoch {epoch})")
    axes[1].axis('off')
    
    axes[2].imshow(gt_np, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f"Ground Truth")
    axes[2].axis('off')
    
    residual = np.abs(gt_np - pred_np)
    im = axes[3].imshow(residual, cmap='hot', vmin=0, vmax=1)
    axes[3].set_title(f"Absolute Residual")
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{fname}_epoch_{epoch:03d}.png"))
    plt.close()

def main():
    base_dir = r"d:\semi-img-restore"
    results_dir = os.path.join(base_dir, "results", "phase03_neural_baseline")
    visuals_dir = os.path.join(results_dir, "visuals")
    checkpoints_dir = os.path.join(results_dir, "checkpoints")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! Please install the CUDA version of PyTorch.")
    device = torch.device('cuda')
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    
    # 1. Configuration
    batch_size = 16 if torch.cuda.is_available() else 4
    num_epochs = 30
    lr = 2e-4
    
    config = {
        'optimizer': 'AdamW',
        'initial_lr': lr,
        'scheduler': 'CosineAnnealingLR',
        'batch_size': batch_size,
        'epochs': num_epochs,
        'seed': 42,
        'loss': 'L1Loss',
        'model': 'ResUNet'
    }
    
    os.makedirs(os.path.join(base_dir, "configs"), exist_ok=True)
    with open(os.path.join(base_dir, "configs", "baseline_resunet.yaml"), "w") as f:
        yaml.dump(config, f)
        
    # 2. Data
    train_loader, val_loader = get_dataloaders(base_dir, val_split=0.1, batch_size=batch_size, seed=42)
    evaluator = Evaluator(device=device)
    
    # 3. Model & Loss
    model = ResUNet().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    best_psnr = 0.0
    logs = []
    
    for epoch in range(1, num_epochs + 1):
        # TRAIN
        model.train()
        train_loss = 0.0
        
        # Use tqdm only if not piping output, but let's keep it simple
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]")
        for noisy, gt, _ in pbar:
            noisy, gt = noisy.to(device), gt.to(device)
            
            optimizer.zero_grad()
            pred = model(noisy)
            loss = criterion(pred, gt)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * noisy.size(0)
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        train_loss /= len(train_loader.dataset)
        scheduler.step()
        
        # EVALUATE
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_ssim = 0.0
        val_lpips = 0.0
        
        visuals_saved = 0
        
        with torch.no_grad():
            for noisy, gt, fnames in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]"):
                noisy, gt = noisy.to(device), gt.to(device)
                
                pred = model(noisy)
                loss = criterion(pred, gt)
                val_loss += loss.item() * noisy.size(0)
                
                metrics = evaluator.evaluate_batch(pred, gt)
                val_psnr += metrics['psnr'] * noisy.size(0)
                val_ssim += metrics['ssim'] * noisy.size(0)
                val_lpips += metrics['lpips'] * noisy.size(0)
                
                if visuals_saved < 3:
                    save_visuals(
                        noisy[0, 0].cpu().numpy(),
                        pred[0, 0].clamp(0, 1).cpu().numpy(),
                        gt[0, 0].cpu().numpy(),
                        fnames[0].replace('.npy', ''),
                        visuals_dir,
                        epoch
                    )
                    visuals_saved += 1
                    
        val_loss /= len(val_loader.dataset)
        val_psnr /= len(val_loader.dataset)
        val_ssim /= len(val_loader.dataset)
        val_lpips /= len(val_loader.dataset)
        
        print(f"Epoch {epoch} Summary: Train L1: {train_loss:.4f} | Val L1: {val_loss:.4f} | PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.4f} | LPIPS: {val_lpips:.4f}")
        
        logs.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'psnr': val_psnr,
            'ssim': val_ssim,
            'lpips': val_lpips,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, "best.pth"))
            print(f"--> Saved new best model with PSNR {best_psnr:.2f}")
            
    # Save metrics CSV
    df = pd.DataFrame(logs)
    df.to_csv(os.path.join(results_dir, "metrics.csv"), index=False)
    
    # Update global experiments.csv
    csv_path = os.path.join(base_dir, "experiments.csv")
    with open(csv_path, "a") as f:
        # psnr, ssim, lpips from best epoch? Let's take final epoch for simplicity, or best PSNR epoch
        best_epoch_log = df.loc[df['psnr'].idxmax()]
        f.write(f"phase03_resunet,ResUNet,L1,None,None,{best_epoch_log['psnr']:.4f},{best_epoch_log['ssim']:.4f},{best_epoch_log['lpips']:.4f},TODO,First Neural Baseline\n")

if __name__ == "__main__":
    main()
