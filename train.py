import os
import sys
import yaml
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm

# Ensure we can import from src directory
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dataset import get_dataloaders
from src.resunet import ResUNet
from src.resunet_advanced import AdvancedResUNet
from src.metrics import Evaluator

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Training script for E06-D Model")
    parser.add_argument('--config', type=str, default=None, help='Path to configuration file')
    parser.add_argument('--experiment_id', type=str, default='phase12_reproduction', help='Experiment ID for results directory')
    args = parser.parse_args()

    base_dir = project_root
    
    # 1. Default Configuration matching Phase 10.1 Ground Truth for E06-D
    config = {
        'optimizer': 'AdamW',
        'initial_lr': 2e-4,
        'scheduler': 'CosineAnnealingLR',
        'batch_size': 16 if torch.cuda.is_available() else 4,
        'epochs': 120,
        'seed': 42,
        'loss': 'Charbonnier',
        'model': 'AdvancedResUNet',
        'use_se': True,
        'use_dilated_bottleneck': True,
        'synthetic_ratio': 0.0,
        'loss_params': {
            'char_eps': 1e-3,
            'lambda_char': 1.0,
            'lambda_ssim': 0.0,
            'lambda_grad': 0.0
        }
    }

    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            file_config = yaml.safe_load(f)
            config.update(file_config)

    # Determine results directory
    results_dir = os.path.join(base_dir, "results", args.experiment_id)
    checkpoints_dir = os.path.join(results_dir, "checkpoints")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Save active config
    with open(os.path.join(results_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
        
    # 2. Data
    batch_size = config['batch_size']
    num_epochs = config['epochs']
    lr = config['initial_lr']
    
    print("Initializing DataLoaders...")
    train_loader, val_loader = get_dataloaders(
        base_dir, val_split=0.1, batch_size=batch_size, seed=config['seed'],
        synthetic_ratio=config['synthetic_ratio']
    )
    evaluator = Evaluator(device=device)
    
    # 3. Model & Loss
    print("Initializing Model...")
    model = AdvancedResUNet(use_se=config['use_se'], use_dilated_bottleneck=config['use_dilated_bottleneck']).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {num_params:,}")
    
    from src.losses import CombinedLoss
    loss_params = config.get('loss_params', {})
    criterion = CombinedLoss(
        loss_type=config['loss'],
        char_eps=loss_params.get('char_eps', 1e-3),
        lambda_char=loss_params.get('lambda_char', 1.0),
        lambda_ssim=loss_params.get('lambda_ssim', 0.0),
        lambda_grad=loss_params.get('lambda_grad', 0.0)
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    best_psnr = 0.0
    logs = []
    
    for epoch in range(1, num_epochs + 1):
        # TRAIN
        model.train()
        train_loss = 0.0
        
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
        
        with torch.no_grad():
            for noisy, gt, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]"):
                noisy, gt = noisy.to(device), gt.to(device)
                
                pred = model(noisy)
                loss = criterion(pred, gt)
                val_loss += loss.item() * noisy.size(0)
                
                # Official metric clamping logic
                pred_clipped = torch.clamp(pred, 0.0, 1.0)
                metrics = evaluator.evaluate_batch(pred_clipped, gt)
                val_psnr += metrics['psnr'] * noisy.size(0)
                    
        val_loss /= len(val_loader.dataset)
        val_psnr /= len(val_loader.dataset)
        
        print(f"Epoch {epoch} Summary: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | PSNR: {val_psnr:.2f}")
        
        logs.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'psnr': val_psnr,
        })
        
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, "best_model.pth"))
            print(f"--> Saved new best model with PSNR {best_psnr:.2f}")
            
    # Save metrics CSV
    df = pd.DataFrame(logs)
    df.to_csv(os.path.join(results_dir, "metrics.csv"), index=False)
    print("Training complete.")

if __name__ == "__main__":
    main()
