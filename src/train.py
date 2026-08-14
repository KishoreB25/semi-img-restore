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
from resunet_advanced import AdvancedResUNet
from metrics import Evaluator
import time

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Path to configuration file')
    parser.add_argument('--loss_type', type=str, default=None, help='Loss type (L1, Charbonnier, Char_SSIM, Char_SSIM_Grad)')
    parser.add_argument('--experiment_id', type=str, default='phase03_neural_baseline', help='Experiment ID for results directory')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--synthetic_ratio', type=float, default=None, help='Ratio of synthetic to real data (e.g. 1.0 = 1:1)')
    parser.add_argument('--use_se', action='store_true', help='Use Squeeze-and-Excitation')
    parser.add_argument('--use_dilated_bottleneck', action='store_true', help='Use Multi-Scale Dilated Bottleneck')
    args = parser.parse_args()

    base_dir = r"d:\semi-img-restore"
    
    # 1. Default Configuration
    config = {
        'optimizer': 'AdamW',
        'initial_lr': 2e-4,
        'scheduler': 'CosineAnnealingLR',
        'batch_size': 16 if torch.cuda.is_available() else 4,
        'epochs': 30,
        'seed': 42,
        'loss': 'L1',
        'model': 'ResUNet',
        'synthetic_ratio': 0.0,
        'loss_params': {
            'char_eps': 1e-3,
            'lambda_char': 0.8,
            'lambda_ssim': 0.2,
            'lambda_grad': 0.1
        }
    }

    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            file_config = yaml.safe_load(f)
            config.update(file_config)

    # CLI Overrides
    if args.loss_type:
        config['loss'] = args.loss_type
    if args.epochs:
        config['epochs'] = args.epochs
    if args.lr:
        config['initial_lr'] = args.lr
    if args.synthetic_ratio is not None:
        config['synthetic_ratio'] = args.synthetic_ratio

    # Load synthetic degradation config if it exists
    deg_cfg = None
    deg_cfg_path = os.path.join(base_dir, "configs", "synthetic_degradation.yaml")
    if os.path.exists(deg_cfg_path):
        with open(deg_cfg_path, 'r') as f:
            deg_cfg_data = yaml.safe_load(f)
            deg_cfg = deg_cfg_data.get('degradation', None)
            # If not set in config/CLI, take from configs/synthetic_degradation.yaml
            if args.synthetic_ratio is None and 'synthetic_ratio' not in config:
                config['synthetic_ratio'] = deg_cfg_data.get('training', {}).get('synthetic_ratio', 0.0)

    # Determine results directory
    if args.experiment_id.startswith('phase04'):
        results_dir = os.path.join(base_dir, "results", "phase04_losses", args.experiment_id)
    elif args.experiment_id.startswith('phase05'):
        results_dir = os.path.join(base_dir, "results", "phase05_synthetic", args.experiment_id)
    else:
        results_dir = os.path.join(base_dir, "results", args.experiment_id)

    visuals_dir = os.path.join(results_dir, "visuals")
    checkpoints_dir = os.path.join(results_dir, "checkpoints")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Save active config
    os.makedirs(os.path.join(base_dir, "configs"), exist_ok=True)
    with open(os.path.join(results_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
        
    # 2. Data
    batch_size = config['batch_size']
    num_epochs = config['epochs']
    lr = config['initial_lr']
    synth_ratio = config['synthetic_ratio']
    
    train_loader, val_loader = get_dataloaders(
        base_dir, val_split=0.1, batch_size=batch_size, seed=42,
        synthetic_ratio=synth_ratio, degradation_cfg=deg_cfg
    )
    evaluator = Evaluator(device=device)
    
    # 3. Model & Loss
    from losses import CombinedLoss
    
    if args.use_se or args.use_dilated_bottleneck or config.get('model') == 'AdvancedResUNet':
        model = AdvancedResUNet(use_se=args.use_se, use_dilated_bottleneck=args.use_dilated_bottleneck).to(device)
        config['model'] = 'AdvancedResUNet'
    else:
        model = ResUNet().to(device)
        
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {num_params:,}")
    
    loss_params = config.get('loss_params', {})
    criterion = CombinedLoss(
        loss_type=config['loss'],
        char_eps=loss_params.get('char_eps', 1e-3),
        lambda_char=loss_params.get('lambda_char', 0.8),
        lambda_ssim=loss_params.get('lambda_ssim', 0.2),
        lambda_grad=loss_params.get('lambda_grad', 0.1)
    )
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
        inference_times = []
        
        with torch.no_grad():
            for noisy, gt, fnames in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]"):
                noisy, gt = noisy.to(device), gt.to(device)
                
                start_time = time.time()
                pred = model(noisy)
                inference_times.append((time.time() - start_time) / noisy.size(0))
                
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
        
        avg_inference_latency = sum(inference_times) / len(inference_times) * 1000 # in ms
        gpu_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0
        
        print(f"Epoch {epoch} Summary: Train L1: {train_loss:.4f} | Val L1: {val_loss:.4f} | PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.4f} | LPIPS: {val_lpips:.4f}")
        print(f"Latency: {avg_inference_latency:.2f}ms | GPU Mem: {gpu_memory:.1f}MB")
        
        logs.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'psnr': val_psnr,
            'ssim': val_ssim,
            'lpips': val_lpips,
            'lr': optimizer.param_groups[0]['lr'],
            'latency_ms': avg_inference_latency,
            'gpu_mem_mb': gpu_memory,
            'params': num_params
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
        best_epoch_log = df.loc[df['psnr'].idxmax()]
        f.write(f"{args.experiment_id},{config['model']},{config['loss']},{config.get('augmentation', 'None')},{config.get('synthetic_ratio', 0.0)},{best_epoch_log['psnr']:.4f},{best_epoch_log['ssim']:.4f},{best_epoch_log['lpips']:.4f},{best_epoch_log['latency_ms']:.2f},{best_epoch_log['params']}\n")

if __name__ == "__main__":
    main()
