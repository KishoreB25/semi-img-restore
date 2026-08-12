import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dataset import get_dataloaders
from model import TinyResNet
from metrics import Evaluator
import yaml

def save_visuals(noisy_np, pred_np, gt_np, fname, save_dir, step):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(noisy_np, cmap='gray')
    axes[0].set_title(f"NoisyLR")
    axes[0].axis('off')
    
    axes[1].imshow(pred_np, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f"Prediction (Step {step})")
    axes[1].axis('off')
    
    axes[2].imshow(gt_np, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f"Ground Truth")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{fname}_step_{step}.png"))
    plt.close()

def main():
    base_dir = r"d:\semi-img-restore"
    results_dir = os.path.join(base_dir, "results", "phase02_overfit")
    os.makedirs(results_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load data
    train_loader, _ = get_dataloaders(base_dir, val_split=0.1, batch_size=2, seed=42)
    evaluator = Evaluator(device=device)
    
    # Extract exactly 1 batch (2 pairs)
    noisy_batch, gt_batch, fnames = next(iter(train_loader))
    noisy_batch, gt_batch = noisy_batch.to(device), gt_batch.to(device)
    
    print(f"Overfitting on samples: {fnames}")
    
    # 2. Setup Model & Loss
    model = TinyResNet().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    
    # 3. Training Loop
    num_iterations = 2000
    log_interval = 100
    
    logs = []
    
    for step in range(1, num_iterations + 1):
        model.train()
        optimizer.zero_grad()
        
        pred = model(noisy_batch)
        loss = criterion(pred, gt_batch)
        
        loss.backward()
        optimizer.step()
        
        if step % log_interval == 0 or step == 1 or step == num_iterations:
            model.eval()
            with torch.no_grad():
                val_pred = model(noisy_batch)
                metrics = evaluator.evaluate_batch(val_pred, gt_batch)
            
            logs.append({
                'step': step,
                'loss': loss.item(),
                'psnr': metrics['psnr'],
                'ssim': metrics['ssim']
            })
            
            print(f"Step {step:04d} | Loss: {loss.item():.4f} | PSNR: {metrics['psnr']:.2f} | SSIM: {metrics['ssim']:.4f}")
            
            # Save visual for the first sample
            if step == num_iterations or step == 1 or step == (num_iterations // 2):
                save_visuals(
                    noisy_batch[0, 0].cpu().numpy(), 
                    val_pred[0, 0].clamp(0, 1).cpu().numpy(), 
                    gt_batch[0, 0].cpu().numpy(), 
                    fnames[0].replace('.npy', ''), 
                    results_dir, 
                    step
                )

    # Save metrics
    df = pd.DataFrame(logs)
    df.to_csv(os.path.join(results_dir, "metrics.csv"), index=False)
    
    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(df['step'], df['loss'], label='L1 Loss')
    plt.title("Overfit Sanity Check - Loss Curve")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "loss_curve.png"))
    plt.close()
    
    # Save final prediction numpy
    with torch.no_grad():
        final_pred = model(noisy_batch)
        final_pred = final_pred.clamp(0, 1).cpu().numpy()
        np.save(os.path.join(results_dir, "sample0_prediction.npy"), final_pred[0, 0])
        
    # Save config
    config = {
        'optimizer': 'AdamW',
        'learning_rate': 1e-3,
        'weight_decay': 0,
        'batch_size': 2,
        'iterations': num_iterations,
        'model': 'TinyResNet'
    }
    with open(os.path.join(results_dir, "run_config.yaml"), "w") as f:
        yaml.dump(config, f)
        
    print(f"Overfit check complete. Results saved to {results_dir}")

if __name__ == "__main__":
    main()
