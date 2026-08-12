import os
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import get_dataloaders
from metrics import Evaluator

def save_visuals(noisy_np, pred_np, gt_np, fname, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(noisy_np[0, 0], cmap='gray')
    axes[0].set_title("NoisyLR (128x128)")
    axes[0].axis('off')
    
    axes[1].imshow(pred_np[0, 0], cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("Bicubic Restored (256x256)")
    axes[1].axis('off')
    
    axes[2].imshow(gt_np[0, 0], cmap='gray', vmin=0, vmax=1)
    axes[2].set_title("Ground Truth (256x256)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{fname}.png"))
    plt.close()

def main():
    base_dir = r"d:\semi-img-restore"
    results_dir = os.path.join(base_dir, "results", "phase01_baseline")
    visuals_dir = os.path.join(results_dir, "visuals")
    os.makedirs(results_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get dataloaders
    train_loader, val_loader = get_dataloaders(base_dir, val_split=0.1, batch_size=16)
    evaluator = Evaluator(device=device)
    
    print(f"Validation set size: {len(val_loader.dataset)} images")
    
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    total_time = 0.0
    num_batches = len(val_loader)
    
    visuals_saved = 0
    
    with torch.no_grad():
        for batch_idx, (noisy, gt, fnames) in enumerate(tqdm(val_loader, desc="Evaluating Bicubic Baseline")):
            noisy = noisy.to(device)
            gt = gt.to(device)
            
            start_time = time.time()
            
            # Phase 01 Bicubic logic:
            # 1. Input is raw float32 NoisyLR without clipping/normalization
            # 2. Upsample from 128x128 to 256x256
            pred = F.interpolate(noisy, size=(256, 256), mode='bicubic', align_corners=False)
            
            # 3. Clip final prediction to [0,1]
            pred = pred.clamp(0.0, 1.0)
            
            # We measure time before metrics
            batch_time = time.time() - start_time
            total_time += batch_time
            
            # Calculate metrics
            batch_metrics = evaluator.evaluate_batch(pred, gt)
            
            total_psnr += batch_metrics['psnr']
            total_ssim += batch_metrics['ssim']
            total_lpips += batch_metrics['lpips']
            
            # Save a few visuals
            if visuals_saved < 10:
                # take first image in batch
                noisy_np = noisy.cpu().numpy()
                pred_np = pred.cpu().numpy()
                gt_np = gt.cpu().numpy()
                save_visuals(noisy_np, pred_np, gt_np, fnames[0].replace('.npy', ''), visuals_dir)
                visuals_saved += 1
                
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    avg_lpips = total_lpips / num_batches
    avg_time_per_image_ms = (total_time / len(val_loader.dataset)) * 1000
    
    metrics = {
        "model": "bicubic_baseline",
        "psnr": avg_psnr,
        "ssim": avg_ssim,
        "lpips": avg_lpips,
        "runtime_ms_per_image": avg_time_per_image_ms,
        "dataset_size": len(val_loader.dataset)
    }
    
    print("\n--- Baseline Results ---")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
            
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Create experiments.csv if not exists
    csv_path = os.path.join(base_dir, "experiments.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            f.write("experiment_id,model,loss,augmentation,synthetic_data,psnr,ssim,lpips,runtime_ms_per_image,notes\n")
            
    with open(csv_path, "a") as f:
        f.write(f"phase01_baseline,bicubic,None,None,None,{avg_psnr:.4f},{avg_ssim:.4f},{avg_lpips:.4f},{avg_time_per_image_ms:.4f},Initial baseline\n")
        
    print(f"Results saved to {results_dir}")

if __name__ == "__main__":
    main()
