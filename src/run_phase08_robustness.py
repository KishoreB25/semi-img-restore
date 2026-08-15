import os
print("Script is starting...")

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset import get_dataloaders
from resunet_advanced import AdvancedResUNet
from metrics import Evaluator
from resunet import ResUNet

def save_comparison_grid(noisy, gt, pred, fname, save_dir, title_prefix=""):
    os.makedirs(save_dir, exist_ok=True)
    
    n_img = noisy[0, 0].cpu().numpy()
    gt_img = gt[0, 0].cpu().numpy()
    p_img = pred[0, 0].cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(n_img, cmap='gray')
    axes[0].set_title(f"Noisy LR (128x128)")
    axes[0].axis('off')
    
    axes[1].imshow(p_img, cmap='gray')
    axes[1].set_title(f"{title_prefix}E06-D Prediction")
    axes[1].axis('off')
    
    axes[2].imshow(gt_img, cmap='gray')
    axes[2].set_title("Ground Truth")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{fname[0]}_comparison.png"))
    plt.close()

def categorize_failure(psnr, ssim, lpips):
    # Heuristic categorization based on relative metrics
    if lpips > 0.4:
        return "Texture Loss / Oversmoothing"
    elif ssim < 0.6:
        return "Structural Distortion"
    elif psnr < 25.0:
        return "Residual Noise"
    else:
        return "Complex / Mixed Artifacts"

def main():
    base_dir = r"d:\semi-img-restore"
    save_dir = os.path.join(base_dir, "results", "phase08_robustness")
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    _, val_loader = get_dataloaders(base_dir, val_split=0.1, batch_size=1, seed=42)
    
    # 2. Setup Evaluator & Model
    evaluator = Evaluator(device)
    
    print("Loading Phase 06 E06-D...")
    model_e06 = AdvancedResUNet(use_se=True, use_dilated_bottleneck=True).to(device)
    e06_ckpt_path = os.path.join(base_dir, "results", "phase06_E06-D_final", "checkpoints", "best.pth")
    model_e06.load_state_dict(torch.load(e06_ckpt_path, map_location=device))
    model_e06.eval()
    
    sample_metrics = []
    
    print("Evaluating all validation images...")
    with torch.no_grad():
        for noisy, gt, fname in tqdm(val_loader, desc="Evaluating"):
            noisy, gt = noisy.to(device), gt.to(device)
            
            pred = model_e06(noisy)
            pred_clipped = torch.clamp(pred, 0.0, 1.0)
            
            res = evaluator.evaluate_batch(pred_clipped, gt)
            
            sample_metrics.append({
                'filename': fname[0],
                'psnr': res['psnr'],
                'ssim': res['ssim'],
                'lpips': res['lpips'],
                'noisy': noisy.cpu(),
                'gt': gt.cpu(),
                'pred': pred_clipped.cpu()
            })
            
    # Create DataFrames
    df_per_image = pd.DataFrame([{k: v for k, v in m.items() if k not in ['noisy', 'gt', 'pred']} for m in sample_metrics])
    df_per_image.to_csv(os.path.join(save_dir, "per_image_metrics.csv"), index=False)
    
    # Calculate summary statistics
    stats = {
        'Metric': ['PSNR', 'SSIM', 'LPIPS'],
        'Mean': [df_per_image['psnr'].mean(), df_per_image['ssim'].mean(), df_per_image['lpips'].mean()],
        'Median': [df_per_image['psnr'].median(), df_per_image['ssim'].median(), df_per_image['lpips'].median()],
        'Std': [df_per_image['psnr'].std(), df_per_image['ssim'].std(), df_per_image['lpips'].std()],
        'Min': [df_per_image['psnr'].min(), df_per_image['ssim'].min(), df_per_image['lpips'].min()],
        'Max': [df_per_image['psnr'].max(), df_per_image['ssim'].max(), df_per_image['lpips'].max()]
    }
    df_stats = pd.DataFrame(stats)
    df_stats.to_csv(os.path.join(save_dir, "robustness_metrics.csv"), index=False)
    
    # Sort by PSNR to find best/worst
    sorted_metrics = sorted(sample_metrics, key=lambda x: x['psnr'])
    worst_5 = sorted_metrics[:5]
    best_5 = sorted_metrics[-5:]
    
    # Typical cases (around median PSNR)
    median_psnr = df_stats[df_stats['Metric'] == 'PSNR']['Median'].values[0]
    typical_cases = sorted(sample_metrics, key=lambda x: abs(x['psnr'] - median_psnr))[:5]
    
    # Save Images
    print("Generating qualitative comparisons...")
    for item in best_5:
        save_comparison_grid(item['noisy'], item['gt'], item['pred'], [item['filename']], save_dir, "BEST_")
    for item in worst_5:
        save_comparison_grid(item['noisy'], item['gt'], item['pred'], [item['filename']], save_dir, "WORST_")
    for item in typical_cases:
        save_comparison_grid(item['noisy'], item['gt'], item['pred'], [item['filename']], save_dir, "TYPICAL_")
        
    # Analyze failures
    failure_data = []
    for item in worst_5:
        category = categorize_failure(item['psnr'], item['ssim'], item['lpips'])
        failure_data.append({
            'filename': item['filename'],
            'psnr': item['psnr'],
            'ssim': item['ssim'],
            'lpips': item['lpips'],
            'category': category
        })
    df_failures = pd.DataFrame(failure_data)
    df_failures.to_csv(os.path.join(save_dir, "failure_analysis.csv"), index=False)
    
    # Latency trade-off (Hardcoded from audited values)
    p4_psnr = 27.821
    p4_lat = 17.10
    e06_psnr = 28.541
    e06_lat = 35.12
    
    lat_diff = e06_lat - p4_lat
    psnr_diff = e06_psnr - p4_psnr
    gain_per_ms = psnr_diff / lat_diff
    
    tradeoff_data = [{
        'Model': 'E06-D vs Phase 04',
        'Latency_Increase_ms': lat_diff,
        'PSNR_Increase_dB': psnr_diff,
        'PSNR_Gain_per_ms': gain_per_ms
    }]
    pd.DataFrame(tradeoff_data).to_csv(os.path.join(save_dir, "latency_quality_analysis.csv"), index=False)
    
    print("Phase 08 robustness evaluation complete!")

if __name__ == "__main__":
    main()
