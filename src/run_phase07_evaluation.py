import os
import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset import get_dataloaders
from resunet import ResUNet
from resunet_advanced import AdvancedResUNet
from metrics import Evaluator

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def get_latency_stats(model_or_fn, x, device, is_model=True, iters=100):
    if is_model:
        model_or_fn.eval()
    
    # warmup
    with torch.no_grad():
        for _ in range(10):
            if is_model:
                _ = model_or_fn(x)
            else:
                _ = model_or_fn(x, size=(256, 256), mode="bicubic", align_corners=False)
            
    times = []
    with torch.no_grad():
        for _ in range(iters):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            start_time = time.time()
            if is_model:
                _ = model_or_fn(x)
            else:
                _ = model_or_fn(x, size=(256, 256), mode="bicubic", align_corners=False)
                
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            times.append((time.time() - start_time) * 1000)
            
    return np.mean(times), np.median(times), np.std(times)

def save_comparison_grid(noisy, gt, pred_bicubic, pred_p4, pred_e06, fname, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy and slice first batch item
    n_img = noisy[0, 0].cpu().numpy()
    gt_img = gt[0, 0].cpu().numpy()
    b_img = pred_bicubic[0, 0].cpu().numpy()
    p4_img = pred_p4[0, 0].cpu().numpy()
    e06_img = pred_e06[0, 0].cpu().numpy()
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    axes[0].imshow(n_img, cmap='gray')
    axes[0].set_title("Noisy LR (128x128)")
    axes[0].axis('off')
    
    axes[1].imshow(b_img, cmap='gray')
    axes[1].set_title("Bicubic Baseline")
    axes[1].axis('off')
    
    axes[2].imshow(p4_img, cmap='gray')
    axes[2].set_title("Phase 04 ResUNet")
    axes[2].axis('off')
    
    axes[3].imshow(e06_img, cmap='gray')
    axes[3].set_title("Phase 06 E06-D")
    axes[3].axis('off')
    
    axes[4].imshow(gt_img, cmap='gray')
    axes[4].set_title("Ground Truth")
    axes[4].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{fname[0]}_comparison.png"))
    plt.close()

def main():
    base_dir = r"d:\semi-img-restore"
    save_dir = os.path.join(base_dir, "results", "phase07_evaluation")
    visuals_dir = os.path.join(save_dir, "visuals")
    failures_dir = os.path.join(save_dir, "failure_cases")
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(visuals_dir, exist_ok=True)
    os.makedirs(failures_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    _, val_loader = get_dataloaders(base_dir, val_split=0.1, batch_size=1, seed=42)
    
    # 2. Setup Evaluator
    evaluator = Evaluator(device)
    
    # 3. Load Models
    print("Loading Phase 04 ResUNet...")
    model_p4 = ResUNet().to(device)
    p4_ckpt_path = os.path.join(base_dir, "results", "phase04_losses", "phase04_loss_charbonnier", "checkpoints", "best.pth")
    model_p4.load_state_dict(torch.load(p4_ckpt_path, map_location=device))
    model_p4.eval()
    
    print("Loading Phase 06 E06-D...")
    model_e06 = AdvancedResUNet(use_se=True, use_dilated_bottleneck=True).to(device)
    e06_ckpt_path = os.path.join(base_dir, "results", "phase06_E06-D_final", "checkpoints", "best.pth")
    model_e06.load_state_dict(torch.load(e06_ckpt_path, map_location=device))
    model_e06.eval()
    
    # 4. Parameters and Latency
    params_bicubic = 0
    params_p4 = count_parameters(model_p4)
    params_e06 = count_parameters(model_e06)
    
    # Measure Latency (batch size 1)
    dummy_input = torch.randn(1, 1, 128, 128).to(device)
    
    b_mean, b_med, b_std = get_latency_stats(F.interpolate, dummy_input, device, is_model=False)
    p4_mean, p4_med, p4_std = get_latency_stats(model_p4, dummy_input, device, is_model=True)
    e06_mean, e06_med, e06_std = get_latency_stats(model_e06, dummy_input, device, is_model=True)
    
    print(f"Latencies (Mean | Median | Std) ms:")
    print(f"  Bicubic: {b_mean:.2f} | {b_med:.2f} | {b_std:.2f}")
    print(f"  Phase04: {p4_mean:.2f} | {p4_med:.2f} | {p4_std:.2f}")
    print(f"  Phase06: {e06_mean:.2f} | {e06_med:.2f} | {e06_std:.2f}")
    
    # 5. Evaluate
    metrics = {
        'Bicubic': {'psnr': 0, 'ssim': 0, 'lpips': 0},
        'Phase04': {'psnr': 0, 'ssim': 0, 'lpips': 0},
        'Phase06': {'psnr': 0, 'ssim': 0, 'lpips': 0}
    }
    
    e06_sample_metrics = []
    
    count = 0
    with torch.no_grad():
        for noisy, gt, fname in tqdm(val_loader, desc="Evaluating"):
            noisy, gt = noisy.to(device), gt.to(device)
            count += 1
            
            # Bicubic
            pred_bicubic = F.interpolate(noisy, size=(256, 256), mode="bicubic", align_corners=False)
            pred_bicubic_clipped = torch.clamp(pred_bicubic, 0.0, 1.0)
            res_b = evaluator.evaluate_batch(pred_bicubic_clipped, gt)
            metrics['Bicubic']['psnr'] += res_b['psnr']
            metrics['Bicubic']['ssim'] += res_b['ssim']
            metrics['Bicubic']['lpips'] += res_b['lpips']
            
            # Phase 04
            pred_p4 = model_p4(noisy)
            pred_p4_clipped = torch.clamp(pred_p4, 0.0, 1.0)
            res_p4 = evaluator.evaluate_batch(pred_p4_clipped, gt)
            metrics['Phase04']['psnr'] += res_p4['psnr']
            metrics['Phase04']['ssim'] += res_p4['ssim']
            metrics['Phase04']['lpips'] += res_p4['lpips']
            
            # Phase 06
            pred_e06 = model_e06(noisy)
            pred_e06_clipped = torch.clamp(pred_e06, 0.0, 1.0)
            res_e06 = evaluator.evaluate_batch(pred_e06_clipped, gt)
            metrics['Phase06']['psnr'] += res_e06['psnr']
            metrics['Phase06']['ssim'] += res_e06['ssim']
            metrics['Phase06']['lpips'] += res_e06['lpips']
            
            e06_sample_metrics.append({
                'fname': fname,
                'psnr': res_e06['psnr'],
                'ssim': res_e06['ssim'],
                'lpips': res_e06['lpips'],
                'noisy': noisy,
                'gt': gt,
                'pred_bicubic': pred_bicubic_clipped,
                'pred_p4': pred_p4_clipped,
                'pred_e06': pred_e06_clipped
            })
            
            # Save qualitative samples (just first 10)
            if count <= 10:
                save_comparison_grid(noisy, gt, pred_bicubic_clipped, pred_p4_clipped, pred_e06_clipped, fname, visuals_dir)
                
    # Average metrics
    for k in metrics:
        metrics[k]['psnr'] /= count
        metrics[k]['ssim'] /= count
        metrics[k]['lpips'] /= count
        
    # Find Failure Cases (worst 5 PSNR for E06-D)
    e06_sample_metrics.sort(key=lambda x: x['psnr'])
    failure_cases = e06_sample_metrics[:5]
    for fc in failure_cases:
        save_comparison_grid(fc['noisy'], fc['gt'], fc['pred_bicubic'], fc['pred_p4'], fc['pred_e06'], fc['fname'], failures_dir)
        
    # Generate Comparison Table
    table_data = [
        {"Model": "Bicubic", "PSNR": metrics['Bicubic']['psnr'], "SSIM": metrics['Bicubic']['ssim'], "LPIPS": metrics['Bicubic']['lpips'], "Parameters": params_bicubic, "Latency_Mean": b_mean, "Latency_Median": b_med},
        {"Model": "Phase 04 ResUNet", "PSNR": metrics['Phase04']['psnr'], "SSIM": metrics['Phase04']['ssim'], "LPIPS": metrics['Phase04']['lpips'], "Parameters": params_p4, "Latency_Mean": p4_mean, "Latency_Median": p4_med},
        {"Model": "Phase 06 E06-D", "PSNR": metrics['Phase06']['psnr'], "SSIM": metrics['Phase06']['ssim'], "LPIPS": metrics['Phase06']['lpips'], "Parameters": params_e06, "Latency_Mean": e06_mean, "Latency_Median": e06_med},
    ]
    df = pd.DataFrame(table_data)
    df.to_csv(os.path.join(save_dir, "final_audited_metrics.csv"), index=False)

    
    print("\n--- Phase 07 Final Evaluation ---")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
