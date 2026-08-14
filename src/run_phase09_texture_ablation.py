import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset import get_dataloaders
from resunet_advanced import AdvancedResUNet
from metrics import Evaluator

def save_comparison_grid(noisy, gt, pred_base, pred_cand, fname, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    n_img = noisy[0, 0].cpu().numpy()
    gt_img = gt[0, 0].cpu().numpy()
    pb_img = pred_base[0, 0].cpu().numpy()
    pc_img = pred_cand[0, 0].cpu().numpy()
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(n_img, cmap='gray')
    axes[0].set_title(f"Noisy LR")
    axes[0].axis('off')
    
    axes[1].imshow(pb_img, cmap='gray')
    axes[1].set_title("E06-D Baseline")
    axes[1].axis('off')
    
    axes[2].imshow(pc_img, cmap='gray')
    axes[2].set_title("E06-D + Grad")
    axes[2].axis('off')
    
    axes[3].imshow(gt_img, cmap='gray')
    axes[3].set_title("Ground Truth")
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{fname}_comparison.png"))
    plt.close()

def main():
    base_dir = r"d:\semi-img-restore"
    save_dir = os.path.join(base_dir, "results", "phase09_texture_ablation")
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    _, val_loader = get_dataloaders(base_dir, val_split=0.1, batch_size=1, seed=42)
    
    # 2. Setup Evaluator & Models
    evaluator = Evaluator(device)
    
    print("Loading Baseline E06-D...")
    model_base = AdvancedResUNet(use_se=True, use_dilated_bottleneck=True).to(device)
    base_ckpt = os.path.join(base_dir, "results", "phase06_E06-D_final", "checkpoints", "best.pth")
    model_base.load_state_dict(torch.load(base_ckpt, map_location=device))
    model_base.eval()
    
    print("Loading Candidate E06-D + Grad Loss...")
    model_cand = AdvancedResUNet(use_se=True, use_dilated_bottleneck=True).to(device)
    cand_ckpt = os.path.join(base_dir, "results", "phase09_E06-D_grad_ablation", "checkpoints", "best.pth")
    model_cand.load_state_dict(torch.load(cand_ckpt, map_location=device))
    model_cand.eval()
    
    sample_metrics = []
    failure_cases = ["000051.npy", "001385.npy", "000900.npy", "000354.npy", "002639.npy"]
    failure_images = {}
    
    print("Evaluating models...")
    with torch.no_grad():
        for noisy, gt, fname in tqdm(val_loader, desc="Evaluating"):
            noisy, gt = noisy.to(device), gt.to(device)
            
            # Predict
            pred_base = model_base(noisy)
            pred_cand = model_cand(noisy)
            
            # Clamp [0,1]
            pred_base = torch.clamp(pred_base, 0.0, 1.0)
            pred_cand = torch.clamp(pred_cand, 0.0, 1.0)
            
            res_base = evaluator.evaluate_batch(pred_base, gt)
            res_cand = evaluator.evaluate_batch(pred_cand, gt)
            
            sample_metrics.append({
                'filename': fname[0],
                'Base_PSNR': res_base['psnr'],
                'Base_SSIM': res_base['ssim'],
                'Base_LPIPS': res_base['lpips'],
                'Cand_PSNR': res_cand['psnr'],
                'Cand_SSIM': res_cand['ssim'],
                'Cand_LPIPS': res_cand['lpips'],
            })
            
            if fname[0] in failure_cases:
                failure_images[fname[0]] = (noisy.cpu(), gt.cpu(), pred_base.cpu(), pred_cand.cpu())
                
    # Create DataFrames
    df_per_image = pd.DataFrame(sample_metrics)
    df_per_image.to_csv(os.path.join(save_dir, "per_image_metrics.csv"), index=False)
    
    # Overall Metrics
    overall = {
        'Model': ['E06-D Baseline', 'E06-D + Grad Loss'],
        'PSNR': [df_per_image['Base_PSNR'].mean(), df_per_image['Cand_PSNR'].mean()],
        'SSIM': [df_per_image['Base_SSIM'].mean(), df_per_image['Cand_SSIM'].mean()],
        'LPIPS': [df_per_image['Base_LPIPS'].mean(), df_per_image['Cand_LPIPS'].mean()]
    }
    df_overall = pd.DataFrame(overall)
    df_overall.to_csv(os.path.join(save_dir, "final_metrics.csv"), index=False)
    
    print("\n--- Phase 09 Final Comparison ---")
    print(df_overall.to_string(index=False))
    
    # Save Failure Case Grids
    print("Saving failure case comparisons...")
    failure_df = df_per_image[df_per_image['filename'].isin(failure_cases)]
    failure_df.to_csv(os.path.join(save_dir, "failure_case_comparisons.csv"), index=False)
    
    for fname, imgs in failure_images.items():
        save_comparison_grid(imgs[0], imgs[1], imgs[2], imgs[3], fname.replace('.npy', ''), save_dir)
        
    print("Phase 09 evaluation complete!")

if __name__ == "__main__":
    main()
