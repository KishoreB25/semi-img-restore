import os
import subprocess
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt

def run_ablation(experiment_id, use_se, use_dilated):
    cmd = [
        r"venv\Scripts\python.exe",
        "src/train.py",
        "--loss_type", "Charbonnier",
        "--experiment_id", experiment_id,
        "--epochs", "30"
    ]
    if use_se:
        cmd.append("--use_se")
    if use_dilated:
        cmd.append("--use_dilated_bottleneck")
        
    print(f"\n==================================================")
    print(f"Starting Ablation: {experiment_id}")
    print(f"Command: {' '.join(cmd)}")
    print(f"==================================================")
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    
    if process.returncode != 0:
        raise RuntimeError(f"Experiment {experiment_id} failed with exit code {process.returncode}")

def generate_comparison_grid(phase_dir, experiments):
    print("\nGenerating Phase 06 Visual Comparison Grid...")
    # Find 3 sample images from the control experiment's visuals folder
    visuals_pattern = os.path.join(phase_dir, "E06-A", "visuals", "*_epoch_*.png")
    all_visuals = glob.glob(visuals_pattern)
    sample_names = sorted(list(set([os.path.basename(f).split('_epoch_')[0] for f in all_visuals])))[:3]
    
    if len(sample_names) > 0:
        fig, axes = plt.subplots(len(sample_names), len(experiments) + 2, figsize=(4 * (len(experiments) + 2), 4 * len(sample_names)))
        if len(sample_names) == 1:
            axes = np.expand_dims(axes, axis=0)
            
        for row_idx, sample in enumerate(sample_names):
            for col_idx, exp_id in enumerate(experiments):
                exp_vis_dir = os.path.join(phase_dir, exp_id, "visuals")
                sample_files = sorted(glob.glob(os.path.join(exp_vis_dir, f"{sample}_epoch_*.png")))
                if sample_files:
                    best_file = sample_files[-1]
                    img = plt.imread(best_file)
                    
                    h, w, c = img.shape
                    panel_w = w // 4
                    
                    if col_idx == 0:
                        # Col 0: NoisyLR
                        axes[row_idx, 0].imshow(img[:, :panel_w])
                        axes[row_idx, 0].set_title("NoisyLR")
                        axes[row_idx, 0].axis("off")
                        
                        # Col 1: Ground Truth
                        axes[row_idx, 1].imshow(img[:, 2*panel_w:3*panel_w])
                        axes[row_idx, 1].set_title("Ground Truth")
                        axes[row_idx, 1].axis("off")
                        
                    # Col 2+: Predictions
                    axes[row_idx, col_idx + 2].imshow(img[:, panel_w:2*panel_w])
                    axes[row_idx, col_idx + 2].set_title(exp_id)
                    axes[row_idx, col_idx + 2].axis("off")
                    
        plt.tight_layout()
        plt.savefig(os.path.join(phase_dir, "visual_comparison.png"))
        plt.close()
        print("Visual comparison grid saved.")

def main():
    base_dir = r"d:\semi-img-restore"
    phase_dir = os.path.join(base_dir, "results", "phase06_ablations")
    os.makedirs(phase_dir, exist_ok=True)
    
    experiments = [
        {"id": "E06-A", "use_se": False, "use_dilated": False},
        {"id": "E06-D", "use_se": True, "use_dilated": True}
    ]
    
    # 1. Run experiments sequentially
    for exp in experiments:
        run_ablation(exp["id"], exp["use_se"], exp["use_dilated"])
        
    # 2. Compile metrics
    summary_data = []
    exp_ids = []
    for exp in experiments:
        exp_ids.append(exp["id"])
        metrics_file = os.path.join(phase_dir, exp["id"], "metrics.csv")
        if os.path.exists(metrics_file):
            df = pd.read_csv(metrics_file)
            best_idx = df["psnr"].idxmax()
            best_row = df.loc[best_idx]
            summary_data.append({
                "Experiment": exp["id"],
                "PSNR": float(best_row["psnr"]),
                "SSIM": float(best_row["ssim"]),
                "LPIPS": float(best_row["lpips"]),
                "Latency_ms": float(best_row["latency_ms"]),
                "GPU_Mem_MB": float(best_row["gpu_mem_mb"]),
                "Params": int(best_row["params"])
            })
            
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(phase_dir, "ablation_matrix.csv"), index=False)
    print("\n--- Phase 06 Ablation Results ---")
    print(summary_df.to_string(index=False))
    
    # 3. Create visual comparison
    generate_comparison_grid(phase_dir, exp_ids)

if __name__ == "__main__":
    main()
