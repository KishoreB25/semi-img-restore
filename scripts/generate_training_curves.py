import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_curves(metrics_csv, output_dir):
    if not os.path.exists(metrics_csv):
        print(f"Metrics file not found: {metrics_csv}")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(metrics_csv)
    
    epochs = df['epoch']
    
    # 1. Loss vs Epoch
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, df['train_loss'], label='Train Loss (Charbonnier)', color='blue')
    plt.plot(epochs, df['val_loss'], label='Validation Loss (Charbonnier)', color='red')
    plt.title('Training and Validation Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()
    
    # 2. PSNR vs Epoch
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, df['psnr'], label='Validation PSNR', color='green')
    plt.title('Validation PSNR vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'psnr_curve.png'))
    plt.close()
    
    # 3. SSIM vs Epoch
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, df['ssim'], label='Validation SSIM', color='orange')
    plt.title('Validation SSIM vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'ssim_curve.png'))
    plt.close()
    
    # 4. LPIPS vs Epoch
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, df['lpips'], label='Validation LPIPS', color='purple')
    plt.title('Validation LPIPS vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('LPIPS')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'lpips_curve.png'))
    plt.close()
    
    print(f"Training curves successfully generated and saved to {output_dir}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics_path = os.path.join(base_dir, 'results', 'phase06_E06-D_final', 'metrics.csv')
    figures_dir = os.path.join(base_dir, 'figures')
    
    generate_curves(metrics_path, figures_dir)
