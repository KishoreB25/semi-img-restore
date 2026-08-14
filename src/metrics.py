import torch
import torch.nn.functional as F
import lpips
import numpy as np
from skimage.metrics import structural_similarity as ssim_metric

class Evaluator:
    def __init__(self, device='cpu'):
        self.device = device
        self.lpips_net = lpips.LPIPS(net='alex').to(self.device)
        # Avoid tracking gradients for evaluation
        self.lpips_net.eval()

    def calculate_psnr(self, pred, target, data_range=1.0):
        """
        Calculates PSNR.
        pred, target: torch tensors of shape (B, 1, H, W)
        """
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return float('inf')
        return 10 * torch.log10((data_range ** 2) / mse).item()

    def calculate_ssim(self, pred, target, data_range=1.0):
        """
        Calculates SSIM.
        pred, target: torch tensors of shape (B, 1, H, W)
        We convert to numpy and use skimage's SSIM for accuracy.
        """
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        batch_size = pred_np.shape[0]
        ssim_val = 0
        for i in range(batch_size):
            # images should be (H, W) for skimage SSIM
            ssim_val += ssim_metric(
                target_np[i, 0], 
                pred_np[i, 0], 
                data_range=data_range,
                channel_axis=None
            )
        return ssim_val / batch_size

    def calculate_lpips(self, pred, target):
        """
        Calculates LPIPS.
        pred, target: torch tensors of shape (B, 1, H, W) in [0, 1] range.
        LPIPS expects inputs in [-1, 1], so we scale them.
        We also replicate the grayscale channel to 3 channels.
        """
        # Replicate channels: (B, 1, H, W) -> (B, 3, H, W)
        pred_3c = pred.repeat(1, 3, 1, 1)
        target_3c = target.repeat(1, 3, 1, 1)
        
        # Scale from [0, 1] to [-1, 1]
        pred_scaled = pred_3c * 2.0 - 1.0
        target_scaled = target_3c * 2.0 - 1.0
        
        with torch.no_grad():
            lpips_val = self.lpips_net(pred_scaled, target_scaled)
            
        return lpips_val.mean().item()

    def evaluate_batch(self, pred, target):
        """
        pred, target: torch tensors of shape (B, 1, H, W) in [0, 1] range.
        """
        # Ensure values are within [0, 1]
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)
        
        psnr = self.calculate_psnr(pred, target)
        ssim = self.calculate_ssim(pred, target)
        lpips_score = self.calculate_lpips(pred, target)
        
        return {
            'psnr': psnr,
            'ssim': ssim,
            'lpips': lpips_score
        }
