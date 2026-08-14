import torch
import torch.nn.functional as F
import random

def add_gaussian_noise(x, sigma):
    """
    y = x + n, where n ~ N(0, sigma^2)
    """
    if sigma <= 0:
        return x
    noise = torch.randn_like(x) * sigma
    return x + noise

def add_speckle_noise(x, sigma):
    """
    y = x * n = x * (1 + epsilon), where epsilon ~ N(0, sigma^2)
    """
    if sigma <= 0:
        return x
    noise = torch.randn_like(x) * sigma
    return x + x * noise

def bicubic_downsample(x):
    """
    Downsamples x (from 256x256 to 128x128).
    x is expected to have shape [C, H, W] or [B, C, H, W].
    """
    # Ensure 4D shape for F.interpolate
    is_3d = (x.ndim == 3)
    if is_3d:
        x = x.unsqueeze(0)
        
    y = F.interpolate(x, size=(128, 128), mode='bicubic', align_corners=False)
    
    if is_3d:
        y = y.squeeze(0)
    return y

class SyntheticDegradationEngine:
    def __init__(self, min_g=0.00, max_g=0.10, min_s=0.10, max_s=0.20):
        self.min_g = min_g
        self.max_g = max_g
        self.min_s = min_s
        self.max_s = max_s
        
    def degrade(self, x, order=None, seed=None):
        """
        Applies degradations on input tensor x (shape [C, H, W] or [H, W]).
        x is expected to be a PyTorch tensor (GT scale [0,1]).
        """
        if seed is not None:
            # Set state for repeatability if needed
            # (Use local generator or simple Python random if we want local scope)
            # To avoid global side effects:
            g = torch.Generator(device=x.device)
            g.manual_seed(seed)
            random_state = random.Random(seed)
        else:
            g = None
            random_state = random
            
        # Sample parameters
        if g is not None:
            sigma_g = self.min_g + (self.max_g - self.min_g) * torch.rand(1, generator=g).item()
            sigma_s = self.min_s + (self.max_s - self.min_s) * torch.rand(1, generator=g).item()
        else:
            sigma_g = random.uniform(self.min_g, self.max_g)
            sigma_s = random.uniform(self.min_s, self.max_s)
            
        # Possible orderings
        orders = [
            ['gaussian', 'speckle', 'downsample'],
            ['speckle', 'gaussian', 'downsample'],
            ['downsample', 'gaussian', 'speckle'],
            ['gaussian', 'downsample', 'speckle'],
            ['speckle', 'downsample', 'gaussian'],
            ['downsample', 'speckle', 'gaussian']
        ]
        
        if order is None:
            order = random_state.choice(orders)
        elif isinstance(order, int):
            order = orders[order % len(orders)]
            
        out = x.clone()
        
        for op in order:
            if op == 'gaussian':
                if g is not None:
                    noise = torch.randn(out.shape, generator=g, device=out.device) * sigma_g
                    out = out + noise
                else:
                    out = add_gaussian_noise(out, sigma_g)
            elif op == 'speckle':
                if g is not None:
                    noise = torch.randn(out.shape, generator=g, device=out.device) * sigma_s
                    out = out + out * noise
                else:
                    out = add_speckle_noise(out, sigma_s)
            elif op == 'downsample':
                out = bicubic_downsample(out)
                
        return out, {
            'sigma_gaussian': sigma_g,
            'sigma_speckle': sigma_s,
            'order': '->'.join(order)
        }
