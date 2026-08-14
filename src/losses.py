import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y) ** 2 + self.eps ** 2))

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.tensor([math.exp(-(x - window_size//2)**2 / (2 * sigma**2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        return self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        # We return (1 - SSIM) as the loss component
        if size_average:
            return 1.0 - ssim_map.mean()
        else:
            return 1.0 - ssim_map.mean(1).mean(1).mean(1)

class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dx_kernel = torch.tensor([[[[-1, 0, 1]]]], dtype=torch.float32)
        self.dy_kernel = torch.tensor([[[[-1], [0], [1]]]], dtype=torch.float32)

    def forward(self, x, y):
        dx_kernel = self.dx_kernel.to(x.device)
        dy_kernel = self.dy_kernel.to(x.device)
        
        grad_x_pred = F.conv2d(x, dx_kernel, padding=(0, 1))
        grad_y_pred = F.conv2d(x, dy_kernel, padding=(1, 0))
        
        grad_x_gt = F.conv2d(y, dx_kernel, padding=(0, 1))
        grad_y_gt = F.conv2d(y, dy_kernel, padding=(1, 0))
        
        loss_x = torch.mean(torch.abs(grad_x_pred - grad_x_gt))
        loss_y = torch.mean(torch.abs(grad_y_pred - grad_y_gt))
        
        return loss_x + loss_y

class CombinedLoss(nn.Module):
    def __init__(self, loss_type='L1', char_eps=1e-3, lambda_char=0.8, lambda_ssim=0.2, lambda_grad=0.1):
        super().__init__()
        self.loss_type = loss_type
        self.char_eps = char_eps
        self.lambda_char = lambda_char
        self.lambda_ssim = lambda_ssim
        self.lambda_grad = lambda_grad
        
        self.l1 = nn.L1Loss()
        self.char = CharbonnierLoss(eps=char_eps)
        self.ssim = SSIMLoss()
        self.grad = GradientLoss()

    def forward(self, pred, gt):
        if self.loss_type == 'L1':
            return self.l1(pred, gt)
        elif self.loss_type == 'Charbonnier':
            return self.char(pred, gt)
        elif self.loss_type == 'Char_Grad':
            l_char = self.char(pred, gt)
            l_grad = self.grad(pred, gt)
            return l_char + self.lambda_grad * l_grad
        elif self.loss_type == 'Char_SSIM':
            l_char = self.char(pred, gt)
            l_ssim = self.ssim(pred, gt)
            return self.lambda_char * l_char + self.lambda_ssim * l_ssim
        elif self.loss_type == 'Char_SSIM_Grad':
            l_char = self.char(pred, gt)
            l_ssim = self.ssim(pred, gt)
            l_grad = self.grad(pred, gt)
            return self.lambda_char * l_char + self.lambda_ssim * l_ssim + self.lambda_grad * l_grad
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
