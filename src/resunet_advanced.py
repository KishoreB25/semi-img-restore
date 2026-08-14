import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvBlock(nn.Module):
    """Standard Convolutional Block (Residual if in_c == out_c for first conv output)"""
    def __init__(self, in_c, out_c, use_se=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(out_c)

    def forward(self, x):
        identity = self.conv1(x)
        out = self.relu(identity)
        out = self.conv2(out)
        if self.use_se:
            out = self.se(out)
        return identity + out

class DilatedBottleneck(nn.Module):
    """Multi-scale dilated bottleneck (parallel rates 1, 2, 4 + fusion)"""
    def __init__(self, in_c, out_c):
        super().__init__()
        # Parallel convolutions with different dilations
        self.conv_d1 = nn.Conv2d(in_c, out_c // 4, kernel_size=3, padding=1, dilation=1)
        self.conv_d2 = nn.Conv2d(in_c, out_c // 4, kernel_size=3, padding=2, dilation=2)
        self.conv_d4 = nn.Conv2d(in_c, out_c // 4, kernel_size=3, padding=4, dilation=4)
        
        # 1x1 conv to capture global/local features
        self.conv_1x1 = nn.Conv2d(in_c, out_c // 4, kernel_size=1)
        
        # Feature fusion
        self.fusion = nn.Conv2d(out_c, out_c, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        
        # Channel matching for residual
        self.match = nn.Conv2d(in_c, out_c, kernel_size=1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        d1 = self.conv_d1(x)
        d2 = self.conv_d2(x)
        d4 = self.conv_d4(x)
        c1 = self.conv_1x1(x)
        
        concat = torch.cat([d1, d2, d4, c1], dim=1)
        out = self.relu(self.fusion(concat))
        return self.match(x) + out

class AdvancedResUNet(nn.Module):
    def __init__(self, in_c=1, out_c=1, use_se=False, use_dilated_bottleneck=False):
        super().__init__()
        
        # Stem
        self.stem = nn.Conv2d(in_c, 32, kernel_size=3, padding=1)
        
        # Encoder
        self.enc1 = ConvBlock(32, 64, use_se=use_se)
        self.pool1 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        
        self.enc2 = ConvBlock(64, 128, use_se=use_se)
        self.pool2 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)
        
        # Bottleneck
        if use_dilated_bottleneck:
            self.bottleneck = DilatedBottleneck(128, 128)
        else:
            self.bottleneck = ConvBlock(128, 128, use_se=use_se)
        
        # Decoder
        self.up2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec2 = ConvBlock(192, 64, use_se=use_se) # 64 + 128 (e2)
        
        self.up1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.dec1 = ConvBlock(96, 32, use_se=use_se) # 32 + 64 (e1)
        
        # Final Feature Conv
        self.feat_conv = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        # PixelShuffle x2
        self.ps_conv = nn.Conv2d(32, 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        
        # Output Conv
        self.out_conv = nn.Conv2d(1, out_c, kernel_size=3, padding=1)

    def forward(self, x):
        x0 = self.stem(x)
        
        e1 = self.enc1(x0)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        b = self.bottleneck(p2)
        
        d2 = F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = self.up2(d2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = self.up1(d1)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.feat_conv(d1)
        out = self.ps_conv(out)
        out = self.pixel_shuffle(out)
        
        out = self.out_conv(out)
        return out
