import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)

    def forward(self, x):
        identity = self.conv1(x) # Handle channel matching if in_c != out_c by just taking first conv as base
        out = self.relu(identity)
        out = self.conv2(out)
        return identity + out

class ResUNet(nn.Module):
    def __init__(self, in_c=1, out_c=1):
        super().__init__()
        
        # Stem
        self.stem = nn.Conv2d(in_c, 32, kernel_size=3, padding=1)
        
        # Encoder
        self.enc1 = ConvBlock(32, 64)
        self.pool1 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1) # Downsample
        
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1) # Downsample
        
        # Bottleneck
        self.bottleneck = ConvBlock(128, 128)
        
        # Decoder
        # Use bilinear interpolation + conv instead of transpose conv for fewer checkerboard artifacts
        self.up2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec2 = ConvBlock(192, 64) # 64 from up2 + 128 from skip e2
        
        self.up1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.dec1 = ConvBlock(96, 32) # 32 from up1 + 64 from skip enc1
        
        # Final Feature Conv
        self.feat_conv = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        # PixelShuffle x2 upsampling (128x128 -> 256x256)
        self.ps_conv = nn.Conv2d(32, 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        
        # Output Conv
        self.out_conv = nn.Conv2d(1, out_c, kernel_size=3, padding=1)

    def forward(self, x):
        # x is 1x128x128
        x0 = self.stem(x) # 32x128x128
        
        # Encoder
        e1 = self.enc1(x0) # 64x128x128
        p1 = self.pool1(e1) # 64x64x64
        
        e2 = self.enc2(p1) # 128x64x64
        p2 = self.pool2(e2) # 128x32x32
        
        # Bottleneck
        b = self.bottleneck(p2) # 128x32x32
        
        # Decoder 2
        d2 = F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=False) # 128x64x64
        d2 = self.up2(d2) # 64x64x64
        d2 = torch.cat([d2, e2], dim=1) # 128x64x64
        d2 = self.dec2(d2) # 64x64x64
        
        # Decoder 1
        d1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False) # 64x128x128
        d1 = self.up1(d1) # 32x128x128
        d1 = torch.cat([d1, e1], dim=1) # 96x128x128
        d1 = self.dec1(d1) # 32x128x128
        
        # Refinement and Upsample
        out = self.feat_conv(d1) # 32x128x128
        out = self.ps_conv(out) # 4x128x128
        out = self.pixel_shuffle(out) # 1x256x256
        
        out = self.out_conv(out) # 1x256x256
        return out
