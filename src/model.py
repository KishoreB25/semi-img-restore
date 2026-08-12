import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return identity + out

class TinyResNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, hidden_channels=32, num_blocks=4):
        super().__init__()
        
        # Initial feature extraction
        self.conv_in = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        
        # Residual blocks
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(hidden_channels) for _ in range(num_blocks)
        ])
        
        # Post-residual convolution
        self.conv_post = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        
        # Upsampling (PixelShuffle)
        # To double resolution, we need 4 times the channels: 1 * (2**2)
        # We project from hidden_channels to 4 before pixel shuffle
        self.conv_up = nn.Conv2d(hidden_channels, 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        
        # Final output convolution
        self.conv_out = nn.Conv2d(1, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # x is Bx1x128x128
        out = self.conv_in(x)
        
        res = self.res_blocks(out)
        out = out + self.conv_post(res)
        
        out = self.conv_up(out)
        out = self.pixel_shuffle(out)
        
        out = self.conv_out(out)
        return out
