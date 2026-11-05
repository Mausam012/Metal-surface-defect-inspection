"""Educational: a ResNet 'BasicBlock' (used in ResNet-18/34) in ~30 lines.

Run:  python models/resnet/residual_block_demo.py      (needs PyTorch)

           x ──────────────┐   (shortcut / skip connection)
           │               │
        conv3x3 -> BN -> ReLU
           │               │
        conv3x3 -> BN      │
           │               │
           F(x)   ──(+)────┘      y = ReLU( F(x) + x )
"""
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.F = nn.Sequential(                                   # the residual function F(x)
            nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False), nn.BatchNorm2d(out_ch))
        # If the shape changes, a 1x1 conv makes x the same shape as F(x); otherwise identity.
        self.shortcut = nn.Identity() if stride == 1 and in_ch == out_ch else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch))

    def forward(self, x):
        return torch.relu(self.F(x) + self.shortcut(x))           # y = F(x) + x


if __name__ == "__main__":
    block = BasicBlock(64, 128, stride=2)
    print(block(torch.randn(1, 64, 56, 56)).shape)                # -> torch.Size([1, 128, 28, 28])
