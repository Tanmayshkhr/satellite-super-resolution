import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias

class TransposedAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        q = F.normalize(q.view(B, self.num_heads, C // self.num_heads, H * W), dim=-1)
        k = F.normalize(k.view(B, self.num_heads, C // self.num_heads, H * W), dim=-1)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W)
        attn = (q @ k.transpose(-2, -1) * self.temperature).softmax(dim=-1)
        out = (attn @ v).reshape(B, C, H, W)
        return self.project_out(out)

class GatedFFN(nn.Module):
    def __init__(self, dim, expansion_factor=2.0):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1, groups=hidden_dim * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)

class RestormerBlock(nn.Module):
    def __init__(self, dim=64, num_heads=4):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = TransposedAttention(dim, num_heads)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GatedFFN(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class SatelliteRestormerSR(nn.Module):
    def __init__(self, in_channels=3, dim=64, num_blocks=6, num_heads=4, scale=4):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[RestormerBlock(dim, num_heads) for _ in range(num_blocks)])
        self.conv_body = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(dim, dim * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(dim, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        base = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        feat = self.conv_first(x)
        res = self.conv_body(self.body(feat)) + feat
        return base + self.upsample(res)

class CharbonnierFFTLoss(nn.Module):
    def __init__(self, eps=1e-3, fft_weight=0.1):
        super().__init__()
        self.eps = eps
        self.fft_weight = fft_weight

    def forward(self, pred, target):
        charb_loss = torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))
        pred_fft = torch.fft.rfft2(pred, norm='backward')
        target_fft = torch.fft.rfft2(target, norm='backward')
        fft_loss = F.l1_loss(torch.view_as_real(pred_fft), torch.view_as_real(target_fft))
        return charb_loss + self.fft_weight * fft_loss
