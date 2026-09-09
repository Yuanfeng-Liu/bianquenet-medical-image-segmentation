import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=64, dilation_rates=(6, 12, 18)):
        super().__init__()

        self.branch1 = ConvBNReLU(in_channels, out_channels, kernel_size=1, padding=0)
        self.branch2 = ConvBNReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation_rates[0],
            dilation=dilation_rates[0],
        )
        self.branch3 = ConvBNReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation_rates[1],
            dilation=dilation_rates[1],
        )
        self.branch4 = ConvBNReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation_rates[2],
            dilation=dilation_rates[2],
        )
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNReLU(in_channels, out_channels, kernel_size=1, padding=0),
        )

        self.project = ConvBNReLU(out_channels * 5, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        size = x.shape[-2:]

        pooled = self.image_pool(x)
        pooled = F.interpolate(pooled, size=size, mode="bilinear", align_corners=False)

        x = torch.cat(
            [
                self.branch1(x),
                self.branch2(x),
                self.branch3(x),
                self.branch4(x),
                pooled,
            ],
            dim=1,
        )

        return self.project(x)


class PSP(nn.Module):
    def __init__(self, in_channels, pool_sizes=(1, 2, 3, 6)):
        super().__init__()

        branch_channels = in_channels // len(pool_sizes)
        self.out_channels = in_channels + branch_channels * len(pool_sizes)

        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_size),
                    ConvBNReLU(in_channels, branch_channels, kernel_size=1, padding=0),
                )
                for pool_size in pool_sizes
            ]
        )

    def forward(self, x):
        size = x.shape[-2:]
        features = [x]

        for branch in self.branches:
            pooled = branch(x)
            pooled = F.interpolate(pooled, size=size, mode="bilinear", align_corners=False)
            features.append(pooled)

        return torch.cat(features, dim=1)


class DFE(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=64,
        pool_sizes=(1, 2, 3, 6),
        dilation_rates=(6, 12, 18),
    ):
        super().__init__()

        self.psp = PSP(in_channels, pool_sizes=pool_sizes)
        self.aspp = ASPP(
            in_channels=self.psp.out_channels,
            out_channels=out_channels,
            dilation_rates=dilation_rates,
        )

    def forward(self, x):
        x = self.psp(x)
        x = self.aspp(x)
        return x


class MFFBlock(nn.Module):
    def __init__(self, low_channels, mid_channels, deep_channels, out_channels=64):
        super().__init__()

        self.low_project = ConvBNReLU(low_channels, out_channels, kernel_size=1, padding=0)
        self.mid_project = ConvBNReLU(mid_channels, out_channels, kernel_size=1, padding=0)
        self.deep_project = ConvBNReLU(deep_channels, out_channels, kernel_size=1, padding=0)

        self.fuse = nn.Sequential(
            ConvBNReLU(out_channels * 3, out_channels),
            ConvBNReLU(out_channels, out_channels),
        )

    def forward(self, low_feature, mid_feature, deep_feature):
        target_size = low_feature.shape[-2:]

        low_feature = self.low_project(low_feature)

        mid_feature = self.mid_project(mid_feature)
        mid_feature = F.interpolate(
            mid_feature,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        deep_feature = self.deep_project(deep_feature)
        deep_feature = F.interpolate(
            deep_feature,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat([low_feature, mid_feature, deep_feature], dim=1)
        return self.fuse(x)


class WindowSelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()

        assert channels % num_heads == 0

        self.channels = channels
        self.num_heads = num_heads
        self.head_channels = channels // num_heads
        self.scale = self.head_channels ** -0.5

        self.qkv = nn.Linear(channels, channels * 3)
        self.project = nn.Linear(channels, channels)

    def forward(self, x, mask=None):
        batch_windows, tokens, channels = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_windows, tokens, 3, self.num_heads, self.head_channels)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attention = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            num_windows = mask.shape[0]
            attention = attention.reshape(
                batch_windows // num_windows,
                num_windows,
                self.num_heads,
                tokens,
                tokens,
            )
            attention = attention + mask.unsqueeze(0).unsqueeze(2)
            attention = attention.reshape(-1, self.num_heads, tokens, tokens)

        attention = attention.softmax(dim=-1)
        x = attention @ v
        x = x.transpose(1, 2).reshape(batch_windows, tokens, channels)

        return self.project(x)


class SwinTransformerBlock(nn.Module):
    def __init__(self, channels, window_size=8, num_heads=4, shift_size=0, mlp_ratio=2.0):
        super().__init__()

        assert 0 <= shift_size < window_size

        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(channels)
        self.attention = WindowSelfAttention(channels, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(channels)

        hidden_channels = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, channels),
        )

    def window_partition(self, x):
        batch_size, height, width, channels = x.shape
        x = x.reshape(
            batch_size,
            height // self.window_size,
            self.window_size,
            width // self.window_size,
            self.window_size,
            channels,
        )
        windows = x.permute(0, 1, 3, 2, 4, 5)
        return windows.reshape(-1, self.window_size * self.window_size, channels)

    def window_reverse(self, windows, batch_size, height, width, channels):
        x = windows.reshape(
            batch_size,
            height // self.window_size,
            width // self.window_size,
            self.window_size,
            self.window_size,
            channels,
        )
        x = x.permute(0, 1, 3, 2, 4, 5)
        return x.reshape(batch_size, height, width, channels)

    def make_attention_mask(self, height, width, device):
        if self.shift_size == 0:
            return None

        image_mask = torch.zeros((1, height, width, 1), device=device)
        height_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        width_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )

        count = 0
        for height_slice in height_slices:
            for width_slice in width_slices:
                image_mask[:, height_slice, width_slice, :] = count
                count += 1

        mask_windows = self.window_partition(image_mask)
        mask_windows = mask_windows.reshape(-1, self.window_size * self.window_size)
        attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attention_mask = attention_mask.masked_fill(attention_mask != 0, -100.0)
        attention_mask = attention_mask.masked_fill(attention_mask == 0, 0.0)

        return attention_mask

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        shortcut = x

        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)

        pad_height = (self.window_size - height % self.window_size) % self.window_size
        pad_width = (self.window_size - width % self.window_size) % self.window_size
        if pad_height > 0 or pad_width > 0:
            x = F.pad(x, (0, 0, 0, pad_width, 0, pad_height))

        padded_height, padded_width = x.shape[1], x.shape[2]

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attention_mask = self.make_attention_mask(padded_height, padded_width, x.device)
        else:
            shifted_x = x
            attention_mask = None

        windows = self.window_partition(shifted_x)
        attention_windows = self.attention(windows, mask=attention_mask)

        shifted_x = self.window_reverse(
            attention_windows,
            batch_size,
            padded_height,
            padded_width,
            channels,
        )

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x[:, :height, :width, :]
        x = x.permute(0, 3, 1, 2)
        x = shortcut + x

        shortcut = x
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2)

        return shortcut + x


class STSCBlock(nn.Module):
    def __init__(self, channels, window_size=8, num_heads=4):
        super().__init__()

        self.layers = nn.Sequential(
            SwinTransformerBlock(
                channels=channels,
                window_size=window_size,
                num_heads=num_heads,
                shift_size=0,
            ),
            SwinTransformerBlock(
                channels=channels,
                window_size=window_size,
                num_heads=num_heads,
                shift_size=window_size // 2,
            ),
        )

    def forward(self, x):
        return self.layers(x)


class BianqueNetMini(nn.Module):
    def __init__(self, in_channels=1, num_classes=4, base_channels=32):
        super().__init__()

        self.enc1 = ConvBNReLU(in_channels, base_channels)
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBNReLU(base_channels, base_channels * 2),
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBNReLU(base_channels * 2, base_channels * 4),
        )

        self.dfe = DFE(in_channels=base_channels * 4, out_channels=base_channels * 4)
        self.decoder = ConvBNReLU(base_channels * 4, base_channels * 2)
        self.head = nn.Conv2d(base_channels * 2, num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[-2:]

        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)

        x = self.dfe(x)
        x = self.decoder(x)
        x = self.head(x)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)

        return x


class BianqueNetMiniMFF(nn.Module):
    def __init__(self, in_channels=1, num_classes=4, base_channels=32):
        super().__init__()

        self.enc1 = ConvBNReLU(in_channels, base_channels)
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBNReLU(base_channels, base_channels * 2),
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBNReLU(base_channels * 2, base_channels * 4),
        )

        self.dfe = DFE(in_channels=base_channels * 4, out_channels=base_channels * 4)
        self.mff = MFFBlock(
            low_channels=base_channels,
            mid_channels=base_channels * 2,
            deep_channels=base_channels * 4,
            out_channels=base_channels * 2,
        )
        self.head = nn.Conv2d(base_channels * 2, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        deep_feature = self.dfe(x3)
        fused_feature = self.mff(x1, x2, deep_feature)
        x = self.head(fused_feature)

        return x


class BianqueNetMiniSTSC(nn.Module):
    def __init__(self, in_channels=1, num_classes=4, base_channels=32):
        super().__init__()

        self.enc1 = ConvBNReLU(in_channels, base_channels)
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBNReLU(base_channels, base_channels * 2),
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBNReLU(base_channels * 2, base_channels * 4),
        )

        self.dfe = DFE(in_channels=base_channels * 4, out_channels=base_channels * 4)
        self.stsc = STSCBlock(channels=base_channels * 4, window_size=8, num_heads=4)
        self.mff = MFFBlock(
            low_channels=base_channels,
            mid_channels=base_channels * 2,
            deep_channels=base_channels * 4,
            out_channels=base_channels * 2,
        )
        self.head = nn.Conv2d(base_channels * 2, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        deep_feature = self.dfe(x3)
        deep_feature = self.stsc(deep_feature)
        fused_feature = self.mff(x1, x2, deep_feature)
        x = self.head(fused_feature)

        return x
