## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881
import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange

from models.wavelet_utils import get_filter_tensors, DWT


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))

        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


# ConvolutionalGLU design follows TransNeXt 24'CVPR
class ConvolutionalGLU(nn.Module):
    def __init__(self, dim, bias=True, drop=0.):
        super(ConvolutionalGLU, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.act = nn.GELU()
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x, v):
        x = self.conv1(x)
        v = self.act(self.dwconv(self.conv2(v)))
        x = x * v
        x = self.drop(x)
        x = self.conv3(x)
        x = self.drop(x)
        return x


##########################################################################
## Wavelet-Aware Convolutional Gating Attention (WACGA)
class WaveletAwareConvGatingAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, wavelet='haar', dec_lo=None, dec_hi=None):
        super(WaveletAwareConvGatingAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # QKV projection
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # Wavelet filters (learnable or reused)
        if dec_lo is not None and dec_hi is not None:
            # Use externally provided learnable filters
            self.dec_lo = dec_lo
            self.dec_hi = dec_hi
        else:
            dec_lo, dec_hi, _, _ = get_filter_tensors(wavelet, flip=True)
            self.dec_lo = nn.Parameter(dec_lo, requires_grad=True)
            self.dec_hi = nn.Parameter(dec_hi, requires_grad=True)

        # Define DWT module
        self.dwt = DWT(self.dec_lo, self.dec_hi, wavelet=wavelet, level=1)

        # Subband-specific processing modules
        # 1. Low-frequency subband (ya): Regular convolution
        self.ya_proj = nn.Conv2d(dim, dim // 4, kernel_size=1, bias=bias)

        # 2. Horizontal subband (yh): Horizontal convolution kernel (1x3)
        self.yh_conv = nn.Conv2d(dim, dim // 4, kernel_size=(1, 3), padding=(0, 1), groups=dim // 4, bias=bias)

        # 3. Vertical subband (yv): Vertical convolution kernel (3x1)
        self.yv_conv = nn.Conv2d(dim, dim // 4, kernel_size=(3, 1), padding=(1, 0), groups=dim // 4, bias=bias)

        # 4. Diagonal subband (yd): Only activation function
        self.yd_act = nn.Tanh()
        self.yd_proj = nn.Conv2d(dim, dim // 4, kernel_size=1, bias=bias)

        # Subband fusion and attention generation
        self.subband_fusion = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias),
            nn.Sigmoid()
        )

        self.ll_conv = nn.Conv2d(dim // 4, dim // 4, kernel_size=3, stride=1, padding=1, groups=dim // 4, bias=bias)
        # Create directional convolution kernels
        self.horizontal_conv, self.vertical_conv, self.diagonal_conv = self.create_wave_conv(dim // 4)

    def create_conv_layer(self, kernel, dim):
        """Create convolution layer with fixed weights"""
        conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim, bias=False)
        conv.weight.data = kernel.repeat(dim, 1, 1, 1)
        # conv.weight.requires_grad = False  # Fixed weights
        return conv

    def create_wave_conv(self, dim):
        """Create convolution kernels matching subband characteristics"""

        # yh
        horizontal_kernel = torch.tensor([[1, 1, 1],
                                          [0, 0, 0],
                                          [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # yv
        vertical_kernel = torch.tensor([[1, 0, -1],
                                        [1, 0, -1],
                                        [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # yd
        diagonal_kernel = torch.tensor([[0, 1, 0],
                                        [1, -4, 1],
                                        [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        horizontal_conv = self.create_conv_layer(horizontal_kernel, dim)
        vertical_conv = self.create_conv_layer(vertical_kernel, dim)
        diagonal_conv = self.create_conv_layer(diagonal_kernel, dim)

        return horizontal_conv, vertical_conv, diagonal_conv

    def forward(self, x):
        b, c, h, w = x.shape

        ya, (yh, yv, yd) = self.dwt(x)  # [B, C, H/2, W/2]

        # ya
        ya_proc = self.ya_proj(ya)  # [B, C/4, H/2, W/2]
        ya_proc = self.ll_conv(ya_proc)

        # yh
        yh_proc = self.yh_conv(yh)  # [B, C/4, H/2, W/2]
        yh_proc = self.horizontal_conv(yh_proc)

        # yv
        yv_proc = self.yv_conv(yv)  # [B, C/4, H/2, W/2]
        yv_proc = self.vertical_conv(yv_proc)

        # yd
        yd_proc = self.yd_act(yd)  # [B, C, H/2, W/2]
        yd_proc = self.yd_proj(yd_proc)
        yd_proc = self.diagonal_conv(yd_proc)

        subbands_proc = torch.cat([ya_proc, yh_proc, yv_proc, yd_proc], dim=1)  # [B, C, H/2, W/2]

        wavelet_attention_map = self.subband_fusion(subbands_proc)
        wavelet_attention_map = F.interpolate(
            wavelet_attention_map,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False
        )


        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)

        out = out * wavelet_attention_map
        return out


##########################################################################
## Spatial-Enhanced Attention (SEA)
class SpatialEnhancedAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(SpatialEnhancedAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # Gate
        self.mlp = nn.Sequential(
            nn.Linear(dim // num_heads, dim // num_heads, bias=True),
            nn.GELU(),
        )

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))

        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        if hasattr(self, 'mlp'):
            out = out * self.mlp(v.transpose(-2, -1)).transpose(-2, -1)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

##########################################################################
class SpatialTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(SpatialTransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = SpatialEnhancedAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
class WaveletTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, dec_lo, dec_hi):
        super(WaveletTransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = WaveletAwareConvGatingAttention(
            dim=dim,
            num_heads=num_heads,
            bias=bias,
            dec_lo=dec_lo,
            dec_hi=dec_hi,
            wavelet='haar'
        )
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        # x [B, 2C, H/2, W/2]

        x_after_attention = x + self.attn(self.norm1(x))

        y = self.ffn(self.norm2(x_after_attention))  # [B, C, H/2, W/2] [B, dim/2, H/2, W/2]

        return y + x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class WaveletDownsample(nn.Module):
    def __init__(self, dim, dec_lo, dec_hi, wavelet='haar'):
        super().__init__()

        self.pixel_down = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)  # (dim//2)*4 = 2*dim
        )

        # 2D-DWT
        self.dec_lo = dec_lo
        self.dec_hi = dec_hi
        self.dwt = DWT(self.dec_lo, self.dec_hi, wavelet=wavelet, level=1)

        # Subband Attention Branch (SAB)
        self.subband_attn = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 4, kernel_size=3, stride=1, padding=1, groups=dim * 4, bias=True),
            nn.GroupNorm(dim * 4, dim * 4),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim * 2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()  # to [0,1]
        )

        # fusion
        self.DWConv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=True)
        self.PWConv = nn.Conv2d(dim * 2, dim * 2, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x_down = self.pixel_down(x)
        # 2D-DWT
        ya, (yh, yv, yd) = self.dwt(x)

        # Subband Attention Branch (SAB)
        concat_subbands = torch.cat([ya, yh, yv, yd], dim=1)  # [B, 4C, H/2, W/2]

        attn_weights = self.subband_attn(concat_subbands)  # (B, 2*dim, H/2, W/2)

        x_enhanced = x_down * attn_weights + x_down

        x_enhanced = self.PWConv(self.DWConv(x_enhanced))

        return x_enhanced


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
##---------- LWTformer -----------------------
class LWTformer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 enc_blk_num=[4, 6, 6],
                 middle_blk_num=8,
                 dec_blk_num=[6, 6, 4],
                 num_refinement_blocks=4,
                 enc_heads=[1, 2, 4],
                 middle_heads=8,
                 dec_heads=[4, 2, 1],
                 refinement_heads=4,
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='BiasFree',
                 wavelet='haar',
                 initialize_wavelet=True
                 ):

        super(LWTformer, self).__init__()

        # Initialize wavelet filters
        dec_lo, dec_hi, _, _ = get_filter_tensors(wavelet, flip=True)
        if initialize_wavelet:
            self.dec_lo = nn.Parameter(dec_lo, requires_grad=True)  # Globally shared low-frequency filter
            self.dec_hi = nn.Parameter(dec_hi, requires_grad=True)  # Globally shared high-frequency filter
        else:
            # Random initialization
            self.dec_lo = nn.Parameter(torch.rand_like(dec_lo) * 2 - 1, requires_grad=True)
            self.dec_hi = nn.Parameter(torch.rand_like(dec_hi) * 2 - 1, requires_grad=True)
        # -------------------------------------------------------------------------

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoders = nn.ModuleList()
        self.middle_blocks = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.reduce_chan = nn.ModuleList()

        chan = dim

        # Encoder
        for i, num in enumerate(enc_blk_num):
            if i == 0:
                self.encoders.append(
                    nn.Sequential(*[
                        SpatialTransformerBlock(dim=chan, num_heads=enc_heads[i],
                                                ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                                LayerNorm_type=LayerNorm_type)
                        for _ in range(num)
                    ])
                )
            else:
                # Use WaveletTransformerBlock
                self.encoders.append(
                    nn.Sequential(*[
                        WaveletTransformerBlock(
                            dim=chan,
                            num_heads=enc_heads[i],
                            ffn_expansion_factor=ffn_expansion_factor,
                            bias=bias,
                            LayerNorm_type=LayerNorm_type,
                            dec_lo=self.dec_lo,  # Global shared filters
                            dec_hi=self.dec_hi  # Global shared filters
                        )
                        for _ in range(num)
                    ])
                )

            # WaveletDownsample
            self.downs.append(WaveletDownsample(
                chan,
                dec_lo=self.dec_lo,  # Global shared filters
                dec_hi=self.dec_hi  # Global shared filters
            ))

            chan = chan * 2

        # Middle Block
        self.middle_blocks = nn.Sequential(*[
            WaveletTransformerBlock(
                dim=chan,
                num_heads=middle_heads,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                dec_lo=self.dec_lo,  # Global shared filters
                dec_hi=self.dec_hi  # Global shared filters
            )
            for _ in range(middle_blk_num)
        ])

        # Decoder
        for i, num in enumerate(dec_blk_num):
            self.ups.append(Upsample(chan))

            if i < len(dec_blk_num) - 1:
                self.reduce_chan.append(nn.Conv2d(int(chan), int(chan // 2), kernel_size=1, bias=bias))
                chan = chan // 2
            else:
                self.reduce_chan.append(nn.Identity())

            self.decoders.append(
                nn.Sequential(*[
                    SpatialTransformerBlock(dim=chan, num_heads=dec_heads[i], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type)
                    for _ in range(num)
                ])
            )

        # Refinement Block
        self.refinement = nn.Sequential(*[
            SpatialTransformerBlock(dim=chan, num_heads=refinement_heads, ffn_expansion_factor=ffn_expansion_factor,
                                    bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_refinement_blocks)])

        self.output = nn.Conv2d(chan, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        x = self.patch_embed(inp_img)
        encs = []

        # Encoder
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        # Middle Block
        x = self.middle_blocks(x)

        # Decoder
        for decoder, up, enc_skip, reduce_ch in zip(self.decoders, self.ups, encs[::-1], self.reduce_chan):
            x = up(x)
            x = torch.cat([x, enc_skip], dim=1)
            x = reduce_ch(x)
            x = decoder(x)

        # Refinement Block
        x = self.refinement(x)
        x = self.output(x) + inp_img

        return x


if __name__ == '__main__':
    # x = torch.rand(1, 3, 128, 128)
    # LWTformer-B dim=32,   LWTformer-L dim=32
    model = LWTformer(inp_channels=3, out_channels=3, dim=32,
                      enc_blk_num=[4, 6, 6], middle_blk_num=8, dec_blk_num=[6, 6, 4], num_refinement_blocks=4,
                      enc_heads=[1, 2, 4], middle_heads=8, dec_heads=[4, 2, 1], refinement_heads=4,
                      ffn_expansion_factor=2.66, bias=False, LayerNorm_type='BiasFree')

    # print(model)
    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(model, (3, 128, 128), as_strings=True,
                                             print_per_layer_stat=True, verbose=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))