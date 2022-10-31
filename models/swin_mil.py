import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .swin_transformer import SwinTransformer

class SS(nn.Module):
    '''SquaringSampler: [B,n,512]->[B,N,512]->[B,_H,_W,512]->[B,512,_H,_W]->[B,512,224,224]
    Down/up samples the input to the given size or the given scale_factor.
    The algorithm used for interpolation is determined by mode.
    Args:
        input (Tensor): the input tensor.
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]): output spatial size.
        mode (str): algorithm used for upsampling: 'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area' | 'nearest-exact'. Default: 'nearest'
    '''
    def __init__(self, size=224, mode=None):
        super(SS, self).__init__()
        self.size = size
        self.mode = mode

    def forward(self, x):
        B, n, C = x.shape
        _H, _W = int(np.ceil(np.sqrt(n))), int(np.ceil(np.sqrt(n)))
        add_length = _H * _W - n # N = _H * _W
        x = torch.cat([x, x[:,:add_length,:]],dim = 1) # [B,n,512] -> [B,N,512]
        x = x.view(B,_H,_W,C) # [B,N,512] -> [B,_H,_W,512]
        x = x.transpose(1,2).transpose(1,3) # [B,_H,_W,512] -> [B,512,_H,_W]
        x = F.interpolate(x, size=self.size, mode=self.mode) # [B,512,_H,_W] -> [B,512,224,224]
        # x = x.transpose(1,2).transpose(2,3) # [B,512,224,224] -> [B,224,224,512]
        return x


class SwinMIL(nn.Module):
    '''SwinMIL
    Args:
        input_embed_dim (int) : Input feature dimension. Default: 512
        mode (str): algorithm used for upsampling: 'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area' | 'nearest-exact'. Default: 'nearest'
        *kwargs: (for SwinTransformer block)
            img_size (int | tuple(int)): Input feature image size. Default: 224
            patch_size (int | tuple(int)): Patch size. Default: 4
            in_chans (int): Number of input image channels, keep the same with 'input_embed_dim'. Default: 512
            num_classes (int): Number of classes for classification head. Default: 2 or 3
            embed_dim (int): Patch embedding dimension. Default: 96
            depths (tuple(int)): Depth of each Swin Transformer layer.
            num_heads (tuple(int)): Number of attention heads in different layers.
            window_size (int): Window size. Default: 7
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
            qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
            qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
            drop_rate (float): Dropout rate. Default: 0
            attn_drop_rate (float): Attention dropout rate. Default: 0
            drop_path_rate (float): Stochastic depth rate. Default: 0.1
            norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
            ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
            patch_norm (bool): If True, add normalization after patch embedding. Default: True
            use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    '''
    def __init__(self, in_embed_dim=512, sample_mode=None, **kwargs):
        super(SwinMIL, self).__init__()
        self._fc1 = nn.Sequential(nn.Linear(1024, in_embed_dim), nn.ReLU())
        self.sample_layer = SS(size=kwargs['img_size'], mode=sample_mode)
        self.swin = SwinTransformer(img_size = kwargs['img_size'],
                                    patch_size = kwargs['patch_size'],
                                    in_chans = kwargs['in_chans'],
                                    num_classes = kwargs['num_classes'],
                                    embed_dim = kwargs['embed_dim'],
                                    depths = kwargs['depths'],
                                    num_heads = kwargs['num_heads'],
                                    window_size = kwargs['window_size'],
                                    mlp_ratio = kwargs['mlp_ratio'],
                                    qkv_bias = kwargs['qkv_bias'],
                                    qk_scale = kwargs['qk_scale'],
                                    drop_rate = kwargs['drop_rate'],
                                    attn_drop_rate = kwargs['attn_drop_rate'],
                                    drop_path_rate = kwargs['drop_path_rate'],
                                    norm_layer = kwargs['norm_layer'],
                                    ape = kwargs['ape'],
                                    patch_norm = kwargs['patch_norm'],
                                    use_checkpoint = kwargs['use_checkpoint'])
        # self.mlp = nn.Sequential(nn.Linear(1024, 32), nn.ReLU(), nn.Linear(32,2))
    
    def forward(self, x):
        # Dimension reduction
        x = self._fc1(x) # [B, n, 1024] -> [B, n , 512]

        # Squaring sampling sequence to (224, 224)
        x = self.sample_layer(x) # [B, n, 512] -> [B,512,224,224]

        # Swin transformer block
        logits = self.swin(x) # [B, C, H, W] -> [B, num_classes]

        # predict
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict