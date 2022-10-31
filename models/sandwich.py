import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention

class TransLayer(nn.Module):
    def __init__(self, dim=512):
        super(TransLayer, self).__init__()
        self.trans = Nystromformer(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            depth = 1,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            attn_values_residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            attn_dropout = 0.1,
            ff_dropout = 0.1
        )

    def forward(self, x):
        x = self.trans(x)
        return x

class ConvLayer(nn.Module):
    def __init__(self, dim=512):
        super(ConvLayer, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = x + self.proj(x) + self.proj1(x) + self.proj2(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Slice(nn.Module):
    def __init__(self, dim=512):
        super(Slice, self).__init__()
        self.trans_layer = TransLayer(dim=dim)
        self.conv_layer = ConvLayer(dim=dim)

    def forward(self, x, H, W):
        x = self.trans_layer(x)
        x = self.conv_layer(x, H, W)
        return x

class Sandwich(nn.Module):
    def __init__(self, num_classes, num_layers):
        super(Sandwich, self).__init__()
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.last_layer = TransLayer(dim=512)
        self.attention = nn.Sequential(nn.Linear(512,128), nn.Tanh(), nn.Linear(128,1))
        self._fc2 = nn.Linear(512, self.num_classes)
        for i_layer in range(self.num_layers):
            layer = Slice(dim=512)
            self.layers.append(layer)

    def forward(self, x):
        x = self._fc1(x) #[B, n, 512]
        
        # squaring
        n = x.shape[1]
        _H, _W = int(np.ceil(np.sqrt(n))), int(np.ceil(np.sqrt(n)))
        add_length = _H * _W - n
        x = torch.cat([x, x[:,:add_length,:]],dim = 1) #[B, N, 512]

        # slice
        for layer in self.layers:
            x = layer(x, _H, _W)
        x = self.last_layer(x) #[B, N, 512]

        # attention
        a = F.softmax(self.attention(x).transpose(1, 2), dim=2) # [B, 1, N]
        x = torch.bmm(a, x).squeeze(1) # [B,512]

        #---->predict
        logits = self._fc2(x) # [B, num_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict
