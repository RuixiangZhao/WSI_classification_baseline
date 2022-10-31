import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import Nystromformer

class TransLayer(nn.Module):
    def __init__(self, dim=512, heads=8, depth=1):
        super(TransLayer, self).__init__()
        self.trans = Nystromformer(
            dim = dim,
            dim_head = dim//heads,
            heads = heads,
            depth = depth,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            attn_values_residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            attn_dropout = 0.1,
            ff_dropout = 0.1
        )

    def forward(self, x):
        x = self.trans(x)
        return x

class TransAtt(nn.Module):
    def __init__(self, num_classes, width, depth, heads):
        print('depth: {}, width: {}, heads: {}'.format(depth, width, heads))
        super(TransAtt, self).__init__()
        self.num_classes = num_classes
        self._fc1 = nn.Sequential(nn.Linear(1024, width), nn.ReLU())
        self.trans_layer = TransLayer(dim = width, heads = heads, depth = depth)
        self.attention = nn.Sequential(nn.Linear(width, width//4), nn.Tanh(), nn.Linear(width//4, 1))
        self._fc2 = nn.Linear(width, self.num_classes)

    def forward(self, x, **kwargs):        
        x = self._fc1(x) #[B, n, width]

        # Translayer
        y = self.trans_layer(x)

        # Attention pooling
        a = self.attention(y) # [B,n,1]
        a = a.transpose(1, 2) # [B,n,1] -> [B,1,n]
        a = F.softmax(a, dim=2) # [B,1,n]
        z = torch.bmm(a, y).squeeze(1) # [B,1,n]*[B,n,width]=[B,1,width] -> [B,width]

        # Predict
        logits = self._fc2(z) #[B, num_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'Attention': a}
        return results_dict