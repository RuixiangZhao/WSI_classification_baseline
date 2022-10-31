import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import Nystromformer
from .ds_mil import DSMIL

class TransLayer(nn.Module):
    '''Base transformer layer
    '''
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
            ff_dropout = 0.1,          
        )

    def forward(self, x):
        x = self.trans(x)
        return x

class MultiWinTrans(nn.Module):
    '''Multi-window transformer
    Args:
        num_windows: number of windows in the transformer layer. 1,2,4,8...
    '''
    def __init__(self, num_windows, dim):
        super(MultiWinTrans, self).__init__()
        self.num_windows = num_windows # equal with cluster's k
        self.multi_win_trans_layer = TransLayer(dim = dim)

    def forward(self, x, cluster):
        # cluster (list)
        y = torch.zeros_like(x).to(x.device)
        for i_cluster in range(self.num_windows):
            y[:,cluster==i_cluster,:] = self.multi_win_trans_layer(x[:,cluster==i_cluster,:])
        return y


class HiTrans(nn.Module):
    '''Hierarchical transformer
    '''
    def __init__(self, num_classes, num_windows, dim):
        super(HiTrans, self).__init__()
        self._fc1 = nn.Sequential(nn.Linear(1024, dim), nn.ReLU())
        self.hi_trans_layers = nn.ModuleList()
        self.attention = nn.Sequential(nn.Linear(dim, dim//4), nn.Tanh(), nn.Linear(dim//4, 1))
        self._fc2 = nn.Linear(dim, num_classes)
        for n_w in num_windows:
            trans_layer = MultiWinTrans(num_windows = n_w, dim = dim)
            self.hi_trans_layers.append(trans_layer)
        
    def forward(self, x, cluster):
        '''
            cluster (dic) : {'2': [], '4': [], '8': []}
        '''
        # Dimension Reduction
        x = self._fc1(x) # [B, n, 512]

        # Transformer
        for trans_layer in self.hi_trans_layers:
            if trans_layer.num_windows == 1:
                y = trans_layer(x, np.zeros(x.shape[1], dtype=int))
            elif trans_layer.num_windows > 1:
                y = trans_layer(x, np.array(cluster[str(trans_layer.num_windows)], dtype=int))
            else:
                raise ValueError

        # Attention pooling
        a = F.softmax(self.attention(y).transpose(1, 2), dim = 2) # [B, 1, n]
        # z = torch.bmm(a, y).squeeze(1) # [B,1,n]*[B,n,512]=[B,1,512] -> [B,512]
        z = torch.bmm(a, x).squeeze(1) # [B,1,n]*[B,n,512]=[B,1,512] -> [B,512]

        # Predict
        logits = self._fc2(z) #[B, num_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'Attention': a}

        return results_dict
        