'''
模型有缺陷,放弃
'''

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

class UTransAtt(nn.Module):
    def __init__(self, num_classes, width, depth, heads):
        print('depth: {}, width: {}, heads: {}'.format(depth, width, heads))
        super(UTransAtt, self).__init__()
        self.num_classes = num_classes
        self._fc1 = nn.Sequential(nn.Linear(1024, width), nn.ReLU())
        self.trans_layer = TransLayer(dim = width, heads = heads, depth = depth)
        self.attention1 = nn.Sequential(nn.Linear(width, width//4), nn.Tanh(), nn.Linear(width//4, 1))
        self.attention2 = nn.Sequential(nn.Linear(width, width//4), nn.Tanh(), nn.Linear(width//4, 1))
        self._fc2 = nn.Linear(2*width, self.num_classes)

    def forward(self, x):        
        x = self._fc1(x) #[B, n, width]

        # Attention pooling for front features
        a1 = F.softmax(self.attention1(x).transpose(1, 2), dim=2) # [B,1,n]
        y1 = torch.bmm(a1, x).squeeze(1) # [B,1,n]*[B,n,width]=[B,1,width] -> [B,width]

        # Translayer
        x = self.trans_layer(x) # [B, n, width]

        # Attention pooling for back features
        a2 = F.softmax(self.attention2(x).transpose(1, 2), dim=2) # [B,1,n]
        y2 = torch.bmm(a2, x).squeeze(1) # [B,1,n]*[B,n,width]=[B,1,width] -> [B,width]

        # print('a1:', a1.cpu().numpy())
        # print('a2:', a2.cpu().numpy())
        # exit(0)
        '''
        结果证明模型训练时会走捷径,只走a1那条路,a2数组实际上全部是平均值,
        相当于transformer没起作用,整个模型相当于一个AttentionPooling
        '''

        # Predict
        logits = self._fc2(torch.cat((y1,y2), dim=1)) #[B, num_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict