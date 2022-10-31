import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention

class MeanPooling(nn.Module):
    def __init__(self, num_classes):
        super(MeanPooling, self).__init__()
        self.num_classes = num_classes
        self.mlp = nn.Sequential(nn.Linear(1024, 32), nn.ReLU(), nn.Linear(32,self.num_classes))

    def forward(self, x, **kwargs):
        logits = self.mlp(x.mean(1)) # [B,n,1024] -> [B,1024] -> [B,num_class]

        # predict
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict

class MaxPooling(nn.Module):
    def __init__(self, num_classes):
        super(MaxPooling, self).__init__()
        self.num_classes = num_classes
        self.mlp = nn.Sequential(nn.Linear(1024, 32), nn.ReLU(), nn.Linear(32,self.num_classes))

    def forward(self, x, **kwargs):
        logits = self.mlp(x.max(1).values) # [B,n,1024] -> [B,1024] -> [B,num_class]

        # predict
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict

class AttentionPooling(nn.Module):
    def __init__(self, num_classes):
        super(AttentionPooling, self).__init__()
        self.num_classes = num_classes
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.attention = nn.Sequential(nn.Linear(512,128), nn.Tanh(), nn.Linear(128,1))
        self._fc2 = nn.Linear(512, self.num_classes)

    def forward(self, x, **kwargs):
        x = self._fc1(x) # [B,n,1024] -> [B,n,512]

        # attention matrix
        a = self.attention(x) # [B,n,1]
        a = a.transpose(1, 2) # [B,n,1] -> [B,1,n]
        a = F.softmax(a, dim=2) # [B,1,n]
        
        # a * x
        x = torch.bmm(a, x).squeeze(1) # [B,1,n]*[B,n,512]=[B,1,512] -> [B,512]

        # predict
        logits = self._fc2(x) #[B, num_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'Attention': a}
        return results_dict

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super(TransLayer, self).__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x

class MSA_cls(nn.Module):
    def __init__(self, num_classes, num_layers):
        super(MSA_cls, self).__init__()
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.num_classes)
        for i_layer in range(self.num_layers):
            layer = TransLayer(dim=512)
            self.layers.append(layer)

    def forward(self, h, **kwargs):        
        h = self._fc1(h) #[B, n, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda(h.device)
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer
        for layer in self.layers:
            h = layer(h)

        #---->cls_token
        h = self.norm(h)[:,0]

        #---->predict
        logits = self._fc2(h) #[B, num_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict

class MSA_mean(nn.Module):
    def __init__(self, num_classes, num_layers):
        super(MSA_mean, self).__init__()
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self._fc2 = nn.Linear(512, self.num_classes)
        for i_layer in range(self.num_layers):
            layer = TransLayer(dim=512)
            self.layers.append(layer)

    def forward(self, h, **kwargs):        
        h = self._fc1(h) #[B, n, 512]

        #---->Translayer
        for layer in self.layers:
            h = layer(h)

        #---->meanpooling
        h = h.mean(1) # [B, 512]

        #---->predict
        logits = self._fc2(h) #[B, num_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict

class MSA_att(nn.Module):
    def __init__(self, num_classes, num_layers):
        super(MSA_att, self).__init__()
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.attention = nn.Sequential(nn.Linear(512,128), nn.Tanh(), nn.Linear(128,1))
        self._fc2 = nn.Linear(512, self.num_classes)
        for i_layer in range(self.num_layers):
            layer = TransLayer(dim=512)
            self.layers.append(layer)

    def forward(self, h, **kwargs):        
        h = self._fc1(h) #[B, n, 512]

        #---->Translayer
        for layer in self.layers:
            h = layer(h)

        #---->attention pooling
        a = self.attention(h) # [B,n,1]
        a = a.transpose(1, 2) # [B,n,1] -> [B,1,n]
        a = F.softmax(a, dim=2) # [B,1,n]
        h = torch.bmm(a, h).squeeze(1) # [B,1,n]*[B,n,512]=[B,1,512] -> [B,512]

        #---->predict
        logits = self._fc2(h) #[B, num_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class TransMIL_cls(nn.Module):
    def __init__(self, num_classes):
        super(TransMIL_cls, self).__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.num_classes = num_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.num_classes)


    def forward(self, h, **kwargs):
        h = self._fc1(h) #[B, n, 512]
        
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda(h.device)
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]
        
        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]

        #---->cls_token
        h = self.norm(h)[:,0]

        #---->predict
        logits = self._fc2(h) #[B, num_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict

class TransMIL_mean(nn.Module):
    def __init__(self, num_classes):
        super(TransMIL_mean, self).__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.num_classes = num_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self._fc2 = nn.Linear(512, self.num_classes)


    def forward(self, h):
        h = self._fc1(h) #[B, n, 512]
        
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N+1, 512]

        #---->PPEG
        h = self.pos_layer(h, _H, _W)[:,1:,:] #[B, N, 512]
        
        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]

        #---->mean pooling
        h = h.mean(1) # [B, 512]

        #---->predict
        logits = self._fc2(h) #[B, num_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict

class TransMIL_att(nn.Module):
    def __init__(self, num_classes):
        super(TransMIL_att, self).__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.num_classes = num_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.attention = nn.Sequential(nn.Linear(512,128), nn.Tanh(), nn.Linear(128,1))
        self._fc2 = nn.Linear(512, self.num_classes)

    def forward(self, h):
        h = self._fc1(h) #[B, n, 512]
        
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda(1)
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N+1, 512]

        #---->PPEG
        h = self.pos_layer(h, _H, _W)[:,1:,:] #[B, N, 512]
        
        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]

        #---->attention pooling
        a = self.attention(h) # [B,N,1]
        a = a.transpose(1, 2) # [B,N,1] -> [B,1,N]
        a = F.softmax(a, dim=2) # [B,1,N]
        h = torch.bmm(a, h).squeeze(1) # [B,1,N]*[B,N,512]=[B,1,512] -> [B,512]

        #---->predict
        logits = self._fc2(h) #[B, num_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict