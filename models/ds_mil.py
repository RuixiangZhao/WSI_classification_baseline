""" DSMIL
Implementation modified from: https://github.com/binli123/dsmil-wsi/blob/master/dsmil.py

Paper: `Dual-stream Multiple Instance Learning Network for Whole Slide Image Classification with Self-supervised Contrastive Learning`
        - https://arxiv.org/abs/2011.08939
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))
    def forward(self, feats):
        x = self.fc(feats)
        return feats, x

class IClassifier(nn.Module):
    def __init__(self, feature_size, output_class):
        super(IClassifier, self).__init__()       
        self.fc = nn.Linear(feature_size, output_class)        
        
    def forward(self, feats): # N x K
        c = self.fc(feats.view(feats.shape[0], -1)) # N x C
        return feats.view(feats.shape[0], -1), c

class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True): # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.lin = nn.Sequential(nn.Linear(input_size, input_size), nn.ReLU())
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.Tanh())
        else:
            self.lin = nn.Identity()
            self.q = nn.Linear(input_size, 128)
        self.v = nn.Sequential(
            nn.Dropout(dropout_v),
            nn.Linear(input_size, input_size)
        )
        
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)  
        
    def forward(self, feats, c): # N x K, N x C
        device = feats.device
        feats = self.lin(feats) # N x K
        V = self.v(feats) # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted
        
        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K 
        q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
                
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        C = self.fcc(B) # 1 x C x 1
        C = C.view(1, -1) # 1 x C
        return C, A, B 
    
class DSMIL(nn.Module):
    def __init__(self, num_classes):
        super(DSMIL, self).__init__()
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.i_classifier = IClassifier(512, num_classes)
        self.b_classifier = BClassifier(512, num_classes, 0.1, True)
        
    def forward(self, x, **kwargs): # B x N x K
        x = self._fc1(x).squeeze() # N x K, dimension reduction
        feats, ins_prediction = self.i_classifier(x) # ins_prediction shape: N x C
        max_prediction, _ = torch.max(ins_prediction, 0) # max_prediction shape: C
        bag_prediction, A, B = self.b_classifier(feats, ins_prediction) # bag_prediction shape: 1 x C
        
        # predict
        logits = 0.5*max_prediction.view(1,-1) + 0.5*bag_prediction.view(1,-1) # [1, num_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'Attention': A[:,1]}
        return results_dict