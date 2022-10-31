import random
import torch
import pandas as pd
from pathlib import Path
import torch.utils.data as data
import numpy as np

class WsiFeatDataset(data.Dataset):
    '''WsiFeatDataset
    Args:
        state (str): train or test stage
        data_dir (str): path to wsi feature data
        label_dir (str): path to label data
        nfolds (int): n-fold cross validation strategy. Default: 4
        fold (int): currently processing i-th fold. Value range: [0,n-1]
        data_shuffle (bool): shuffle strategy. Default: False
    '''
    def __init__(self, state=None, data_dir=None, label_dir=None, nfold=4, fold=0, data_shuffle=False, **kwargs):
        super(WsiFeatDataset, self).__init__()

        self.state = state
        self.nfolds = nfold
        self.fold = fold
        self.feature_dir = data_dir
        self.csv_dir = label_dir + f'fold{self.fold}.csv'
        self.slide_data = pd.read_csv(self.csv_dir, index_col=0)
        self.data_shuffle = data_shuffle

        # split dataset
        if state == 'train':
            self.data = self.slide_data.loc[:, 'train'].dropna()
            self.label = self.slide_data.loc[:, 'train_label'].dropna()
        if state == 'val':
            self.data = self.slide_data.loc[:, 'val'].dropna()
            self.label = self.slide_data.loc[:, 'val_label'].dropna()
        if state == 'test':
            self.data = self.slide_data.loc[:, 'test'].dropna()
            self.label = self.slide_data.loc[:, 'test_label'].dropna()
    
    def __len__(self):
        return len(self.data)

    # dataloader automatically call __getitem__()
    def __getitem__(self, idx):
        slide_id = self.data[idx]
        label = int(self.label[idx])
        full_path = Path(self.feature_dir) / f'{slide_id}.pt'
        features = torch.load(full_path)

        # shuffle
        if self.data_shuffle == True:
            index = [x for x in range(features.shape[0])]
            random.shuffle(index)
            features = features[index]

        return slide_id, features, label