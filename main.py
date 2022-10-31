from utils.utils import *
from models import build_model
from datasets import build_dataset
from utils.core_utils import train
from utils.eval_utils import evalate

import argparse
import torch
from torch.utils.data import DataLoader
import time
import os
import numpy as np

model_name = 'ds_mil'

def make_parse():
    parser = argparse.ArgumentParser(description='Configurations for SwinMIL')
    parser.add_argument('--stage', type=str, choices=['train', 'test'])
    parser.add_argument('--config', type=str, help='path to config file')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for training and testing')
    parser.add_argument('--fold', type=int, default=0, help='currently processing i-th fold')
    args = parser.parse_args()
    return args

def main(cfg):
    # set seed
    # np.random.seed(cfg.General.seed)
    # torch.manual_seed(cfg.General.seed)
    # torch.cuda.manual_seed_all(cfg.General.seed)

    # set device
    device = None
    if not torch.cuda.is_available():
        raise Exception(f"cuda is not available, stop because cpu is too slow!")
    else:
        device = torch.device(f"cuda:{cfg.General.local_rank}")

    # build model
    print(f'########## Build Model: {cfg.Model.name} ##########')
    model = build_model(cfg)

    # build dataset
    print(f'########## Build Dataset: {cfg.Data.dataset_name} ##########')
    train_dataset, val_dataset, test_dataset = build_dataset(cfg)

    # build dataloader
    print(f'########## Build Dataloader ##########')
    if cfg.General.server == 'train':
        train_dataloader = DataLoader(train_dataset, batch_size=cfg.Data.train_dataloader.batch_size, num_workers=cfg.Data.train_dataloader.num_workers, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=cfg.Data.test_dataloader.batch_size, num_workers=cfg.Data.test_dataloader.num_workers, shuffle=False)
    else:
        test_dataloader = DataLoader(test_dataset, batch_size=cfg.Data.test_dataloader.batch_size, num_workers=cfg.Data.test_dataloader.num_workers, shuffle=False)

    # train or test
    if cfg.General.server == 'train':
        print('########## Start Training ##########')
        loss_fn = build_loss_fn(cfg)
        optimizer = build_optimizer(model, cfg)
        scheduler = bulid_scheduler(optimizer, cfg)
        train_dic= {'model': model,
                    'train_dataloader': train_dataloader,
                    'val_dataloader': val_dataloader,
                    'device': device,
                    'cur_fold': cfg.Data.fold,
                    'num_classes': cfg.Model.num_classes,
                    'loss_fn': loss_fn,
                    'optimizer': optimizer,
                    'scheduler': scheduler,
                    'max_epoch': cfg.General.max_epoch,
                    'cluster_path': cfg.Data.cluster_path,
                    'results_dir': cfg.General.results_dir}
        train(**train_dic)
    else:
        print('########## Start Evalation ##########')
        test_dic = {'model': model,
                    'ckpt_path': os.path.join(cfg.General.results_dir, "fold{}".format(cfg.Data.fold), 'checkpoint_'+model_name+'.pt'),
                    'test_dataloader': test_dataloader,
                    'device': device,
                    'num_classes': cfg.Model.num_classes,
                    'cur_fold': cfg.Data.fold,
                    'cluster_path': cfg.Data.cluster_path,
                    'results_dir': cfg.General.results_dir}
        evalate(**test_dic)
        

if __name__ == '__main__':
    args = make_parse()
    cfg = read_yaml(args.config)

    # update cfg
    cfg.config = args.config
    cfg.General.local_rank = args.local_rank
    cfg.General.server = args.stage
    cfg.Data.fold = args.fold

    # main
    main(cfg)