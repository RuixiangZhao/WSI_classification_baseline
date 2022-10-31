import openslide
import os
import json
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from models import build_model
from utils.utils import read_yaml

model_name = 'hi_trans4_test'
config_path = 'configs/camelyon16.yaml'
pt_path = '/data1/zrx/SwinMIL_results_10x/Camelyon16_features/pt_files/'
h5_path = '/data1/zrx/SwinMIL_results_10x/Camelyon16_features/h5_files/'
slide_path = '/data1/zrx/Camelyon16/'
cluster_path = 'dataset_csv/camelyon16/all_clusters_10x.json'
heatmap_save_path = os.path.join('heatmap_results/camelyon16', model_name)
# config_path = 'configs/tcga-rcc.yaml'
# pt_path = '/data/zrx/SwinMIL_results_10x/TCGA-RCC_features/pt_files/'
# h5_path = '/data/zrx/SwinMIL_results_10x/TCGA-RCC_features/h5_files/'
# slide_path = '/data/zrx/TCGA-RCC/'
# cluster_path = 'dataset_csv/tcga-rcc/all_clusters_10x.json'
# heatmap_save_path = os.path.join('heatmap_results/tcga-rcc', model_name)
level = -2

if not os.path.exists(heatmap_save_path):
    os.makedirs(heatmap_save_path)

def DrawAttentionMap(model, slide_id):
    model.eval()
    feature = torch.load(os.path.join(pt_path, slide_id+'.pt')).unsqueeze(0)
    position = np.array(h5py.File(os.path.join(h5_path, slide_id+'.h5'))['coords']) # [(width, hight)]
    slide = openslide.OpenSlide(os.path.join(slide_path, slide_id+'.tif'))
    with open(cluster_path, 'r') as f:
        cluster = json.load(f)[slide_id]
    
    with torch.no_grad():
        results_dict = model(feature, cluster=cluster)

    Y_prob = results_dict['Y_prob'].squeeze().numpy()
    Y_hat = results_dict['Y_hat'].squeeze().numpy()
    print('Y_hat:', Y_hat)
    A = results_dict['Attention'].squeeze().numpy()

    thumbnail = slide.get_thumbnail(slide.level_dimensions[level]) # (hight, width, channel)

    # 因为在倒数第二层上可视化，所以所有坐标都要放缩
    mag = slide.level_dimensions[0][0] / slide.level_dimensions[level][0]
    _W, _H = slide.level_dimensions[level]
    attention_map = np.zeros((_H, _W))
    target_position = (position / mag).astype(int)
    target_patch_size = int(1024 / mag)
    for i in range(len(target_position)):
        w, h = target_position[i]
        attention_map[h:h+target_patch_size, w:w+target_patch_size] = A[i]
    
    plt.figure(dpi=200)
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()
    plt.axis('off')
    plt.imshow(thumbnail, alpha=1)
    # normed_mask = attention_map / attention_map.max()
    # sns.heatmap(normed_mask, cmap='jet', annot=False, alpha=0.3)
    normed_mask = (attention_map / attention_map.max() * 200).astype(int)
    plt.imshow(normed_mask, alpha=0.3, cmap='jet')
    plt.savefig(os.path.join(heatmap_save_path, slide_id+'.png'), bbox_inches='tight', pad_inches=0)




# load model
cfg = read_yaml(config_path)
model = build_model(cfg)
ckpt_path = os.path.join(cfg.General.results_dir, "fold{}".format(cfg.Data.fold), 'checkpoint_'+model_name+'.pt')
ckpt = torch.load(ckpt_path)
model.load_state_dict(ckpt, strict=True)

# slide_id = 'TCGA-G6-A8L8-01Z-00-DX1.32E022BA-9959-4E2F-9D9D-AEEB5B7E7E9F'
slide_id = 'test_001'
DrawAttentionMap(model, slide_id)