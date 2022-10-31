import torch
import os
import h5py
import time
import numpy as np
import json
import openslide
# from PIL import ImageDraw


"""
数patch数量
"""
pt_path = '/data1/zrx/SwinMIL_results_20x/Camelyon16_features/pt_files/'

slide_num = 0
patch_num = 0
for root, dirs, files in os.walk(pt_path):
    for f in files:
        feature = torch.load(os.path.join(root, f))
        slide_num += 1
        patch_num += len(feature)
        del feature

print('slide_num:', slide_num)
print('patch_num:', patch_num)
print('avg_patch_num:', patch_num/slide_num)

# with open('dataset_csv/tcga-nsclc/all_clusters.json', 'r') as f:
#     dic = json.load(f)
#     print(type(dic['TCGA-56-7579-01Z-00-DX1.627f65b9-ac66-4f71-a6f4-394338b647f0']))

# slide_path = '/data1/zrx/TCGA-NSCLC/TCGA-56-A4BY-01Z-00-DX1.DCA0D153-7DB7-44FB-A87A-20931D36856A.svs'
# slide = openslide.OpenSlide(slide_path)
# print(slide.level_dimensions)
# position = (30000,20000)
# position2 = (1250,1250)
# position3 = (1330,1330)
# thumbnail = slide.get_thumbnail(slide.level_dimensions[-1])
# # thumbnail_draw = ImageDraw.ImageDraw(thumbnail)
# # thumbnail_draw.rectangle((position2,position3), fill=None, outline='black', width=5)
# region = slide.read_region(position, 1, (256,256))
# region.save('patch5.png')
# thumbnail.save('thumbnail.png')
# slide.close()