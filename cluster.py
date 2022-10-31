import os
from utils.utils import my_cluster
import h5py
import time
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# h5_path = '/data1/zrx/SwinMIL_results_20x/TCGA-NSCLC_features/h5_files/'
# save_path = 'dataset_csv/tcga-nsclc/'

# K = [2,4,8,16,32]
# all_clusters = {}
# for root, dirs, files in os.walk(h5_path):
#     for f in files:
#         f_name = '.'.join(f.split('.')[0:-1])
#         print(f'processing {f_name}')
#         x = np.array(h5py.File(os.path.join(root, f))['coords'])
#         all_clusters[f_name] = my_cluster(K=K, x=x)

# json_str = json.dumps(all_clusters)
# with open(os.path.join(save_path, 'all_clusters_20x.json'), 'w') as json_file:
#     json_file.write(json_str)



'''对一张图片聚类结果的可视化
'''
# h5_path = '/data1/zrx/SwinMIL_results_20x/TCGA-NSCLC_features/h5_files/TCGA-56-A4BY-01Z-00-DX1.DCA0D153-7DB7-44FB-A87A-20931D36856A.h5'
h5_path = '/data1/zrx/SwinMIL_results_10x/Camelyon16_features/h5_files/test_001.h5'

x = np.array(h5py.File(h5_path)['coords'])

plt.figure(figsize=(10,10))
ax = plt.gca()
ax.xaxis.set_ticks_position('top')
ax.invert_yaxis()
plt.axis('off')
plt.scatter(x[:,0], x[:,1], c='grey', s=4)
plt.savefig('test.png')

y = KMeans(n_clusters=8, random_state=2022).fit_predict(x).tolist()

plt.figure(figsize=(10,10))
ax = plt.gca()
ax.xaxis.set_ticks_position('top')
ax.invert_yaxis()
plt.axis('off')
cmap = plt.cm.get_cmap("rainbow")
plt.scatter(x[:,0], x[:,1], c=y, s=4, cmap=cmap)
plt.savefig('test2.png')

plt.figure(figsize=(10,10))
ax = plt.gca()
ax.xaxis.set_ticks_position('top')
ax.invert_yaxis()
plt.axis('off')
cmap = plt.cm.get_cmap("rainbow")
plt.scatter(x[:,0], x[:,1], c=y, s=4, cmap=cmap)
for i in range(len(y)):
    if y[i] == 5:
        plt.scatter(x[i,0], x[i,1], c='orange', s=30)
plt.savefig('test3.png')