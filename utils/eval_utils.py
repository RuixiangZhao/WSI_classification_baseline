import numpy as np
import torch
import os
import json
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

from .utils import specifity_score

model_name = 'ds_mil'

def initiate_model(model, ckpt_path):
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt, strict=True)
    return model

def evalate(model, ckpt_path, test_dataloader, device, num_classes, cur_fold, cluster_path, results_dir):
    with open(cluster_path, 'r') as f:
        all_clusters = json.load(f)
    results_dir = os.path.join(results_dir, f'fold{cur_fold}')
    model = initiate_model(model, ckpt_path)
    model.to(device)
    
    test_acc, test_sen, test_spe, test_auc = summary(model, test_dataloader, device, num_classes, cur_fold, results_dir, all_clusters)
    print('test accuracy: ', test_acc)
    print('test sensitivity: ', test_sen)
    print('test specifity: ', test_spe)
    print('test auc: ', test_auc)

def summary(model, test_dataloader, device, num_classes, cur_fold, results_dir, all_clusters):
    model.eval()

    all_probs = [] # store predicted scores in all batches, will be used for calculate auc and other metrics.
    all_preds = [] # store predicted labels in all batches, will be used for metrics.
    all_labels = [] # store true labels in all batches, will be used for calculate auc.
    all_slide_ids = []

    for batch_idx, (slide_id, data, label) in enumerate(test_dataloader):
        data, label = data.to(device), label.to(device)
        with torch.no_grad():
            results_dict = model(data, cluster=all_clusters[slide_id[0]])
            #! results_dict = model(x=data)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        # update eval results
        all_probs.append(Y_prob.cpu().numpy())
        all_preds.append(Y_hat.cpu().numpy())
        all_labels.append(label.cpu().numpy())
        all_slide_ids.append(slide_id[0])

    # update raw evalation results after all batches
    all_probs = np.concatenate(all_probs, axis=0) # 2D, shape: [len(test_dataset), num_classes]
    all_preds = np.concatenate(all_preds, axis=0) # 1D, shape: [len(test_dataset)]
    all_labels = np.concatenate(all_labels, axis=0) # 1D, shape: [len(test_dataset)]
    # all_slide_ids = np.concatenate(all_slide_ids, axis=0) # 1D, shape: [len(test_dataset)]

    # save raw evalation results
    raw_results_dict = {'slide_id': all_slide_ids, 'true_label': all_labels, 'predict_label': all_preds}
    for c in range(num_classes):
        raw_results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    pd.DataFrame(raw_results_dict).to_csv(os.path.join(results_dir, 'raw_evalation_results_'+model_name+'.csv'))

    # calculate metrics from raw results
    test_acc = accuracy_score(all_labels, all_preds) # accuracy
    test_sen = None
    test_spe = None
    test_auc = None
    if num_classes == 2:
        test_sen = recall_score(all_labels, all_preds) # sensitivity
        test_spe = specifity_score(all_labels, all_preds) # specifity
        test_auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        test_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    metrics = {'accuracy':[test_acc], 'sensitivity':[test_sen], 'specifity':[test_spe], 'auc':[test_auc]}
    pd.DataFrame(metrics).to_csv(os.path.join(results_dir, 'evalation_metrics_'+model_name+'.csv'))

    return test_acc, test_sen, test_spe, test_auc