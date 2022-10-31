import numpy as np
import pandas as pd
import torch
import os
import json
from sklearn.metrics import roc_auc_score

from .utils import calculate_accuracy

model_name = 'ds_mil'

class EarlyStopping:
    '''EarlyStopping: early stops the training if validation doesn't improve after a given patience.
    validate() will call this
    Args:
        patience (int): How long to wait after last time validation improved. Default: 20
        stop_epoch (int): Earliest epoch possible for stopping
        verbose (bool): If True, prints a message for each validation improvement. Default: False
    '''
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.early_stop = False
        self.counter = 0
        self.best_score = None
        self.val_loss_min = np.Inf
        self.val_acc_max = 0
        self.val_auc_max = 0

    def __call__(self, epoch, val_loss, val_acc, val_auc, model, ckpt_name='checkpoint.pt'):
        score = -val_loss
        # score = val_acc
        # score = val_auc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, val_acc, val_auc, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch >= self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, val_acc, val_auc, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_acc, val_auc, model, ckpt_name):
        '''
        Save model when validation imporve
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            # print(f'Validation acc increased ({self.val_acc_max:.6f} --> {val_acc:.6f}).  Saving model ...')
            # print(f'Validation auc increased ({self.val_auc_max:.6f} --> {val_auc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss
        self.val_acc_max = val_acc
        self.val_auc_max = val_auc

def train(model, train_dataloader, val_dataloader, device, cur_fold, num_classes, loss_fn, optimizer, scheduler=None, max_epoch=200, cluster_path=None, results_dir=None):
    '''train for a single fold
    Args:
        model (nn.Module): model.
        train_dataset (data.DataLoader): training dataloader.
        val_dataset (data.DataLoader): validation dataloader.
        device (torch.device): cuda or cpu.
        cur_fold (int): currently processing fold.
        num_classes (int): number of classes.
        loss_fn (): loss function.
        optimizer (): optimizer.
        scheduler (): scheduler.
        max_epoch (int): max epoches for training. Default: 200
        results (str): dir to save model.pt and other eval results
    '''
    # load cluster json
    with open(cluster_path, 'r') as f:
        all_clusters = json.load(f)
    results_dir = os.path.join(results_dir, f'fold{cur_fold}')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    model.to(device)
    early_stopping = EarlyStopping(patience=10, stop_epoch=20, verbose=True) #!!
    
    all_epoch = []
    all_train_loss = []
    all_train_acc = []
    all_val_loss = []
    all_val_acc = []
    all_val_auc = []
    for epoch in range(max_epoch):
        train_loss, train_acc = train_one_epoch(epoch, model, train_dataloader, device, num_classes, loss_fn, optimizer, scheduler, all_clusters)
        val_loss, val_acc, val_auc, stop = validate(epoch, model, val_dataloader, device, cur_fold, num_classes, loss_fn, early_stopping, results_dir, all_clusters)
        all_epoch.append(epoch)
        all_train_loss.append(train_loss)
        all_train_acc.append(train_acc)
        all_val_loss.append(val_loss)
        all_val_acc.append(val_acc)
        all_val_auc.append(val_auc)
        if stop: # early stop
            pd.DataFrame({'epoch':all_epoch,'train_loss':all_train_loss,'train_acc':all_train_acc,
                        'val_loss':all_val_loss,'val_acc':all_val_acc,'val_auc':all_val_auc}).to_csv(os.path.join(results_dir,'training_logs_'+model_name+'.csv'))
            break

def train_one_epoch(epoch, model, train_dataloader, device, num_classes, loss_fn, optimizer, scheduler, all_clusters):
    '''train in one epoch, train() will call this
    Args:
        part of train()'s args
    '''
    model.train()
    train_loss = 0.
    train_acc = 0.
    for batch_idx, (slide_id, data, label) in enumerate(train_dataloader):
        data, label = data.to(device), label.to(device)
        results_dict = model(data, cluster=all_clusters[slide_id[0]]) # {'logits':, 'Y_prob':, 'Y_hat':}
        #! results_dict = model(x=data) # {'logits':, 'Y_prob':, 'Y_hat':}
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        loss = loss_fn(logits, label)
        train_loss += loss.item()

        acc = calculate_accuracy(Y_hat, label)
        train_acc += acc

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if scheduler != None:
            scheduler.step()

    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    print('\nEpoch: {}, train loss: {:.4f}, train accuracy: {:.4f}'.format(epoch, train_loss, train_acc))
    return train_loss, train_acc

def validate(epoch, model, val_dataloader, device, cur_fold, num_classes, loss_fn, early_stopping, results_dir, all_clusters):
    '''validate in one epoch, train() will call this
    Args:
        part of train()'s args
        early_stopping (EarlyStopping)
    '''
    model.eval()
    val_loss = 0.
    val_acc = 0.
    val_auc = 0.
    all_probs = [] # store predicted scores in all batches, will be used for calculate auc.
    all_labels = [] # store true labels in all batches, will be used for calculate auc.
    with torch.no_grad():
        for batch_idx, (slide_id, data, label) in enumerate(val_dataloader):
            # model forward
            data, label = data.to(device), label.to(device)
            results_dict = model(data, cluster=all_clusters[slide_id[0]]) # {'logits':, 'Y_prob':, 'Y_hat':}
            #! results_dict = model(x=data) # {'logits':, 'Y_prob':, 'Y_hat':}
            logits = results_dict['logits']
            Y_prob = results_dict['Y_prob']
            Y_hat = results_dict['Y_hat']

            # calculate loss
            loss = loss_fn(logits, label)
            val_loss += loss.item()

            # calculate accuracy
            acc = calculate_accuracy(Y_hat, label)
            val_acc += acc

            # update list
            all_probs.append(Y_prob.cpu().numpy())
            all_labels.append(label.cpu().numpy())
    # update raw evalation results after all batches
    val_acc /= len(val_dataloader)
    val_loss /= len(val_dataloader)
    all_probs = np.concatenate(all_probs, axis=0) # 2D, shape: [len(val_dataset), num_classes]
    all_labels = np.concatenate(all_labels, axis=0) # 1D, shape: [len(val_dataset)]
    
    if num_classes == 2:
        val_auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        val_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')

    print('\tVal, val loss: {:.4f}, val accuracy: {:.4f}, val auc: {:.4f}'.format(val_loss, val_acc, val_auc))

    early_stopping(epoch, val_loss, val_acc, val_auc, model, ckpt_name = os.path.join(results_dir, 'checkpoint_'+model_name+'.pt'))
    stop = False
    if early_stopping.early_stop:
        print("Early stopping")
        stop = True
    return val_loss, val_acc, val_auc, stop