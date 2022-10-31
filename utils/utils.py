import torch.optim as optim
import torch.nn as nn
import torch
from sklearn.metrics import confusion_matrix
import yaml
from addict import Dict
from sklearn.cluster import KMeans

def build_optimizer(model, config):
    opt_lower = config.Optimizer.opt.lower()
    optimizer = None
    if opt_lower == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.Optimizer.lr, weight_decay=config.Optimizer.weight_decay)
    elif opt_lower == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config.Optimizer.lr, momentum=config.Optimizer.momentum, weight_decay=config.Optimizer.weight_decay)
    else:
        raise NotImplementedError(f"Unkown optimizer: {config.Optimizer.opt}")
    return optimizer

def bulid_scheduler(optimizer, config):
    is_use = config.Scheduler.is_use
    scheduler = None
    if is_use:
        sch_lower = config.Scheduler.sch.lower()
        if sch_lower == "multisteplr":
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.Scheduler.milestones, gamma=config.Scheduler.lr_decay)
        else:
            raise NotImplementedError(f"Unkown scheduler: {config.Scheduler.sch}")
    return scheduler

def build_loss_fn(config):
    loss_fn_lower = config.Loss.loss.lower()
    loss_fn = None
    if loss_fn_lower == 'crossentropyloss':
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f"Unkown loss function: {config.Loss.loss}")
    return loss_fn

def read_yaml(fpath=None):
    '''
    load config file(.yaml)
    '''
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)

def calculate_accuracy(Y_hat, Y):
    '''
    Y_hat (Tensor): predicted label, shape: [batch_size, 1]
    Y (Tensor): true label, shape: [batch_size, 1]
    '''
    acc = Y_hat.float().eq(Y.float()).float().mean().item()
    return acc

def specifity_score(gt, pred):
    '''specifity
    Args:
        gt (ndarray): ground truth.
        pred (ndarray): predict label.
    '''
    confusion = confusion_matrix(gt,pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    return TN / float(TN+FP)

def my_cluster(K = None, x = None):
    '''
    Args:
        K (list)
        x (np.array)
    '''
    clus = {}
    for k in K:
        clus[k] = KMeans(n_clusters=k, random_state=2022).fit_predict(x).tolist()
    return clus