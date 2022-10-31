from .wsi_feat_dataset import WsiFeatDataset

def build_dataset(config):
    dataset_name = config.Data.dataset_name
    train_dataset = None
    val_dataset = None
    test_dataset = None
    if dataset_name in ['camelyon16', 'tcga-nsclc', 'tcga-rcc']:
        stage = config.General.server # train or test
        data_dic = {'data_dir': config.Data.data_dir,
                    'label_dir': config.Data.label_dir,
                    'nfold': config.Data.nfold,
                    'fold': config.Data.fold,
                    'data_shuffle': config.Data.data_shuffle}
        if stage == 'train':
            train_dataset = WsiFeatDataset(state='train',**data_dic)
            val_dataset = WsiFeatDataset(state='val', **data_dic)
        elif stage == 'test':
            test_dataset = WsiFeatDataset(state='test', **data_dic)
        else:
            raise NotImplementedError(f"Unkown stage: {stage}")
    else:
        raise NotImplementedError(f"Unkown dataset: {dataset_name}")

    return train_dataset, val_dataset, test_dataset