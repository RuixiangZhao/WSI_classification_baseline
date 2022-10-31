from .swin_mil import SwinMIL
from .sandwich import Sandwich
from .trans_att import TransAtt
from .utrans_att import UTransAtt
from .ds_mil import DSMIL
from .hierarchical_transformer import HiTrans
from .baseline import *
import torch.nn as nn

def load_norm(norm_name):
    '''name(str) -> nn.module
    '''
    my_norm = None
    if norm_name == 'LayerNorm':
        my_norm = nn.LayerNorm
    else:
        raise NotImplementedError(f"Unkown norm: {norm_name}")
    return my_norm

def build_model(config):
    model_name = config.Model.name
    model = None
    if model_name == 'SwinMIL':
        my_norm = load_norm(config.Model.swin_transformer.norm_layer)
        model_dic = {'in_embed_dim': config.Model.in_embed_dim,
                    'sample_mode': config.Model.sample_mode,
                    'img_size': config.Model.swin_transformer.img_size,
                    'patch_size': config.Model.swin_transformer.patch_size,
                    'in_chans': config.Model.swin_transformer.in_chans,
                    'num_classes': config.Model.num_classes,
                    'embed_dim': config.Model.swin_transformer.embed_dim,
                    'depths': config.Model.swin_transformer.depths,
                    'num_heads': config.Model.swin_transformer.num_heads,
                    'window_size': config.Model.swin_transformer.window_size,
                    'mlp_ratio': config.Model.swin_transformer.mlp_ratio,
                    'qkv_bias': config.Model.swin_transformer.qkv_bias,
                    'qk_scale': config.Model.swin_transformer.qk_scale,
                    'drop_rate': config.Model.swin_transformer.drop_rate,
                    'attn_drop_rate': config.Model.swin_transformer.attn_drop_rate,
                    'drop_path_rate': config.Model.swin_transformer.drop_path_rate,
                    'norm_layer': my_norm,
                    'ape': config.Model.swin_transformer.ape,
                    'patch_norm': config.Model.swin_transformer.patch_norm,
                    'use_checkpoint': config.Model.swin_transformer.use_checkpoint}
        model = SwinMIL(**model_dic)
    if model_name == 'HiTrans':
        model = HiTrans(num_classes=config.Model.num_classes, num_windows=config.Model.hi_trans.num_windows,
                        dim=config.Model.hi_trans.dim)
    elif model_name == 'Sandwich':
        model = Sandwich(num_classes = config.Model.num_classes, num_layers=config.Model.num_layers)
    elif model_name == 'TransAtt':
        model = TransAtt(num_classes=config.Model.num_classes, width=config.Model.trans_att.width,
                        depth=config.Model.trans_att.depth, heads=config.Model.trans_att.heads)
    elif model_name == 'UTransAtt':
        model = UTransAtt(num_classes=config.Model.num_classes, width=config.Model.trans_att.width,
                        depth=config.Model.trans_att.depth, heads=config.Model.trans_att.heads)
    elif model_name == 'MeanPooling':
        model = MeanPooling(num_classes = config.Model.num_classes)
    elif model_name == 'MaxPooling':
        model = MaxPooling(num_classes = config.Model.num_classes)
    elif model_name == 'AttentionPooling':
        model = AttentionPooling(num_classes = config.Model.num_classes)
    elif model_name == 'MSA_cls':
        model = MSA_cls(num_classes = config.Model.num_classes, num_layers=config.Model.num_layers)
    elif model_name == 'MSA_mean':
        model = MSA_mean(num_classes = config.Model.num_classes, num_layers=config.Model.num_layers)
    elif model_name == 'MSA_att':
        model = MSA_att(num_classes = config.Model.num_classes, num_layers=config.Model.num_layers)
    elif model_name == 'TransMIL_cls':
        model = TransMIL_cls(num_classes = config.Model.num_classes)
    elif model_name == 'TransMIL_mean':
        model = TransMIL_mean(num_classes = config.Model.num_classes)
    elif model_name == 'TransMIL_att':
        model = TransMIL_att(num_classes = config.Model.num_classes)
    elif model_name == 'DSMIL':
        model = DSMIL(num_classes = config.Model.num_classes)
    else:
        raise NotImplementedError(f"Unkown model: {model_name}")
    return model