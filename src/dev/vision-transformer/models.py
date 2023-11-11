import torch
import torch.nn as nn
from functools import partial

import timm
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

"""
転移学習に使用されるモデルを定義する。

注意: 'timm' ライブラリがインストールされていることを確認してください。
    インストールコマンド: pip install timm
"""

# Vision Transformerモデルの登録
@register_model
def vit_tiny_16_224(pretrained=False, **kwargs):
    model = timm.create_model('vit_tiny_patch16_224', pretrained=pretrained)
    model.default_cfg = _cfg()
    return model

@register_model
def vit_small_16_224(pretrained=False, **kwargs):
    model = timm.create_model('vit_small_patch16_224', pretrained=pretrained)
    model.default_cfg = _cfg()
    return model
