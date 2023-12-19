import os
import torch
import timm
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from models.transformer.vision_transformer import CustomVisionTransformer

FIXED_DIRECTORY = '/usr/src/ai-lab/src'

# Vision Transfomerモデルの登録
@register_model
def vit_custom(in_channels:int=3, num_classes:int=10, emb_dim:int=384, num_patch_row:int=2, image_size:int=32, num_blocks:int=7, head:int=8, hidden_dim:int=384*4, dropout:float=0.)->CustomVisionTransformer:
    """
    引数:
        in_channels:int=3 入力画像のチャンネル数
        num_classes:int=10 画像分類のクラス数
        emb_dim:int=384 埋め込み後のベクトルの長さ
        num_patch_row:int=2: 1辺のパッチの数
        image_size:int=32 入力画像の1辺の大きさ。入力画像の高さと幅は同じであると仮定
        num_blocks:int=7 Encoder Blockの数
        head:int=8 ヘッドの数
        hidden_dim:int=384*4 Encoder BlockのMLPにおける中間層のベクトルの長さ
        dropout:float=0. ドロップアウト率
    """
    # model = timm.create_model('vit_custom', model_factory=CustomVisionTransformer, pretrained=False)
    model = CustomVisionTransformer(in_channels, num_classes, emb_dim, num_patch_row, image_size, num_blocks, head, hidden_dim, dropout)
    return model

@register_model
def vit_base_16_224(pretrained:bool=False, num_classes:int=1000, img_size:int=224, **kwargs):
    model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes, img_size=img_size)
    model.default_cfg = _cfg()
    return model

@register_model
def vit_tiny_16_224(pretrained:bool=False, num_classes:int=1000, img_size:int=224,  **kwargs):
    model = timm.create_model('vit_tiny_patch16_224', pretrained=pretrained, num_classes=num_classes, img_size=img_size)
    model.default_cfg = _cfg()
    return model

@register_model
def vit_small_16_224(pretrained=False, **kwargs):
    model = timm.create_model('vit_small_patch16_224', pretrained=pretrained)
    model.default_cfg = _cfg()
    return model

# BEiTモデルの登録
@register_model
def beit_base_16_224(num_classes:int=100, img_size:int=224, **kwargs):
    """BEiT(base-sized model, fine-tuned on ImageNet-22k)

    Args:
        pretrained (bool, optional): _description_. Defaults to False.
        num_classes (int, optional): _description_. Defaults to 100.
        img_size (int, optional): _description_. Defaults to 224.

    Returns:
        _type_: _description_
    """

    pretrained_path = 'pretrained/beit_base_patch16_224_pt22k_ft22k.pth'
    absolute_path = os.path.join(FIXED_DIRECTORY, pretrained_path)
    model = timm.create_model(
        'beit_base_patch16_224',
        pretrained=False,
        num_classes=num_classes,
        img_size=img_size
    )
    model.load_state_dict(torch.load(absolute_path))
    return model

@register_model
def beit2_base_16_224(pretrained=False, num_classes:int=100, img_size:int=224, **kwargs):
    """BEiT2()

    Args:
        pretrained (bool, optional): _description_. Defaults to False.
        num_classes (int, optional): _description_. Defaults to 100.
        img_size (int, optional): _description_. Defaults to 224.

    Returns:
        _type_: _description_
    """

    pretrained_path = 'pretrained/beit/beitv2_base_patch16_224_pt1k_ft21k.pth'
    absolute_path = os.path.join(FIXED_DIRECTORY, pretrained_path)

    model = timm.create_model(
        'beitv2_base_patch16_224',
        num_classes=num_classes,
        img_size=img_size
    )
    model.load_state_dict(torch.load(absolute_path))
    return model

# Swinモデル　
@register_model
def swin_base_4_224(pretrained=False, num_classes:int=100, img_size:int=224, **kwargs):
    model = timm.create_model(
        'swin_base_patch4_window7_224',
        pretrained=False,
        num_classes=num_classes,
    )
    return model

@register_model
def swin_tiny_4_224(pretrained=False, num_classes:int=100, img_size:int=224, **kwargs):
    model = timm.create_model(
        'swin_tiny_patch4_window7_224',
        pretrained=False,
        num_classes=num_classes,
    )
    return model

#CNNベース
#resnet resnet101
@register_model
def cnn_resnet101(pretrained=False, num_classes:int=100, **kwargs):
    model = timm.create_model('resnet101', pretrained=pretrained, num_classes=num_classes,)
    return model

# vgg19
@register_model
def cnn_vgg19(pretrained=False, num_classes:int=100, **kwargs):
    model = timm.create_model('vgg19', pretrained=pretrained, num_classes=num_classes)
    return model
