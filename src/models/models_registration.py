import timm
from models.transformer.vision_transformer import CustomVisionTransformer
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

# Vision Transfomerモデルの登録
@register_model
def vit_custom(pretrained=False, **kwargs):
    # model = timm.create_model('vit_custom', model_factory=CustomVisionTransformer, pretrained=False)
    model = CustomVisionTransformer(**kwargs)
    return model  

@register_model
def vit_tiny_16_224(pretrained:bool=False, num_classes:int=1000, **kwargs):
    model = timm.create_model('vit_tiny_patch16_224', pretrained=pretrained, num_classes=num_classes)
    model.default_cfg = _cfg()
    return model

@register_model
def vit_small_16_224(pretrained=False, **kwargs):
    model = timm.create_model('vit_small_patch16_224', pretrained=pretrained)
    model.default_cfg = _cfg()
    return model
