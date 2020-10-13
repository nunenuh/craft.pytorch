" base idea from https://github.com/mkisantal/backboned-unet"

import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.nn import functional as F
from typing import *


model_name = {
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnet152': models.resnet152,
    'vgg16_bn': models.vgg16_bn,
    'vgg19_bn': models.vgg19_bn,
    'densenet121': models.densenet121,
    'densenet161': models.densenet161,
    'densenet169': models.densenet169,
    'densenet201': models.densenet201  
}


def _backbone_model(name, pretrained=True):
    if name.startswith('resnet'):
        resnetx = model_name[name]
        return resnetx(pretrained=pretrained)
    elif name.startswith('vgg'):
        vggx_bn = model_name[name]
        return vggx_bn(pretrained=pretrained).features
    elif name.startswith('densenet'):
        densenetx = model_name[name]
        return densenetx(pretrained=pretrained).features
    else:
        raise NotImplemented(f'backbone model with name : {name} is not implemented!')


def _backbone_feature_names(name):
    if name.startswith('resnet'):
        feature_names = [None, 'relu', 'layer1', 'layer2', 'layer3']
        output_name = 'layer4'
    elif name.startswith('densenet'):
        feature_names = [None, 'relu0', 'denseblock1', 'denseblock2', 'denseblock3']
        output_name = 'denseblock4'
    elif name == 'vgg16_bn':
        feature_names = ['5','12','22','32','42']
        output_name = '43'
    elif name == 'vgg19_bn':
        feature_names = ['5', '12', '25', '38', '51']
        output_name = '52'
    else:
        raise NotImplemented(f'backbone model with name : {name} is not implemented!')
    
    return feature_names, output_name


def get_backbone(name, pretrained=True) -> Tuple[nn.Module, List, str]:
    model = _backbone_model(name, pretrained=pretrained)
    fname, oname = _backbone_feature_names(name)
    
    return model, fname, oname



if __name__ == '__main__':
    model = get_backbone('vgg16_bn')
    print(model)