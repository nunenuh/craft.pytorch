import torch

from ..models.craft import CRAFT
from collections import OrderedDict


def copy_state_dict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def net_forward(model, images):
    predict, feature = model(images)
    pred_char = predict[:, :, :, 0].unsqueeze(dim=1).cuda()
    pred_aff = predict[:, :, :, 1].unsqueeze(dim=1).cuda()
    return pred_char, pred_aff, feature


def freeze_network(model):
    for p in model.parameters():
        p.required_grad = False


def unfreeze_conv_cls_module(model):
    for p in model.conv_cls.parameters():
        p.required_grad = True


def load_craft_network(use_gpu=True, weight_path='weights/craft_mlt_25k.pth'):
    model = CRAFT(pretrained=True)
    weights = torch.load(weight_path, map_location=torch.device('cpu'))
    weights = copy_state_dict(weights)
    model.load_state_dict(weights)
    if use_gpu:
        model = model.cuda()
    return model



