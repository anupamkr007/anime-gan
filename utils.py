import torch.nn as nn
import os

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant(m.bias.data, 0)

def filesInDir(path):
    """files in dir and subdirs"""
    imgfiles = [os.path.join(root, name)
                 for root, dirs, files in os.walk(path)
                 for name in files
                 if name.endswith((".png", ".jpg"))]
    return imgfiles
