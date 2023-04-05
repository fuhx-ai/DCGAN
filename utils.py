#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/5 11:06
# @Author  : Fuhx
# @File    : utils.py
# @Brief   :
# @Version : 0.1
import numpy as np
from matplotlib import pyplot as plt
from torch import nn
from config import device
import torchvision.utils as vutils


def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def show_dataset(dataloader):
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()
