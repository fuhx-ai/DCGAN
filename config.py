#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/5 10:49
# @Author  : Fuhx
# @File    : config.py
# @Brief   :
# @Version : 0.1
import torch

# Batch size during training
batch_size = 8

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Path of images
data_root = '../imgs/'

# training device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')