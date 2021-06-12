

import numpy as np
import torch
import torch.nn as nn
import torchvision
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn import metrics
import torchvision.transforms as transforms

def get_target_label_idx(labels, targets):
    return np.argwhere(np.isin(labels, targets)).flatten().tolist()


def global_contrast_normalization(x: torch.tensor, scale='l2'):

    n_features = int(np.prod(x.shape))

    mean = torch.mean(x)
    x -= mean

    if scale == 'l1':
        x_scale = torch.mean(torch.abs(x))

    if scale == 'l2':
        x_scale = torch.sqrt(torch.sum(x ** 2)) / n_features

    x /= x_scale

    return x

def OneClass(train_dataset,test_dataset,Class):
  Samples = []
  for i in train_dataset:
    x,y = i
    if y == Class:
      Samples.append(x)
  len(Samples)
  labels = []
  test_points = []
  bp = 0
  counter = 0
  for i in test_dataset:
    x,y = i
    if y == Class:
      bp+=1
      LBL = 0
      labels.append(LBL)
      test_points.append(x)
    elif y!=Class:
      counter+=1
      LBL = 1
      labels.append(LBL)
      test_points.append(x)
  return Samples,test_points,labels






