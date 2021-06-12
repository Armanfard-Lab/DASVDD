
import numpy as np
import torch
import torch.nn as nn
import torchvision
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn import metrics
import torchvision.transforms as transforms

def tune_gamma(AE,in_shape,criterion,train_loader,device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),T=10):
  gamma = 0
  for k in range(T):
    model = AE(input_shape=in_shape).to(device)
    R = 0
    RE = 0
    for batch_features in train_loader:
        if isinstance(batch_features, list):
          batch_features = batch_features[0]
        batch_features = batch_features.view(-1, in_shape).to(device)
        outputs,code = model(batch_features)
        R += torch.sum((code.to(device)) ** 2, dim=1)[0]
        RE += criterion(outputs,batch_features)
    R = R/len(train_loader)
    RE = RE/len(train_loader)
    gamma += RE/R

  gamma = gamma/T
  gamma = gamma.detach().item()
  return gamma


