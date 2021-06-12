
import numpy as np
import torch
import torch.nn as nn
import torchvision
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn import metrics
import torchvision.transforms as transforms

def DASVDD_test(model,in_shape,Gamma,test_loader,labels,criterion):

  with torch.no_grad():
    score = []
    i = 0
    for x_test in test_loader:
      if isinstance(x_test, list):
          x_test = x_test[0]
      x_test =x_test.view(-1, in_shape).to(device)
      x_test_hat,code_test = model(x_test)
      loss = criterion(x_test_hat,x_test) + Gamma*torch.sum((code_test.to(device) - C) ** 2, dim=1)[0]
      score.append(loss.to("cpu").item())
      i+=1
    return metrics.roc_auc_score(labels,score)*100



