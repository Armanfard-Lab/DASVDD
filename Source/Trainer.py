
import numpy as np
import torch
import torch.nn as nn
import torchvision
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn import metrics
import torchvision.transforms as transforms

def DASVDD_trainer(model,in_shape,code_size,C,train_loader,optimizer,update_center,criterion,Gamma,device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                   num_epochs = 300,K=0.9,verbosity=0):
  
  
  num_epochs = 300
  K = 0.9
  L1 = np.zeros(num_epochs)
  L2 = np.zeros(num_epochs)
  L3 = np.zeros(num_epochs)
  c_vals = np.zeros(num_epochs)

  for epoch in range(num_epochs):
    loss = 0
    aeloss = 0
    svddloss = 0
    for batch_features in train_loader:
      if isinstance(batch_features, list):
          batch_features = batch_features[0]
      batch_features = batch_features.view(-1, in_shape).to(device)
      Num_batch = int(np.ceil(K*batch_features.size()[0]))
      optimizer.zero_grad()
      outputs,code = model(batch_features[:Num_batch,:])
      R = torch.sum((code.to(device) - C) ** 2, dim=1)[0]
      train_loss = criterion(outputs,batch_features[:Num_batch,:]) + Gamma*R
      train_loss.backward()
      optimizer.step()
      loss += train_loss.item()
      aeloss += criterion(outputs,batch_features[:Num_batch,:]).item()
      svddloss += R.item()
      _,c_code = model(batch_features[Num_batch:,:])
      center = torch.mean(c_code,axis=0)
      center_loss = criterion(C,center)
      center_loss.backward()
      update_center.step()
      c_vals[epoch]+= C[0]
    c_vals[epoch] = c_vals[epoch]/len(train_loader)
    loss = loss/len(train_loader)
    aeloss = aeloss/len(train_loader)
    svddloss = svddloss/len(train_loader)
    L1[epoch] = loss
    L2[epoch] = aeloss
    L3[epoch] = svddloss
    if verbosity==1:
      print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, num_epochs, loss))
    if verbosity == 2:
      return L1,L2,L3,c_vals


# In[ ]:




