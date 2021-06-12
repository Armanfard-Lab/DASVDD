import numpy as np
import torch
import torch.nn as nn
import torchvision
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn import metrics
import torchvision.transforms as transforms

class AE_MNIST(nn.Module):
  def __init__(self, **kwargs):
    super().__init__()
    In_shape = kwargs["input_shape"]
    self.encoder_hidden_layer = nn.Linear(in_features=In_shape, out_features=1024)
    self.encoder_middle = nn.Linear(in_features=1024,out_features=256)
    self.encoder_output_layer = nn.Linear(in_features=256,out_features=256)
    self.decoder_hidden_layer = nn.Linear(in_features=256,out_features=256)
    self.decoder_middle = nn.Linear(in_features=256,out_features=1024)
    self.decoder_output_layer = nn.Linear(in_features=1024,out_features=In_shape)

  def forward(self,features):
    activation = F.leaky_relu(self.encoder_middle(F.leaky_relu(self.encoder_hidden_layer(features))))
    code = F.leaky_relu(self.encoder_output_layer(activation)) 
    activation = F.leaky_relu(self.decoder_middle(F.leaky_relu(self.decoder_hidden_layer(code))))
    reconstructed = torch.sigmoid(self.decoder_output_layer(activation))
    return reconstructed,code




