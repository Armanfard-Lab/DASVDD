import numpy as np
import torch
import torch.nn as nn
import torchvision
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn import metrics
import torchvision.transforms as transforms
from Dataset.Utils import get_target_label_idx,global_contrast_normalization,OneClass
from Dataset.DatasetLoader import MNIST_loader,FMNIST_loader, CIFAR_loader,Speech_loader,PIMA_loader
from Network.CIFARNet import AE_CIFAR
from Network.MNISTNet import AE_MNIST
from Network.PIMANet import AE_PIMA
from Network.SpeechNet import AE_Speech
from Source.GammaTune import tune_gamma
from Source.Tester import DASVDD_test
from Source.Trainer import DASVDD_trainer

train_loader,test_loader,labels = MNIST_loader(train_batch=200,test_batch=1,Class=0)
in_shape = 28*28
code_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AE_MNIST(input_shape=in_shape).to(device)
params = list(model.parameters())
optimizer = torch.optim.Adam(params,lr=1e-3)
C = torch.randn(code_size,device = device,requires_grad=True)
update_center = torch.optim.Adagrad([C],lr=1,lr_decay=0.01)
criterion=nn.MSELoss()
Gamma = tune_gamma(AE_MNIST,in_shape,criterion,train_loader,device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),T=10)

DASVDD_trainer(model,in_shape,code_size,C,train_loader,optimizer,update_center,criterion,Gamma,device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                   num_epochs = 300,K=0.9)

DASVDD_test(model,C,in_shape,Gamma,test_loader,labels,criterion,C)

