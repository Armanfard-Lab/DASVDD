import numpy as np
import torch
import torch.nn as nn
import torchvision
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn import metrics
import torchvision.transforms as transforms
from .Utils import get_target_label_idx,global_contrast_normalization,OneClass


def MNIST_loader(train_batch,test_batch,Class):
  transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
  train_dataset = torchvision.datasets.MNIST(root="~/torch_datasets",train=True,transform=transform,download=True)
  test_dataset = torchvision.datasets.MNIST(root="~/torch_datasets", train=False, transform=transform, download=True)


  Digits,new_test,labels = OneClass(train_dataset,test_dataset,Class)
  train_loader = torch.utils.data.DataLoader(Digits,batch_size=train_batch,shuffle=True,num_workers=2,pin_memory=True,drop_last = True)
  test_loader = torch.utils.data.DataLoader(new_test,batch_size=test_batch,shuffle=False,num_workers=2)
  return train_loader,test_loader,labels

def FMNIST_loader(train_batch,test_batch,Class):
  transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
  train_dataset = torchvision.datasets.FashionMNIST(root="~/torch_datasets",train=True,transform=transform,download=True)
  test_dataset = torchvision.datasets.FashionMNIST(root="~/torch_datasets", train=False, transform=transform, download=True)


  Digits,new_test,labels = OneClass(train_dataset,test_dataset,Class)
  train_loader = torch.utils.data.DataLoader(Digits,batch_size=train_batch,shuffle=True,num_workers=2,pin_memory=True,drop_last = True)
  test_loader = torch.utils.data.DataLoader(new_test,batch_size=test_batch,shuffle=False,num_workers=2)
  return train_loader,test_loader,labels

def CIFAR_loader(train_batch,test_batch,Class):
  min_max = [(-28.94083453598571, 13.802961825439636),
              (-6.681770233365245, 9.158067708230273),
              (-34.924463588638204, 14.419298165027628),
              (-10.599172931391799, 11.093187820377565),
              (-11.945022995801637, 10.628045447867583),
              (-9.691969487694928, 8.948326776180823),
              (-9.174940012342555, 13.847014686472365),
              (-6.876682005899029, 12.282371383343161),
              (-15.603507135507172, 15.2464923804279),
              (-6.132882973622672, 8.046098172351265)]
  transform = transforms.Compose(
      [transforms.ToTensor(),transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l2'))])

  train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)


  test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
  Digits,new_test,labels = OneClass(train_dataset,test_dataset,Class)
  train_loader = torch.utils.data.DataLoader(Digits,batch_size=train_batch,shuffle=True,num_workers=2,pin_memory=True,drop_last = True)
  test_loader = torch.utils.data.DataLoader(new_test,batch_size=test_batch,shuffle=False,num_workers=2)
  return train_loader,test_loader,labels



def Speech_loader(train_batch,test_batch):
  X = pd.read_csv('speech.csv',header=None)
  y = X[400].copy()
  X.drop(columns=400,inplace=True)
  R = int((3625 + 61)*0.1)
  X_test = X[:R]
  X_train = X[R:]
  y_test = y[:R]
  y_train = y[R:]
  train = torch.utils.data.TensorDataset(torch.Tensor(X_train.values))
  train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch, shuffle=True,drop_last=True)
  test = torch.utils.data.TensorDataset(torch.Tensor(X_test.values),torch.Tensor(y_test.values))
  test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch, shuffle=False)
  return train_loader,test_loader,y_test

def PIMA_loader(train_batch,test_batch):
  X = pd.read_csv('pima.csv',header=None)
  X.sort_values(by=8,inplace=True,ascending=False)
  y = X[8].copy()
  X.drop(columns=8,inplace=True)
  R = int((768)*0.4)
  X_test = X[:R]
  X_train = X[R:]
  y_test = y[:R]
  y_train = y[R:]
  train = torch.utils.data.TensorDataset(torch.Tensor(X_train.values))
  train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch, shuffle=True,drop_last=True)
  test = torch.utils.data.TensorDataset(torch.Tensor(X_test.values),torch.Tensor(y_test.values))
  test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch, shuffle=False)
  return train_loader,test_loader,y_test





