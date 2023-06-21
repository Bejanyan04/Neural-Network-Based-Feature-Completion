#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch

class MyDataset(Dataset):
    def __init__(self, X_train, Y_train):
        x = X_train.values
        y = Y_train.values
        self.x_train=torch.from_numpy(x).float()
        self.y_train=torch.from_numpy(y).float()

    def __len__(self):
        return len(self.y_train)
   
    def __getitem__(self,idx):
        return self.x_train[idx].float(), self.y_train[idx].float()

