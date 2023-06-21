#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import torch
import torch.nn as nn
import numpy as np


#Dense layers
class NeuralNetwork(nn.Module): 
    def __init__(self):  
        super(NeuralNetwork, self).__init__()
        self.input = nn.Linear(166,600)
        self.hidden1= nn.Linear(600,300)
        self.hidden2= nn.Linear(300,600)
        self.output = nn.Linear(600,163)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        x = self.input(x)
        x = self.relu(x)
        x = self.hidden1(x)
        #x = self.Tconv1d(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.output(x)  
        return x

#Transpose conv1D with batch normalization
class NeuralNetwork2(nn.Module): 
    def __init__(self):  
        super(NeuralNetwork2, self).__init__()
        self.input = nn.Linear(166,400)
        self.Tconv1d = nn.ConvTranspose1d(in_channels = 1,out_channels = 1,
                        kernel_size=101, stride=1, padding=0, 
                        output_padding=0, groups=1, bias=True, dilation=1)
        self.batch_norm = nn.BatchNorm1d(1, affine=False)        
        self.output = nn.Linear(500,163)
        self.relu = nn.ReLU()
    #L(out)=(L(in)−1)×stride−2×padding+dilation×(kernel_size−1)+output_padding+1
    def forward(self,x):
        x = self.relu(x)
        x = self.input(x)
        x = self.relu(x)
        x = self.batch_norm(x)
        x = self.Tconv1d(x)
        x = self.relu(x)
        x = self.output(x)  
        return x

# Net3 tranpose conv1D + dense 
class NeuralNetwork3(nn.Module): 
    def __init__(self):  
        super(NeuralNetwork3, self).__init__()
        self.input = nn.Linear(166,500)
        self.hidden1= nn.Linear(500,300)
        self.hidden2 = nn.Linear(300,100)

        self.Tconv1d = nn.ConvTranspose1d(in_channels = 1,out_channels = 1,
                        kernel_size=64, stride=1, padding=0, 
                        output_padding=0, groups=1, bias=True, dilation=1)
        #L(out)=(L(in)−1)×stride−2×padding+dilation×(kernel_size−1)+output_padding+1
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.relu(x)
        x = self.input(x)
        x = self.relu(x)
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.Tconv1d(x)
        return x
    
class NeuralNetwork4(nn.Module): 
    def __init__(self):  
        super(NeuralNetwork4, self).__init__()
        self.input = nn.Linear(166,100)
        self.hidden1 = nn.Linear(100,33) 
        self.Tconv1d = nn.ConvTranspose1d(in_channels = 1,out_channels = 1,
                        kernel_size=37, stride=3, padding=0, 
                        output_padding=0, groups=1, bias=True, dilation=1)
    
        self.output = nn.Linear(133,163)
        self.relu = nn.ReLU()
    
    #L(out)=(L(in)−1)×stride−2×padding+dilation×(kernel_size−1)+output_padding+1

    def forward(self,x):
        x = self.relu(x)
        x = self.input(x)
        x = self.hidden1(x)
        x = self.Tconv1d(x)
        x = self.relu(x)
        x = self.output(x)  
        return x
    
#Convolution 1D with max pooling
class NeuralNetwork5(nn.Module): 
    def __init__(self):  
        super(NeuralNetwork5, self).__init__()
        self.input = nn.Linear(166,500)
        self.hidden1= nn.Linear(500,300)

        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=30, stride=1,
                        padding=0, dilation=1, groups=1, bias=True, 
                        padding_mode='zeros', device=None, dtype=None)
        self.max_pool = torch.nn.MaxPool1d(kernel_size=4, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.relu = nn.ReLU()
        self.hidden2 = nn.Linear(67,120)
        self.output = nn.Linear(120,163)
        

    def forward(self,x):
        x = self.relu(x)
        x = self.input(x)
        x = self.relu(x)
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)

        x=self.max_pool(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.output(x)
        return x
    
#Dense layers with tanh activation function
class NeuralNetwork6(nn.Module): 
    def __init__(self):  
        super(NeuralNetwork6, self).__init__()
        self.input = nn.Linear(166,560)
        self.hidden1= nn.Linear(560,360)
        self.hidden2= nn.Linear(760,900)
        self.hidden3= nn.Linear(900,760)
        self.hidden4= nn.Linear(760,563)
        self.output = nn.Linear(360,163)
        self.tanh = nn.Tanh()
        
    def forward(self,x):
        x = self.input(x)
        x = self.tanh(x)
        x = self.hidden1(x)
        x = self.tanh(x)
        
        x = self.hidden2(x)
        x = self.tanh(x)
        x = self.hidden3(x)
        x = self.tanh(x)
        x = self.hidden4(x)
        x = self.tanh(x)
        x = self.output(x)  
        return x

    
class ConvNet(nn.Module): 
    def __init__(self,means,indices): 
        self.means = means
        self.indices = indices
        super(ConvNet, self).__init__()   
        self.conv = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=10)
        self.dropout = nn.Dropout(p=0.2)
        self.batch_norm = nn.BatchNorm1d(1, affine=False)  
        self.dense0 = nn.Linear(166,140)
        self.dense1 = nn.Linear(280,200)
        self.dense2 = nn.Linear(200,170)
        self.output = nn.Linear(170,133)
        self.relu = nn.ReLU()
        self.max_pool = torch.nn.MaxPool1d(kernel_size=5, stride=1)

    def forward(self,x):
        x = self.batch_norm(x)
        x = x.squeeze(1)
        data_166 = np.repeat([self.means],len(x), axis = 0) #filled with mean values
        data_166 = torch.from_numpy(data_166).float()
        data_166[:,self.indices] = x 
        x = data_166.unsqueeze(1)
        x1 = self.dense0(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.dropout(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.dropout(x)
        x = torch.cat((x,x1), axis=2)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.relu(x)
        x = self.output(x)  
        return x
