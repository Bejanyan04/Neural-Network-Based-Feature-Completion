#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from DataLoader import MyDataset
from torch.utils.data import DataLoader
from  Models import *

class UseNeuralNetwork():
    def __init__(self,NeuralNetwork,x_train,x_test,y_train,y_test,indices,optimizer,lr_rate=0.001,batch_size=32,num_epochs=20,num_classes=163,inp_size=166,loss=nn.MSELoss()):
        self.missing_features = torch.from_numpy(np.setdiff1d(np.arange(0,163), indices, assume_unique=True))
        self.model = NeuralNetwork()
        #self.preprocess = DataPreprocessing()
        self.input_size = inp_size
        self.num_classes = num_classes
        self.learning_rate = lr_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lossFunction = loss
        self.optimizer = optimizer
        self.x_train,self.x_test,self.y_train,self.y_test = x_train,x_test,y_train,y_test

            
    def TrainNN(self, group_split_ind,weights=[10,4]):
        Ds = MyDataset(self.x_train, self.y_train)
        train_loader=DataLoader(Ds,self.batch_size,shuffle=True)
        for epoch in range(self.num_epochs):
            acc=0
            print("Epoch: " + str(epoch))
            epoch_loss = 0
            for i, (data, targets) in enumerate(train_loader):
                #data = torch.index_select(data, 1, train_133)
                targets_133 = torch.index_select(targets, 1, self.missing_features)
                data = data.float().unsqueeze(1)
                #Forward
                predictions = self.model(data)
                predictions = predictions.squeeze(1)
                predictions_133 = torch.index_select(predictions, 1, self.missing_features)
                predictions_16 = torch.stack([i.sum(axis=1) for i in torch.tensor_split(predictions,group_split_ind, dim=1)]).T
                targets_16 = torch.stack([i.sum(axis=1) for i in torch.tensor_split(targets,group_split_ind, dim=1)]).T
                loss_133= self.lossFunction(predictions_133, targets_133)
                loss_16 = self.lossFunction(predictions_16,targets_16)
                if epoch < self.num_epochs/2:
                    loss = (weights[0]*loss_133 + weights[1]*loss_16)
                else:
                    loss = (weights[1]*loss_133 + weights[0]*loss_16)    
                epoch_loss += loss
                #Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print("loss: ",epoch_loss/i)
                
   
        
    def Plot(self, metric, metric_type, indices):
        #Plot accuracy metric values for each feature
        all_index = np.arange(0,163) 
        all_col = self.x_train.columns[all_index]
        ind_col = self.x_train.columns[indices]
        colors = ['green' if ind in indices else 'blue' for ind in all_index]
        plt.figure(figsize = (80,20))
        plt.bar(all_col, metric, color = colors)
        plt.xlabel("Feautures")
        plt.ylabel(metric_type)
        plt.show()
    
    def CheckAccuracy(self,input_scaled,target_unscaled,train_mean,SCALE_PARAM=3.5):
        """Accuarcy computed for whole dataset only for absent data(133 features)"""
        input_scaled = torch.from_numpy(input_scaled.values).float().unsqueeze(1)
        target_unscaled = torch.from_numpy(target_unscaled.values).float()
        ##test and get prediction
        with torch.no_grad():
            pred_scaled = self.model(input_scaled) 
         #separate 133 features to compute accuracy only for them   
        target_unscaled = torch.index_select(target_unscaled, 1, self.missing_features)
        #unscale prediction values to [-5,5]
        prediction_float = (pred_scaled * SCALE_PARAM) + torch.from_numpy(train_mean)
        prediction_float = torch.reshape(prediction_float,(prediction_float.shape[0],prediction_float.shape[-1]))
        prediction_float = torch.index_select(prediction_float, 1, self.missing_features)  #133 feature separation
        #metric 1:binary accuracy
        prediction_int = torch.round(prediction_float) #round values
        equality_matrix = (prediction_int==target_unscaled)
        num_correct = equality_matrix.sum()
        num_samples = (equality_matrix.shape[0]*equality_matrix.shape[-1])
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples):.5f}') 
        #metric 2:mean abs distance 
        prediction_float = torch.reshape(prediction_float,(prediction_float.shape[0],prediction_float.shape[-1]))
        distance_accuracy = torch.abs(prediction_float - target_unscaled).mean()
        print(distance_accuracy , "  accuracy  with second accuracy metric") 
        
        
    def CheckAccuracyFeatures (self,input_scaled,target_unscaled,train_mean,indices,SCALE_PARAM=3.5):
        """ Accuracy for each column, for accuracy of predicted 163 features """
        input_scaled = torch.from_numpy(input_scaled.values).float().unsqueeze(1)
        target_unscaled = torch.from_numpy(target_unscaled.values).float()
        #test and get prediction
        with torch.no_grad():
            pred_scaled = self.model(input_scaled)
        #unscale prediction   
        prediction_float = (pred_scaled * SCALE_PARAM) + torch.from_numpy(train_mean)
        prediction_int = torch.round(prediction_float)
        prediction_int = torch.reshape(prediction_int,(prediction_int.shape[0],prediction_int.shape[-1]))
        equality_matrix = (prediction_int==target_unscaled)
        metric1 = equality_matrix.sum(axis=0)/equality_matrix.shape[0]
        self.Plot(metric1,"binary:metric1",indices)
        # Mean abs distance metric 
        prediction_float = torch.reshape(prediction_float,(prediction_float.shape[0],prediction_float.shape[-1]))
        metric2 = torch.abs(prediction_float - target_unscaled).mean(axis=0)
        self.Plot(metric2, "Mean abs distance metric:metric2",indices)
        #Earth mover distance for each column
        metric3_col = []
        for col in range(163):
            target_one_col = target_unscaled[col]
            predict_int_one_col = prediction_int[col]
            metric3 = wasserstein_distance(target_one_col,predict_int_one_col)
            metric3_col.append(metric3)
        self.Plot(metric3_col,"The earth mover distance",indices)        
        

