
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataPreprocessing():
    def __init__(self,data,weigths):
        self.CLIP_MIN = 13 # const values to indicate max and min of clipping
        self.CLIP_MAX = 90
        self.SCALE_PARAM =3.5
        self.dataframe = data 
        self.weights = weigths
        self.unscaled_y_train = 0
        self.unscaled_y_test = 0
        self.train_mean = 0 # mean value of training data :163 columns
        self.group_count  = 0
        
    def UpdateWeights(self):
        """Add weights for the last three columns of the dataframe"""
        self.weights = pd.concat([self.weights,pd.Series([1,1,1])],ignore_index=True)
        
    def CorrectOutliers(self):
        """Handle outliers existing in some columns"""
        
        self.dataframe['A'] = self.dataframe['A'].clip(self.CLIP_MIN,self.CLIP_MAX)
    
    
    def EncodeStrings(self):
        """Encode String Values of the last column to the range [-1,1]"""
        self.df_value_counts = self.dataframe.loc[:,"C"].value_counts()[:10]
        self.names  = np.append(self.df_value_counts.index.values, "others")
        self.dict_ = dict(zip(self.names,np.arange(10,-1,-1)))
        self.dataframe.loc[:,"C"] = self.dataframe.loc[:,'C'].map(self.dict_)
        self.dataframe["C"]  = self.dataframe["C"].fillna(0)
        self.dataframe["C"] = (self.dataframe["C"] - 5)/6
  
    def RandomChoice(self):
        """Choose random 30 column indices and concatenate last three column indices"""
        return np.concatenate((np.random.choice([i for i in range(163)],30, replace=False),[163,164,165]))
    
        
    def SplitData(self,train_size=0.9):
        """Split unscaled data to train, test, and save unscaled values of train and test data 
        in purpose of using them during Ai model testing step"""
        train_data, test_data =  train_test_split(self.dataframe,train_size=train_size) ##166 columns
        return train_data, test_data
    

    def Normalization(self,data):
        
        """Normalization Function"""
        data = data.copy()
        data.loc[:,'G'].replace({0:1, 1:-1, 2:0, 3:1}, inplace=True)
        data.loc[:,'A']  = (data.loc[:,'A']-30)/50
        data.iloc[:,:-3] = (data.iloc[:,:-3] - self.unscaled_train_mean) / self.SCALE_PARAM
        #data.iloc[:,:-3] = (data.iloc[:,:-3]-2.5)/3
        return data

    def NormalizeData(self,unscaled_train_data,unscaled_test_data):
        
        """Normalize data to [-1,1] """
        self.unscaled_train_mean = unscaled_train_data.mean().iloc[:-3] #
        self.unscaled_y_train = unscaled_train_data.iloc[:,:-3].copy() 
        self.unscaled_y_test = unscaled_test_data.iloc[:,:-3].copy() 
        self.scaled_train_data, self.scaled_test_data = list(map(self.Normalization,[unscaled_train_data,unscaled_test_data]))
        self.scaled_train_mean = self.scaled_train_data.mean()
        return self.scaled_train_data, self.scaled_test_data
        # save train mean and normalize data

    def GetAiInput33(self,indices):
        """Normalize data to [-1,1] """

        col_indices = self.dataframe.columns[indices]
        scaled_x_train33 = self.scaled_train_data[col_indices]
        scaled_x_test33 = self.scaled_test_data[col_indices]
        scaled_y_train = self.scaled_train_data.iloc[:,:-3].copy() 
        scaled_y_test = self.scaled_test_data.iloc[:,:-3].copy()  #target output (163 features)
        return scaled_x_train33, scaled_x_test33, scaled_y_train, scaled_y_test 
    
    def GroupNumber(self):
        """Save indices of group starting"""
        #save size of each group
        #return indexing for spliting data with groups in training and testing of Ai model
        first_letters = [i[0] for i in list(self.dataframe.iloc[:,:-3].columns)]
        self.group_count = [first_letters.count(el) for el in np.unique(first_letters)]
        return [sum(self.group_count[0:i]) for i in range (1,len(self.group_count))]
    
    def MissingFeatures(self, indices):
        """Get indices of the missing features"""
        return  np.setdiff1d(np.arange(0,163), indices, assume_unique=True)
    
    def CorrectData(self):
        self.UpdateWeights()
        self.CorrectOutliers()
        self.EncodeStrings()

 
