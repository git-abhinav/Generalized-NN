# this cell is for neural network
import numpy as np        
import pandas as pd
import sklearn

class Neural_network(object):
    '''
        creating the class 'Neural_network' 
        that has 'hidden_layers' number of hidden layers
        and 'nodes_on_each_layer'defines the no. of nodes 
        in each of the hidden layers of the neural network
    '''
    def __init__(self, hidden_layers, nodes_on_each_layer):
        self.features = None
        self.labels = None
        self.hidden_layers = hidden_layers
        self.W = None 
        self.B = None 
    def load_data(self, normalize_how = 0):  
        '''
            input : normalize_how indicates which 
                    domain for the features we wan
                    if this attribute is 0 the domain is {0, 1}
                    else domain or the range is {-1, 1}
            output : none 
        '''
        
        # loading data from csv file using pandas library 
        data = pd.read_csv("avila-tr.csv")
        
        # setting up the names for the features 
        X = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10']
        
        # labels storese the class label of the dataset
        self.labels = data[['class']]
        
        X = data[X] 
        if normalize_how == 0:  # {0, 1}
            self.features = (X-np.min(X))/(np.max(X)-np.min(X))
        else:                   # {-1, 1}
            self.features = (X-np.mean(X))/(np.max(X)-np.min(X))
        
        # printing unique class labels
        print("Data set loaded and Unique class labels are : ")
        print(np.unique(self.labels)) 
        
        # providing the necessay information about the class label(summary)
        print("\n") 
        print("Description about the class labels : ")
        print "-"*38
        print("\n")
        for i in np.unique(self.labels):
            l = self.labels==i
            l = l[l==True] 
            print i, " class label has ", (l.dropna()).shape[0], " instances ~ ",round((float((l.dropna()).shape[0])/(self.labels.shape[0]))*100), "%" 
    def Set_parameters(self, no_of_hidden_layers, nodes_on_each_layer):
        pass
n = Neural_network(4,5)
n.load_data(1)
            
            
            
        
    