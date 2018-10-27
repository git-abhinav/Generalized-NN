# this cell is for neural network
import numpy as np        
import pandas as pd
from sklearn.datasets import load_breast_cancer
import sklearn
np.random.seed(2)

class Neural_network(object):
    '''
        creating the class 'Neural_network' 
        that has 'hidden_layers' number of hidden layers 
        deafault value of 'hidden_layes' is 1
        and 'nodes_on_each_layer'defines the no. of nodes 
        in each of the hidden layers of the neural network
    '''
    def __init__(self, hidden_layers=1, units_on_each_layer = np.array([3])):
        self.features = None
        self.labels = None
        self.n_x = None
        self.hidden_layers = hidden_layers
        self.a = {}
        self.z = None
        self.input_layer = None
        self.output_layer = None 
        self.units_on_each_layer = units_on_each_layer
        self.W = {} 
        self.b = {} 
        self.m = None
        self.Z = {}
    def sigmoid(self, z):
        a = 1.0/(1+np.exp(-z))
        return a
    def load_data_breast_cancer(self):
        '''
            objective: to load breast_cancer dataset form sklearn
        '''
        d = load_breast_cancer()
        self.features = d.data
        self.labels = d.target
        self.features = self.features.T
        self.labels = self.labels.reshape(1, self.labels.shape[0])
        self.n_x = self.features.shape[0]
        self.m = self.features.shape[1]
        #self.features = (self.features-np.mean(self.features))/(np.max(self.features)-np.min(self.features))
        X = self.features
        X = (X-np.mean(X, axis=1, keepdims = True))/(np.max(X, axis=1, keepdims = True)-np.min(X, axis = 1, keepdims = True))
        self.features = X
    
    def initialize_parameters(self):
        for i in range(1, self.hidden_layers+1):
            if i == 1:
                self.W[i] = np.random.randn(self.units_on_each_layer[i-1], self.features.shape[0])
            else:
                self.W[i] = np.random.randn(self.units_on_each_layer[i-1], self.W[i-1].shape[0])
            n.b[i] =  np.zeros((n.units_on_each_layer[i-1], 1))
    def load_data_avila(self, normalize_how = 0):  
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
    
    def forward_propagation(self):
        for i in range(1, self.hidden_layers+1):
            if i == 1:
                self.Z[i] = np.dot(self.W[1], self.features) + self.b[i]
                self.a[i] = np.tanh(self.Z[i])
            elif i == self.hidden_layers:
                self.Z[i] = np.dot(self.W[i], self.a[i-1]) + self.b[i]
                self.a[i] = np.tanh(self.Z[i])
            else:
                self.Z[i] = np.dot(self.W[i], self.a[i-1]) + self.b[i]
                self.a[i] = sigmoid(self, self.Z[i])
        return self.a[self.hidden_layers]
    
    def compute_cost(self):
        #logprobs = np.multiply(self.labels, np.log(self.a[self.hidden_layers]) ) + np.multiply((1 - self.labels), np.log(1 - self.a[self.hidden_layers]))
        print(self.a[1])
        print(" m== ", self.m)
        #cost = -np.sum(logprobs)/self.m
        
        #cost = np.squeeze(cost)
       # assert(isinstance(cost, float))
       # return cost
'''
    main code starts from here 
'''
n = Neural_network(hidden_layers=3, units_on_each_layer = np.array([4, 3, 1]))
#n.load_data_avila(1)
n.load_data_breast_cancer()      
n.initialize_parameters()
n.forward_propagation()
n.compute_cost()

