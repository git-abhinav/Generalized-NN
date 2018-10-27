import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_breast_cancer
import sklearn.linear_model
%matplotlib inline

np.random.seed(2) 

dataset = load_breast_cancer()
X = dataset.data
Y = dataset.target
print(Y.shape)

idx = np.random.permutation(X.shape[0])
X, Y = X[idx], Y[idx]

X = X.T


X = (X-np.mean(X, axis=1, keepdims = True))/(np.max(X, axis=1, keepdims = True)-np.min(X, axis = 1, keepdims = True))

shape_X = X.shape

Y = Y.reshape(1,Y.shape[0])

shape_Y = Y.shape

m = X.shape[1]  
print(X)
print("Rows and columns in feature set(X) : ", X.shape)
print('Label vector : ', Y.shape[0])
print ('Number of training examples:', m)



def layer_sizes(X, Y):
    n_x = X.shape[0]
    n_h_one = 4 
    n_h_two = 3
    n_y = Y.shape[0]
    print("n_y is ",n_y)
    return (n_x, n_h_one, n_h_two, n_y)

(n_x, n_h_one, n_h_two, n_y) = layer_sizes(X, Y)

print("Input layer size : ",str (n_x))
print("Hidden layer 1 size : ", str(n_h_one))
print("Hidden layer 2 size : ", str(n_h_two))
print("Outout layer size : ", str(n_y))

def initialize_parameters(n_x, n_h_one, n_h_two, n_y):
    np.random.seed(2)
    W1 = np.random.randn(n_h_one, n_x)*0.005
    b1 = np.zeros((n_h_one, 1))
    print("W1.shape", W1.shape)
    print("b1.shape", b1.shape)
    
    # this is the first hidden layer
    # and will have input from input layer 
    W2 = np.random.randn(n_h_two, n_h_one)*0.005
    b2 = np.zeros((n_h_two, 1))
    print("W2.shape", W2.shape)
    print("b2.shape", b2.shape)
    
    #
    W3 = np.random.randn(n_y, n_h_two)*0.005
    b3 = np.zeros((n_y, 1))

    print("W3.shape", W3.shape)
    print("b3.shape", b3.shape)
    parameters = {
        "W1":W1,
        "W2":W2,
        "W3":W3,
        "b1":b1,
        "b2":b2,
        "b3":b3
    }
    return parameters 

parameters = initialize_parameters(n_x, n_h_one, n_h_two, n_y)





def sigmoid(z):
    s = 1.0/(1+np.exp(-z))
    return s
def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = np.tanh(Z2)
    Z3 = np.dot(W3,A2) + b3
    A3 = sigmoid(Z3)
    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2,
        "Z3": Z3,
        "A3": A3
    }
    return A3, cache
A3, cache = forward_propagation(X, parameters)
def compute_cost(A3, Y, parameters):
    m = Y.shape[1]
    print("m ===", m)
    print("A3 ", A3)
    logprobs = np.multiply(Y, np.log(A3) ) + np.multiply((1 - Y), np.log(1 - A3))
    cost = -np.sum(logprobs) / m
    cost = np.squeeze(cost)
    assert(isinstance(cost, float))
    return cost
def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    A3 = cache["A3"]
    
    dZ3 = A3-Y
    dW3 = 1/m*(np.dot(dZ3,A2.T))
    db3 = 1/m*(np.sum(dZ3, axis=1, keepdims=True))
    
    dZ2 = np.multiply(np.dot(W3.T,dZ3),(1-np.power(A2,2)))
    dW2 = 1/m*(np.dot(dZ2,A1.T))
    db2 = 1/m*(np.sum(dZ2,axis=1, keepdims=True))
    
    dZ1 = np.multiply(np.dot(W2.T,dZ2),(1-np.power(A1,2)))
    dW1 = 1/m*(np.dot(dZ1,X.T))
    db1 = 1/m*(np.sum(dZ1,axis=1, keepdims=True))
    
    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2,
        "dW3": dW3,
        "db3": db3
    }
    return grads
def update_parameters(parameters, grads, learning_rate = 0.2):
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    dW3 = grads["dW3"]
    db3 = grads["db3"]
    
    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2
    W3 = W3 - learning_rate*dW3
    b3 = b3 - learning_rate*db3
    
    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "W3": W3,
        "b3": b3
    } 
    return parameters

n_x = layer_sizes(X, Y)[0]
n_y = layer_sizes(X, Y)[3]

def nn_model(X, Y, n_h_one, n_h_two, num_iterations = 10000, print_cost=False):
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[3]
    
    parameters = initialize_parameters(n_x,n_h_one,n_h_two,n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    for i in range(0, num_iterations):
        A3, cache = forward_propagation(X,parameters)
        cost = compute_cost(A3,Y,parameters)
        grads = backward_propagation(parameters,cache,X,Y)
        parameters = update_parameters(parameters,grads)
        if print_cost and i % 2000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    return parameters

def predict(parameters, X):
    A3, cache = forward_propagation(X, parameters)
    prediction = 1*(A3>0.5)
    return prediction

parameters = nn_model(X, Y, n_h_one = 4, n_h_two = 3, num_iterations = 20000, print_cost=True)
predictions = predict(parameters, X)
print("Predictions are : ")
#print(predictions)

print( X.shape, predictions.shape)
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')
