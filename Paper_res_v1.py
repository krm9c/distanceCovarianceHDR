import numpy as np
import matplotlib.pyplot as plt


# Lets do this, get the rolling element data in and run the logistic regression for it
Results=[]
from tqdm import tqdm
import os, sys
sys.path.append('../CommonLibrariesDissertation')
from Library_Paper_two import *
import Network_class
import tensorflow as tf
import gzip, cPickle
import numpy as np
from sklearn import preprocessing
import tflearn
import math
from SparsePCA import *
import gc
from random import random
import sklearn
from sklearn import linear_model

def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))

def sigmoid_dev(scores):
    return (sigmoid(scores)*(1-sigmoid(scores)))

def cross_entropy_scores(features, target, weights):
    ll = np.sum( target*log(scores) - (1-target)*np.log(1 - scores))
    return ll

def xavier(fan_in, fan_out):
    low = -4*np.sqrt(4.0/(fan_in + fan_out)) # use 4 for sigmoid, 1 for tanh activation
    high = 4*np.sqrt(4.0/(fan_in + fan_out))
    return np.random.uniform(low, high, [fan_in, fan_out])

def weight_variable( in_, out):
    initial = xavier( in_, out)
    return initial

 # Bias function
def bias_variable(in_, out):
    return np.random.normal(size = [in_ , out])

def import_pickled_data(string):
    f = gzip.open('../data/'+string+'.pkl.gz','rb')
    dataset = cPickle.load(f)
    X_train = dataset[0]
    X_test  = dataset[1]
    y_train = dataset[2]
    y_test  = dataset[3]
    return X_train, y_train, X_test, y_test

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

# my (correct) solution:
def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x/div

# Define the cost function
def cost(y, t):
    return - np.sum(np.multiply(t, np.log(y)) + np.multiply((1-t), np.log(1-y)))

# Class
class reg():
    def __init__(self, inputs, classes):
        self.regression = {}
        self.regression["weight"] = weight_variable(inputs, classes)
        self.regression["bias"] = bias_variable(1, classes)
        self.regression["lr"] = 0.01
        self.feature_reduction = {}
        self.beta_11 = 0.01
        self.beta_22 = 0.001
        self.m_1 = np.zeros(self.regression["weight"].shape)
        self.v_1 = np.zeros(self.regression["weight"].shape)

    def update_weight(self, score, batch_xs, batch_ys, weight, bias, lr):
        grad = 2*np.dot((score - batch_ys).T, batch_xs).T

        ## Calculate momentum and variance parameters
        self.m_1   = (self.beta_11*self.m_1 + (1-self.beta_11)*grad)
        self.v_1   = self.beta_22*self.v_1+(1-self.beta_22)*np.square(grad)

        self.m_1 = (self.m_1/float(1-self.beta_11+0.000001))
        self.v_1 = (self.v_1/float(1-self.beta_22+0.000001))
        ## Tune the sparsity parameter
        # decay learning rates
        self.beta_11 = 0.99*self.beta_11
        self.beta_22 = 0.99*self.beta_22
        return (weight- lr* np.divide(self.m_1, (np.sqrt(self.v_1) + 0.000001) ) ) , bias

    def logistic_regression(self, X, Y, classes, inputs, iterations):
        # self.regression["lr"] = 0.001
        for i in xrange(iterations):
            self.regression["lr"] = 0.99*self.regression["lr"]
            for batch in iterate_minibatches(X, Y, 200, shuffle=True):
                batch_xs, batch_ys = batch
                score = softmax(np.dot(batch_xs, self.regression["weight"]))
                self.regression["weight"], self.regression["bias"] = \
                self.update_weight(score, batch_xs, batch_ys, self.regression["weight"],\
                self.regression["bias"], self.regression["lr"])
            final_scores = (np.dot(X, self.regression["weight"]))
            preds = softmax(final_scores)
            print 'Accuracy from scratch: {0}'.format(accuracy(preds, Y))

        return self

dataset = 'rolling'

from sklearn.decomposition import TruncatedSVD

## Data In
def Low_Rank_Approx(r, X, X1):
    # U, S_1, V, E = sparse_pca(Sigma,n_comp , x, maxit=100,\
    # method='lars', n_jobs=1, verbose=0)
    # T = np.dot(U,np.diag(S_1))

    Sigma = np.corrcoef(np.transpose(X))

    from sklearn.linear_model import ElasticNet
    e_net = ElasticNet(alpha= 0.1, copy_X=False, fit_intercept=False, l1_ratio=0.7,
      max_iter=1000, normalize=True, positive=False, precompute=False,
      random_state=None, selection='cyclic', tol=0.000001, warm_start=False)

    ## Define a low rank matrix
    P = np.identity(Sigma.shape[0])
    svd = TruncatedSVD(n_components=r, n_iter=7, random_state=42)
    P = svd.fit_transform(P)


    e_net.fit(np.dot(Sigma, Sigma), P)
    e, e_vec = np.linalg.eig(P)

    V = e_net.coef_
    print("Coefficient Size", V.shape )
    # V = V[~np.all(V == 0, axis=1)]
    Temp_proj =[]
    Temp_proj_test = []
    for pc in V:
        Temp_proj.append( np.array([np.dot(X[p,:], pc) for p in xrange(X.shape[0])]) )
        Temp_proj_test.append( np.array([np.dot(X1[p,:], pc) for p in xrange(X1.shape[0])]) )

    return np.transpose(np.array(Temp_proj)), np.transpose(np.array(Temp_proj_test))



def classification(X, y, model, iterate):
    model = model.logistic_regression(X, y, classes, inputs, iterate)
    return model


import random
from sklearn.metrics import mean_squared_error
N, y_train, T, y_test = import_pickled_data(dataset)

from random import randint
from tqdm import tqdm
model = None

for p in xrange(1):
    # Transform the train data-set
    scaler = preprocessing.StandardScaler(with_mean = True, with_std = True).fit(N)
    X_train = scaler.transform(N)
    
    # transform the test data-set
    temp_scalar = preprocessing.StandardScaler(with_mean = True, with_std = True).fit(T)
    X_test = temp_scalar.transform(T)

    for k in tqdm(xrange(1)):
            # Dimension reduction
            # 1 - Nonlinear Dimension reduction
            from distanceHDR import dim_reduction, dim_reduction_test
            Level, Train = dim_reduction(X_train, i_dim = X_train.shape[1], o_dim = 5, g_size= 2)
            print("Dimension reduced shape", Train.shape)
            Test = dim_reduction_test(X_test, Level, i_dim=X_train.shape[1], o_dim= 5, g_size=2)
            print("Dimension reduced shape", Test.shape)
            # Classification
            inputs = Train.shape[1]
            classes = int(max(y_train))
            y = tflearn.data_utils.to_categorical((y_train-1), classes)

            # Lets start with creating a model and then train batch wise.
            model = reg(inputs, classes)

            # classification
            model = classification(Train, y, model, iterate = 1000)

            # Error from the classification
            final_scores = (np.dot(Train, model.regression["weight"]))
            preds = softmax(final_scores)


            acc = accuracy(preds, y)
            print("Accuracy is", acc)

        # except Exception as e:
        #     print("Exception is", e)
        #     del model
        #     gc.collect()
        #     continue
