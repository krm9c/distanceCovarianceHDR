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
from sklearn.decomposition import TruncatedSVD, dict_learning
import tflearn

import math
from SparsePCA import *
import gc
from sklearn.linear_model import LogisticRegression
from random import random
import sklearn


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
        self.regression["lr"] = 0.001
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
        self.regression["lr"] = 0.001
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
        # print 'Accuracy from scratch: {0}'.format(accuracy(preds, Y))
        return self

dataset = 'rolling'
## Data In
def dimension_Reduced(x, X, X1):
    # U, S_1, V, E = sparse_pca(Sigma,n_comp , x, maxit=100,\
    # method='lars', n_jobs=1, verbose=0)
    # T = np.dot(U,np.diag(S_1))
    Sigma = np.corrcoef(np.transpose(X))
    from sklearn.linear_model import ElasticNet
    e_net = ElasticNet(alpha= x, copy_X=True, fit_intercept=False, l1_ratio=0.7,
      max_iter=1000, normalize=True, positive=False, precompute=False,
      random_state=None, selection='cyclic', tol=0.000001, warm_start=False)
    e_net.fit(np.dot(Sigma, Sigma),Sigma)
    V = e_net.coef_
    # V = V[~np.all(V == 0, axis=1)]
    Temp_proj =[]
    Temp_proj_test = []
    for pc in V:
        Temp_proj.append( np.array([np.dot(X[p,:], pc) for p in xrange(X.shape[0])]) )
        Temp_proj_test.append( np.array([np.dot(X1[p,:], pc) for p in xrange(X1.shape[0])]) )
    return np.transpose(np.array(Temp_proj)), np.transpose(np.array(Temp_proj_test))


def classification(X, y, model):
    model = model.logistic_regression(X, y, classes, inputs, 200)
    final_scores = (np.dot(X_train, model.regression["weight"]))
    preds = softmax(final_scores)
    # print 'Accuracy from scratch: {0}'.format(accuracy(preds, y))
    return model


import random
from sklearn.metrics import mean_squared_error
N, y_train, T, y_test = import_pickled_data(dataset)
Temp_sparsity_initial =[]
Temp_sparsity =[]
Temp_accuracy =[]
mom = 0.0
v = 0.0
l1 = 0.001
beta_1 = 0.01
beta_2 = 0.0001
from random import randint
from tqdm import tqdm
for k in tqdm(xrange(100)):
    scaler = preprocessing.StandardScaler(with_mean = True, with_std = True).fit(N)
    X_train = scaler.transform(N)
    temp_scalar = preprocessing.StandardScaler(with_mean = True, with_std = True).fit(T)
    X_test = temp_scalar.transform(T)
    x = random.random()
    # Classification
    inputs = X_train.shape[1]
    classes = int(max(y_train))
    y = tflearn.data_utils.to_categorical((y_train-1), classes)

    # Lets start with creating a model and then train batch wise.
    model = reg(inputs, classes)
    Temp_sparsity_initial.append(x)
    for i in tqdm(xrange(500)):

        scaler = preprocessing.StandardScaler(with_mean = True, with_std = True).fit(N)
        X_train = scaler.transform(N)

        scaler = preprocessing.StandardScaler(with_mean = True, with_std = True).fit(T)
        X_test = scaler.transform(T)

        try:
            # Dimension reduction
            X_train, X_test = dimension_Reduced(x, X_train, X_test)
            # classification
            model = classification(X_train, y, model)
            ## Error from the classification
            final_scores = (np.dot(X_train, model.regression["weight"]))
            preds = softmax(final_scores)

            acc = accuracy(preds, y)

            ## Calculate momentum and variance parameters
            mse = sklearn.metrics.mean_squared_error(y, preds)
        except Exception as e:
            del model
            gc.collect()
            continue


        # mse = np.sum(y-preds, axis=1)
        # mse = np.dot(mse,np.sum(y,axis=1))
        # U, S, V = np.linalg.svd(X_train)
        # m_sum = np.sum(S)
        # loss = np.asscalar(mse + m_sum)
        regul = 0.0001
        #cost = -log((1-mse)*x)+reg*np.linalg.norm(x)
        loss = (1-mse)*x+regul*(np.linalg.norm(x)-1)
        # x = x-l1*(1-mse)
        mom = beta_1*mom + (1-beta_1)*loss
        v   = beta_2*v+(1-beta_2)*(loss*loss)
        mom = mom/(float(1-beta_1)+0.0000001)
        v=v/(float(1-beta_2)+0.000000001)

        # print("momentum is", mom, "bias is", v)
        # Tune the sparsity parameter
        x = x  - (l1*(mom/float(sqrt(v)+0.0000001)));

        if (x < 0.1):
            x = 0.1
        elif(x >0.984584):
            x =0.9

        # print("i -- ", i, "x is", x, "cost is", cost, "loss is", loss)
        # decay learning rates
        l1 = 0.99*l1
        beta_1 = 0.99*beta_1
        beta_2 = 0.99*beta_2

    Temp_sparsity.append(x)
    Temp_accuracy.append(acc)
    del model
    gc.collect()

print("sparsity length", len(Temp_sparsity))
print("sparsity initial length", len(Temp_sparsity_initial))
print("Accuracy length", len(Temp_accuracy))
plt.plot(Temp_sparsity, Temp_accuracy)
plt.ylabel('Accuracy')
plt.xlabel('Sparsity')
plt.show()
np.savetxt("Sparsity.csv", np.array(Temp_sparsity))
np.savetxt("Sparsity_initial.csv", np.array(Temp_sparsity_initial))
np.savetxt("accuracy.csv", np.array(Temp_accuracy))
plt.plot(Temp_sparsity_initial, Temp_sparsity)
plt.ylabel('Initial')
plt.xlabel('Final')
plt.show()

    # for i in xrange(1):
    #     model = classification(X_train, y, model)
