"""
Testing For Paper-II
"""
# import all the required Libraries
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import gzip
import os, sys

# Append all the path
sys.path.append('..//CommonLibrariesDissertation')
path_store = '../FinalDistSamples/'
## We have to now import our libraries
from Data_import import *
from Lib_new_group  import *
from Library_Paper_one import traditional_MTS
paper = 'paper_2'
pickle_file = 'sensorless'



# Initialize the parameters, graph and the final update laws
# hyper parameter setting
image_size = 28
batch_size = 256
valid_size = test_size = 10000
num_data_input = 48
num_hidden = 100
num_labels = 11
act_f = "relu"
init_f = "uniform"
back_init_f = "uniform"
weight_uni_range = 0.01
back_uni_range = 0.1
lr = 0.01
num_layer = 5 #should be >= 3
num_steps = 5000

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, num_data_input)).astype(np.float32)/float(255)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    labels = labels.reshape((-1, num_labels)).astype(np.float32)
    return dataset, labels

## Extract samples from the data
def extract_samples(X, y, p):
    index_1= [i for i,v in enumerate(y) if v == p]
    N = X[index_1,:];
    return N

f = gzip.open('../data/'+pickle_file+'.pkl.gz','rb')
dataset = pickle.load(f)
train_dataset = dataset[0]
test_dataset  = dataset[1]
train_labels  = dataset[2]
test_labels   = dataset[3]
valid_dataset = dataset[1]
valid_labels  = dataset[3]

del f  # hint to help gc free up memory
# valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
from sklearn import preprocessing
train_dataset =  preprocessing.scale(train_dataset)
valid_dataset =  preprocessing.scale(valid_dataset)
test_dataset  =  preprocessing.scale(test_dataset)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


# 1 - Nonlinear Dimension reduction
from distanceHDR import dim_reduction, dim_reduction_test
Train = train_dataset
Test = test_dataset
print("New dimension reduction")
start = time.time()
Level, Train = dim_reduction(train_dataset, i_dim=train_dataset.shape[1], o_dim=20, g_size=2)
print("Time elapsed", start-time.time())
print("Dimension reduced shape", Train.shape)
## Linear Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
y_pred = lda.fit(Train, train_labels).predict(Train)
print("1 -- LDA")
from sklearn.metrics import accuracy_score
print (accuracy_score(train_labels, y_pred, normalize=True, sample_weight=None))

qda = QuadraticDiscriminantAnalysis(store_covariances=True)
y_pred = qda.fit(Train, train_labels).predict(Train)
print("2 -- QDA")
from sklearn.metrics import accuracy_score
print (accuracy_score(train_labels, y_pred, normalize=True, sample_weight=None))

from sklearn.naive_bayes import GaussianNB
print("3 -- GaussianNB")
gnb = GaussianNB()
y_pred = gnb.fit(Train, train_labels).predict(Train)
from sklearn.metrics import accuracy_score
print (accuracy_score(train_labels, y_pred, normalize=True, sample_weight=None))

from sklearn.svm import SVC
print("4 -- SVC")
clf = SVC(kernel="linear", C=0.025)
y_pred = clf.fit(Train, train_labels).predict(Train)
print (accuracy_score(train_labels, y_pred, normalize=True, sample_weight=None))

from sklearn.neighbors import KNeighborsClassifier
print("5 -- KNN")
clf = KNeighborsClassifier(3)
y_pred = clf.fit(Train, train_labels).predict(Train)
print (accuracy_score(train_labels, y_pred, normalize=True, sample_weight=None))



import os,sys
sys.path.append('../CommonLibrariesDissertation')
from Library_Paper_two import *
## 2 - Hierarchical Dimension Reduction
print("HDR");
start = time.time()
Train, Tree = initialize_calculation(T = None, Data = train_dataset, gsize = 2,\
par_train = 0, output_dimension = 20)
print("Time elapsed", start-time.time())
print("Dimension reduced shape", Train.shape)

## Linear Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
y_pred = lda.fit(Train, train_labels).predict(Train)
print("1 -- LDA")
from sklearn.metrics import accuracy_score
print (accuracy_score(train_labels, y_pred, normalize=True, sample_weight=None))

qda = QuadraticDiscriminantAnalysis(store_covariances=True)
y_pred = qda.fit(Train, train_labels).predict(Train)
print("2 -- QDA")
from sklearn.metrics import accuracy_score
print (accuracy_score(train_labels, y_pred, normalize=True, sample_weight=None))

from sklearn.naive_bayes import GaussianNB
print("3 -- GaussianNB")
gnb = GaussianNB()
y_pred = gnb.fit(Train, train_labels).predict(Train)
from sklearn.metrics import accuracy_score
print (accuracy_score(train_labels, y_pred, normalize=True, sample_weight=None))

from sklearn.svm import SVC
print("4 -- SVC")
clf = SVC(kernel="linear", C=0.025)
y_pred = clf.fit(Train, train_labels).predict(Train)
print (accuracy_score(train_labels, y_pred, normalize=True, sample_weight=None))

from sklearn.neighbors import KNeighborsClassifier
print("5 -- KNN")
clf = KNeighborsClassifier(3)
y_pred = clf.fit(Train, train_labels).predict(Train)
print (accuracy_score(train_labels, y_pred, normalize=True, sample_weight=None))



import os,sys
sys.path.append('../CommonLibrariesDissertation')
from Library_Paper_two import *
## 3 - No dimension reduction
print("Real");
Train = train_dataset
print("Dimension reduced shape", Train.shape)
## Linear Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
y_pred = lda.fit(Train, train_labels).predict(Train)
print("1 -- LDA")
from sklearn.metrics import accuracy_score
print (accuracy_score(train_labels, y_pred, normalize=True, sample_weight=None))

qda = QuadraticDiscriminantAnalysis(store_covariances=True)
y_pred = qda.fit(Train, train_labels).predict(Train)
print("2 -- QDA")
from sklearn.metrics import accuracy_score
print (accuracy_score(train_labels, y_pred, normalize=True, sample_weight=None))

from sklearn.naive_bayes import GaussianNB
print("3 -- GaussianNB")
gnb = GaussianNB()
y_pred = gnb.fit(Train, train_labels).predict(Train)
from sklearn.metrics import accuracy_score
print (accuracy_score(train_labels, y_pred, normalize=True, sample_weight=None))

from sklearn.svm import SVC
print("4 -- SVC")
clf = SVC(kernel="linear", C=0.025)
y_pred = clf.fit(Train, train_labels).predict(Train)
print (accuracy_score(train_labels, y_pred, normalize=True, sample_weight=None))

from sklearn.neighbors import KNeighborsClassifier
print("5 -- KNN")
clf = KNeighborsClassifier(3)
y_pred = clf.fit(Train, train_labels).predict(Train)
print (accuracy_score(train_labels, y_pred, normalize=True, sample_weight=None))
