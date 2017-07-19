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
pickle_file = 'rolling'



# Initialize the parameters, graph and the final update laws
# hyper parameter setting
image_size = 28
batch_size = 256
valid_size = test_size = 10000
num_data_input = 11
num_hidden = 100
num_labels = 4
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
# # valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
# test_dataset, test_labels = reformat(test_dataset, test_labels)
# from sklearn import preprocessing
# train_dataset =  preprocessing.scale(train_dataset)
# valid_dataset =  preprocessing.scale(valid_dataset)
# test_dataset  =  preprocessing.scale(test_dataset)


print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)




from scipy.spatial.distance  import pdist, squareform

def distance_covariance(x,y):
        x = np.reshape(x, [-1, 1])
        y = np.reshape(y, [-1, 1])

        # calculate distance matrix first
        A = squareform(pdist(x, 'euclidean'))
        N = A.shape[0]
        one_n = np.ones((N,1))/N
        A = A -A.dot(one_n) -np.transpose(one_n).dot(A) +(np.transpose(one_n).dot(A)).dot(one_n)
        np.fill_diagonal(A, 0)

        # Second distance matrix
        B = squareform(pdist(y, 'euclidean'))
        N = B.shape[0]
        one_n = np.ones((N,1))/N
        B = B -B.dot(one_n) -np.transpose(one_n).dot(B) +np.transpose(one_n).dot(B).dot(one_n)
        np.fill_diagonal(B, 0)

        Temp1 = np.multiply(A, B)
        Temp2 = np.multiply(A, A)
        Temp3 = np.multiply(B, B)

        nu_xy = (1/float(x.shape[0]*y.shape[0]))*np.sum(Temp1)
        nu_xx = (1/float(x.shape[0]*x.shape[0]))*np.sum(Temp2)
        nu_yy = (1/float(y.shape[0]*y.shape[0]))*np.sum(Temp3)
        if nu_xx*nu_yy < 1e-5:
            return 1e-3
        else:
            t_cor = nu_xy/float(sqrt(nu_xx*nu_yy))
            return t_cor

def dependence_calculation(X):
    n  = X.shape[0];
    m  = X.shape[1];
    C  = np.zeros((m,m));
    rng = np.random.RandomState(0)
    idx = rng.randint(n, size=500)
    P = X[idx, :]
    T = " "
    for i in xrange(0,m):
        T ="\t"
        x = P[:,i]
        for j in xrange(0,m):
            y = P[:,j]
            C[i][j] = distance_covariance(x, y)
            T = T+"("+str(i)+","+str(j)+")"+str(C[i][j])+"\t"
        print(T, "\n")
    return C

from scipy import stats
# Gather the normal and the test samples from the data
N = extract_samples(train_dataset, train_labels, 1);
T = extract_samples(train_dataset, train_labels, 4);

temp_scalar = preprocessing.StandardScaler(with_mean = True, with_std = True).fit(N)
N = temp_scalar.transform(N)
T = temp_scalar.transform(T)
import time
start = time.time()


# # 1 -- Dimension reduction
# Ref, Tree = initialize_calculation(T = None, Data = Xtr_s, gsize = 2,\
# par_train = 0, output_dimension = 4)
# # Test, Tree = initialize_calculation(T = Tree, Data = Xte_s, gsize = 2,\
# # par_train = 1, output_dimension = 4)

# 2 - Second type of dimension reduction
from distanceHDR import dim_reduction, dim_reduction_test
start = time.time()
Level, Ref = dim_reduction(N, i_dim=N.shape[1], o_dim=5, g_size=2)
# print(stats.describe(Ref))
Test = dim_reduction_test(T, Level, i_dim=T.shape[1], o_dim=5, g_size=2)
print("The time elapsed is", time.time()-start)
print("\nRef", Ref.shape, "Test", Test.shape)
print("\nref-ref", traditional_MTS(Ref, Ref, 0).mean())
print("\nref-test", traditional_MTS(Ref, Test, 0).mean())
