import os, sys
sys.path.append('../CommonLibrariesDissertation')
from Library_Paper_two import *


import Network_class
import tensorflow as tf
import gzip, cPickle
import numpy as np


Train_batch_size = 256
Train_Glob_Iterations = 50
total_sparsity = 100
total_mean = 10
dataset = 'rolling'

###################################################################################
def import_pickled_data(string):
    f = gzip.open('../data/'+string+'.pkl.gz','rb')
    dataset = cPickle.load(f)
    X_train = dataset[0]
    X_test  = dataset[1]
    y_train = dataset[2]
    y_test  = dataset[3]
    return X_train, y_train, X_test, y_test

###################################################################################
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

###################################################################################
def return_dict(placeholder, List, model, batch_x, batch_y):

    S ={}
    for i, element in enumerate(List):
        S[placeholder[i]] = element
    S[model.Deep['FL_layer0']    ] = batch_x
    S[model.classifier['Target'] ] = batch_y
    return S


#####################################################################################
def Analyse_custom_Optimizer(X_train, y_train, X_test, y_test):

    import gc
    # Lets start with creating a model and then train batch wise.
    model = Network_class.Agent()
    model = model.init_NN_custom(classes, 0.001, [inputs, 500], tf.sigmoid)

    try:
        t = xrange(Train_Glob_Iterations)
        from tqdm import tqdm
        for i in t:
            for batch in iterate_minibatches(X_train, y_train, Train_batch_size, shuffle=True):
                batch_xs, batch_ys  = batch

                # Gather Gradients
                grads = model.sess.run([ model.Trainer["grads"] ],
                feed_dict ={ model.Deep['FL_layer0'] : batch_xs, model.classifier['Target']: batch_ys })
                List = [g for g in grads[0]]

                # Apply gradients
                summary, _ = model.sess.run( [ model.Summaries['merged'], model.Trainer["apply_placeholder_op"] ], \
                feed_dict= return_dict( model.Trainer["grad_placeholder"], List, model, batch_xs, batch_ys) )
                # model.Summaries['train_writer'].add_summary(summary, i)

            if i % 1 == 0:
                summary, a  = model.sess.run( [model.Summaries['merged'], model.Evaluation['accuracy']], feed_dict={ model.Deep['FL_layer0'] : \
                X_test, model.classifier['Target'] : y_test})
                # print("The accuracy is", a)
                # model.Summaries['test_writer'].add_summary(summary, i)
            if a > 0.99:
                summary, pr  = model.sess.run( [ model.Summaries['merged'], model.Evaluation['prob'] ], \
                feed_dict ={ model.Deep['FL_layer0'] : X_test, model.classifier['Target'] : y_test } )
                # model.Summaries['test_writer'].add_summary(summary, i)
    except Exception as e:
        print e
        print "I found an exception"
        tf.reset_default_graph()
        del model
        gc.collect()
        return 0

    tf.reset_default_graph()
    del model
    gc.collect()
    return a

Results=[]
from tqdm import tqdm
N, y_train, T, y_test = import_pickled_data(dataset)
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD, dict_learning
# K = np.random.randint(1, 10, (10,1))
n_comp = 6
K = np.random.rand(total_sparsity,1).reshape(-1)
scaler = preprocessing.StandardScaler(with_mean = True, with_std = True).fit(N)
X_train = scaler.transform(N)
# Get the dependency matrix for the group
Sigma = np.corrcoef(np.transpose(X_train));
for i in tqdm(xrange(total_sparsity)):
    scaler = preprocessing.StandardScaler(with_mean = True, with_std = True).fit(N)
    X_train = scaler.transform(N)
    temp_scalar = preprocessing.StandardScaler(with_mean = True, with_std = True).fit(T)
    X_test = temp_scalar.transform(T)

    x = np.asscalar(K[i])
    P = dict_learning(Sigma, n_components = n_comp, alpha = x, max_iter=100, tol=1e-08)
    if(P[1].shape[0] == n_comp):
        Trans = P[1]
    else:
        Trans = np.transpose(P[1])

    print("The transformaiton", Trans.shape)
    print("Shape of the array", X_train.shape, X_test.shape);
    Temp_proj =[]
    Temp_proj_test = []
    for pc in Trans:
        Temp_proj.append( np.array([ np.dot(X_train[p,:], pc) for p in xrange(X_train.shape[0])]) )
        Temp_proj_test.append( np.array([ np.dot(X_test[p,:], pc) for p in xrange(X_test.shape[0])]) )
    X_train = np.array(np.transpose(Temp_proj))
    X_test = np.array(np.transpose(Temp_proj_test))
    print("Shape of the array after reduction", X_train.shape, X_test.shape);
    print("Optimization started")
    inputs = X_train.shape[1]
    classes = int(max(y_train))
    import tflearn
    T_NNarray = []
    for i in tqdm(xrange(total_mean)):
        T_NNarray.append(Analyse_custom_Optimizer(X_train,\
         tflearn.data_utils.to_categorical((y_train-1), classes),\
          X_test, tflearn.data_utils.to_categorical((y_test-1), classes)))
    Results.append( sum(T_NNarray)/float(len(T_NNarray)) )

import matplotlib.pyplot as plt
P = np.argsort(np.array(K))
P = P.astype(int)
print len(Results), P
Res = np.array(Results).reshape(-1)
plt.plot(K[P], Res[P])
plt.ylabel('Accuracy')
plt.show()
