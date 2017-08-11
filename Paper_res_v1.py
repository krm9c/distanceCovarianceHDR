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

from sklearn.decomposition import TruncatedSVD
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis



dataset = 'rolling'
from sklearn.metrics import accuracy_score
import random
from sklearn.metrics import mean_squared_error
from random import randint
from tqdm import tqdm



# Class
class reg():
    def __init__(self, inputs, classes):
        self.regression = {}
        self.regression["weight"] = weight_variable(inputs, classes)
        self.regression["bias"] = bias_variable(classes,1)
        self.regression["lr"] = 0.01
        self.feature_reduction = {}
        self.beta_11 = 0.001
        self.beta_22 = 0.01
        self.m_1 = np.zeros(self.regression["weight"].shape)
        self.v_1 = np.zeros(self.regression["weight"].shape)

    def update_weight(self, score, batch_xs, batch_ys, weight, bias, lr):
        grad   = 2*np.dot((score - batch_ys).T, batch_xs).T  # 2*0.001*weight
        grad_1 = np.dot(2*(score - batch_ys).T, bias)
        print(grad_1.shape)
        ## Calculate momentum and variance parameters
        self.m_1   = (self.beta_11*self.m_1+ (1-self.beta_11)*grad)
        self.v_1   = self.beta_22*self.v_1+   (1-self.beta_22)*np.square(grad)

        self.m_1 = (self.m_1/float(1-self.beta_11+0.000001))
        self.v_1 = (self.v_1/float(1-self.beta_22+0.000001))
        ## Tune the sparsity parameter
        # decay learning rates
        self.beta_11 = 0.99*self.beta_11
        self.beta_22 = 0.99*self.beta_22
        return (weight- lr* np.divide(self.m_1, (np.sqrt(self.v_1) + 0.000001) ) ), bias-lr*grad_1

    def logistic_regression(self, X, Y, classes, inputs, iterations):
        # self.regression["lr"] = 0.01
        for i in xrange(iterations):
            self.regression["lr"] = 0.99*self.regression["lr"]
            for batch in iterate_minibatches(X, Y, 200, shuffle=True):
                batch_xs, batch_ys = batch
                score = softmax(np.dot(batch_xs, self.regression["weight"]))
                self.regression["weight"], self.regression["bias"] = \
                self.update_weight(score, batch_xs, batch_ys, self.regression["weight"],\
                self.regression["bias"], self.regression["lr"])
            final_scores = (np.dot(X, self.regression["weight"])+self.regression["bias"])
            preds = softmax(final_scores)
            print 'Accuracy from scratch: {0}'.format(accuracy(preds, Y))

        return self

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

def return_dict(placeholder, List, model, batch_x, batch_y):
    S ={}
    for i, element in enumerate(List):
        S[placeholder[i]] = element
    S[model.Deep['FL_layer0']    ] = batch_x
    S[model.classifier['Target'] ] = batch_y
    return S

def Analytic_Regression(model, Xtr, ytr, Xte, yte, iterate):
    print("In regression")
    try:
        t = xrange(iterate)
        from tqdm import tqdm
        for i in tqdm(t):
            for batch in iterate_minibatches(Xtr, ytr, 256, shuffle=True):
                batch_xs, batch_ys  = batch
                # Gather Gradients
                grads = model.sess.run([ model.Trainer["grads"] ],
                feed_dict ={ model.Deep['FL_layer0'] : batch_xs, model.classifier['Target']: batch_ys })
                List = [g for g in grads[0]]

                # Apply gradients
                summary, _ = model.sess.run( [ model.Summaries['merged'], model.Trainer["apply_placeholder_op"] ], \
                feed_dict= return_dict( model.Trainer["grad_placeholder"], List, model, batch_xs, batch_ys) )

            if i % 10 == 0:
                summary, a  = model.sess.run( [model.Summaries['merged'], model.Evaluation['accuracy']], feed_dict={ model.Deep['FL_layer0'] : \
                Xte, model.classifier['Target'] : yte})
                print("i", i, "--", a)
    except Exception as e:
        print(e)

def classification(X, y, XT, yT, iterate, classes):
    # Lets start with creating a model and then train batch wise.
    inputs = X.shape[1];
    model = Network_class.Agent()
    model = model.init_NN_custom(classes, 0.01, [inputs], tf.nn.relu)
    Analytic_Regression(model, X, y, XT, yT, iterate)
    # model = model.logistic_regression(X, y, classes, inputs, iterate)
    return model

def Log_regression_our_method(X_train, X_test, y_train, y_test):
    for k in xrange(1):
            # Reduce dimensions in the data
            from distanceHDR import dim_reduction, dim_reduction_test
            Level, Train = dim_reduction(X_train, i_dim = X_train.shape[1], o_dim = 2, g_size=2)
            Test = dim_reduction_test(X_test, Level, i_dim = X_train.shape[1], o_dim = 2, g_size=2)
            y_train = y_train.reshape(-1)
            y_test = y_test.reshape(-1)


            # Classification
            inputs = Train.shape[1]
            classes = int(max(y_train))
            y = tflearn.data_utils.to_categorical(y_train-1, classes)
            yT = tflearn.data_utils.to_categorical(y_test-1, classes)
            # classification
            model = classification(Train, y, Test, yT, iterate = 400, classes = classes)

            
def generate_new_data(n_sam, n_fea, n_inf):
    X,y = make_classification(n_samples=n_sam, n_features=n_fea, n_informative=n_inf, n_redundant=(n_fea-n_inf),\
    n_classes=2, n_clusters_per_class=1, weights=None, flip_y=0.01, class_sep=2.0,\
    hypercube=True, shift=10.0, scale=1.0, shuffle=True, random_state= 9000)

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.25)
    return X_train, X_test, y_train, y_test

def comparison_class(X, y, XT, yT):
    names = ["Nearest Neighbors", "Linear SVM", "lda",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]
    classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    #SVC(gamma=2, C=1),
    LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None,\
     n_components=None, store_covariance=False, tol=0.0001),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()
    ]

    s =[]
    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        clf.fit(X, y)
        s.append(clf.score(XT, yT))
        # print("classifier", name, "score", clf.score(XT, yT))
    return names, np.array(s).reshape(1,9)

def dim_reduction_comparison(n_comp, g_size):

    dataset= 'sensorless'
    N, y_train, T, y_test = import_pickled_data(dataset)
    name_1 = ["PCA", "ISOMAP", "LLE", "FA", "KPCA"]
    dims =[PCA(n_components=n_comp, \
    copy=False, whiten=True, \
    svd_solver='auto', tol=0.00001, iterated_power='auto', random_state=None),

    Isomap(n_neighbors=n_comp, n_components=10, eigen_solver='auto',\
     tol=0, max_iter=None, path_method='auto', neighbors_algorithm='auto', n_jobs=1),

     LocallyLinearEmbedding(n_neighbors=5, \
     n_components=n_comp, reg=0.001, eigen_solver='auto', tol=1e-06, max_iter=100, \
     method='standard', hessian_tol=0.0001, modified_tol=1e-12, \
     neighbors_algorithm='auto', random_state=None, n_jobs=1),

    FactorAnalysis(n_components= n_comp, tol=0.01, \
    copy=True, max_iter=1000, noise_variance_init=None,\
     svd_method='randomized', iterated_power=3, random_state=0),

    KernelPCA(n_components= n_comp, kernel='linear', gamma=None, degree=3, \
    coef0=1, kernel_params=None, alpha=1.0, \
    fit_inverse_transform=False, eigen_solver='auto', tol=0, max_iter=None,\
    remove_zero_eig=False, random_state=None, copy_X=True, n_jobs=1),
    ]

    # Transform the train data-set
    scaler = preprocessing.StandardScaler(with_mean = True,\
     with_std = True).fit(N)
    X_train = scaler.transform(N)
    X_test = scaler.transform(T)


    Res = np.zeros(( (len(dims)+1),2) )
    N = 100
    p = 0
    for n, clf in zip(name_1, dims):
        scores = np.zeros((N,9));
        print("DR is", n)
        for i in tqdm(xrange(N)):
            Train = clf.fit_transform(X_train)
            Test =  clf.transform(X_test)
            names, scores[i,:] = comparison_class(Train, y_train, Test, y_test)

        np.savetxt(str(n)+".csv",scores)
        Res[p,:] = np.array([scores.mean(), scores.std()])
    p=p+1
    names.append("NDR")
    scores = np.zeros((N,9))
    print("DR is NDR")
    for i in tqdm(xrange(N)):
        #from distanceHDR import dim_reduction, dim_reduction_test
        Level, Train = dim_reduction(X_train, i_dim = X_train.shape[1], o_dim = n_comp, g_size=g_size)
        Test = dim_reduction_test(X_test, Level, i_dim = X_train.shape[1], o_dim = n_comp, g_size=g_size)
        names, scores[i,:] = comparison_class(Train, y_train, Test, y_test)
    Res[p,:] = np.array([scores.mean(), scores.std()])
    ##Log_regression_our_method(X_train, X_test, y_train, y_test)
    np.savetxt("Fin.txt",Res)

from sklearn.decomposition import PCA, FactorAnalysis, KernelPCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from distanceHDR import dim_reduction, dim_reduction_test

def computational_comparson():
    ## Lets define define an array that can define dimensions
    D =[ 800, 1000, 1600, 3200, 6400, 10000]
    name_1 = ["PCA", "ISOMAP", "LLE", "FA", "KPCA"]

    dims =[PCA(n_components=9, \
    copy=False, whiten=True, \
    svd_solver='auto', tol=0.00001, iterated_power='auto', random_state=None),

    # Isomap(n_neighbors=9, n_components=10, eigen_solver='auto',\
    #  tol=0, max_iter=None, path_method='auto', neighbors_algorithm='auto', n_jobs=1),
    #
    #  LocallyLinearEmbedding(n_neighbors=5, \
    #  n_components=9, reg=0.001, eigen_solver='auto', tol=1e-06, max_iter=100, \
    #  method='standard', hessian_tol=0.0001, modified_tol=1e-12, \
    #  neighbors_algorithm='auto', random_state=None, n_jobs=1),
    #
    # FactorAnalysis(n_components= 9, tol=0.01, \
    # copy=True, max_iter=1000, noise_variance_init=None,\
    #  svd_method='randomized', iterated_power=3, random_state=0),
    #
    # KernelPCA(n_components= 9, kernel='linear', gamma=None, degree=3, \
    # coef0=1, kernel_params=None, alpha=1.0, \
    # fit_inverse_transform=False, eigen_solver='auto', tol=0, max_iter=None,\
    # remove_zero_eig=False, random_state=None, copy_X=True, n_jobs=1),
    ]

    for element in D:
        X_train, X_test, y_train, y_test = generate_new_data( (1000+(element*2)), element, n_inf=4)
        start = time.time()

        # Transform the train data-set
        scaler = preprocessing.StandardScaler(with_mean = True,\
         with_std = True).fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        N = 2
        p = 0
        Time = np.zeros((N,len(dims)+1))
        for n, clf in zip(name_1, dims):
            scores = np.zeros((N,9));
            print("DR is", n)
            for i in tqdm(xrange(N)):
                start = time.time()
                Train = clf.fit_transform(X_train)
                Test =  clf.transform(X_test)
                names, scores[i,:] = comparison_class(Train, y_train, Test, y_test)
                Time[i,p] = time.time()-start
            p = p+1
            np.savetxt(str(n)+str(element)+"acc.csv",scores)
        print("The value of p after the first set", p)
        names.append("NDR")
        scores = np.zeros((N,9))
        for i in tqdm(xrange(N)):
            #from distanceHDR import dim_reduction, dim_reduction_test
            start = time.time()
            Level, Train = dim_reduction(X_train, i_dim = X_train.shape[1], o_dim = 4, g_size=2)
            Test = dim_reduction_test(X_test, Level, i_dim = X_train.shape[1], o_dim = 4, g_size=2)

            print("Train shape", Train.shape, "Test shape", Test.shape)
            names, scores[i,:] = comparison_class(Train, y_train, Test, y_test)
            Time[i,p] = time.time()-start
        np.savetxt(str(element)+"Time.csv",Time)

# Compare classifiers for a simple datasets
def comparson_dataset():
    ## Lets define define an array that can define dimensions
    D =[ 800, 1000, 1600, 3200, 6400, 10000]
    name_1 = ["PCA", "ISOMAP", "LLE", "FA", "KPCA"]

    dims =[PCA(n_components=9, \
    copy=False, whiten=True, \
    svd_solver='auto', tol=0.00001, iterated_power='auto', random_state=None),
    # Isomap(n_neighbors=9, n_components=10, eigen_solver='auto',\
    #  tol=0, max_iter=None, path_method='auto', neighbors_algorithm='auto', n_jobs=1),
    #
    #  LocallyLinearEmbedding(n_neighbors=5, \
    #  n_components=9, reg=0.001, eigen_solver='auto', tol=1e-06, max_iter=100, \
    #  method='standard', hessian_tol=0.0001, modified_tol=1e-12, \
    #  neighbors_algorithm='auto', random_state=None, n_jobs=1),
    #
    # FactorAnalysis(n_components= 9, tol=0.01, \
    # copy=True, max_iter=1000, noise_variance_init=None,\
    #  svd_method='randomized', iterated_power=3, random_state=0),
    #
    # KernelPCA(n_components= 9, kernel='linear', gamma=None, degree=3, \
    # coef0=1, kernel_params=None, alpha=1.0, \
    # fit_inverse_transform=False, eigen_solver='auto', tol=0, max_iter=None,\
    # remove_zero_eig=False, random_state=None, copy_X=True, n_jobs=1),
    ]

    for element in D:
        X_train, X_test, y_train, y_test = generate_new_data( (1000+(element*2)), element, n_inf=4)
        start = time.time()

        # Transform the train data-set
        scaler = preprocessing.StandardScaler(with_mean = True,\
         with_std = True).fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        N = 2
        p = 0
        Time = np.zeros((N,len(dims)+1))
        for n, clf in zip(name_1, dims):
            scores = np.zeros((N,9));
            print("DR is", n)
            for i in tqdm(xrange(N)):
                start = time.time()
                Train = clf.fit_transform(X_train)
                Test =  clf.transform(X_test)
                names, scores[i,:] = comparison_class(Train, y_train, Test, y_test)
                Time[i,p] = time.time()-start
            p = p+1
            np.savetxt(str(n)+str(element)+"acc.csv",scores)
        print("The value of p after the first set", p)
        names.append("NDR")
        scores = np.zeros((N,9))
        for i in tqdm(xrange(N)):
            #from distanceHDR import dim_reduction, dim_reduction_test
            start = time.time()
            Level, Train = dim_reduction(X_train, i_dim = X_train.shape[1], o_dim = 4, g_size=2)
            Test = dim_reduction_test(X_test, Level, i_dim = X_train.shape[1], o_dim = 4, g_size=2)

            print("Train shape", Train.shape, "Test shape", Test.shape)
            names, scores[i,:] = comparison_class(Train, y_train, Test, y_test)
            Time[i,p] = time.time()-start
        np.savetxt(str(element)+"Time.csv",Time)


def dim_reduction_comparison(dataset, n_comp, g_size):

    N, y_train, T, y_test = import_pickled_data(dataset)

    name_1 = ["PCA", "ISOMAP", "LLE", "FA", "KPCA"]
    dims =[PCA(n_components=n_comp, \
    copy=False, whiten=True, \
    svd_solver='auto', tol=0.00001, iterated_power='auto', random_state=None),

    Isomap(n_neighbors=n_comp, n_components=10, eigen_solver='auto',\
     tol=0, max_iter=None, path_method='auto', neighbors_algorithm='auto', n_jobs=1),

     LocallyLinearEmbedding(n_neighbors=5, \
     n_components=n_comp, reg=0.001, eigen_solver='auto', tol=1e-06, max_iter=100, \
     method='standard', hessian_tol=0.0001, modified_tol=1e-12, \
     neighbors_algorithm='auto', random_state=None, n_jobs=1),

    FactorAnalysis(n_components= n_comp, tol=0.01, \
    copy=True, max_iter=1000, noise_variance_init=None,\
     svd_method='randomized', iterated_power=3, random_state=0),

    KernelPCA(n_components= n_comp, kernel='linear', gamma=None, degree=3, \
    coef0=1, kernel_params=None, alpha=1.0, \
    fit_inverse_transform=False, eigen_solver='auto', tol=0, max_iter=None,\
    remove_zero_eig=False, random_state=None, copy_X=True, n_jobs=1),
    ]

    # Transform the train data-set
    scaler = preprocessing.StandardScaler(with_mean = True,\
     with_std = True).fit(N)
    X_train = scaler.transform(N)
    X_test = scaler.transform(T)

    N = 1
    for n, clf in zip(name_1, dims):
        scores = np.zeros((N,9));
        print("DR is", n)
        for i in tqdm(xrange(N)):
            Train = clf.fit_transform(X_train)
            Test =  clf.transform(X_test)
            names, scores[i,:] = comparison_class(Train, y_train, Test, y_test)
        np.savetxt(str(n)+str(dataset)+".csv",scores)
        print("score is", scores)

    scores = np.zeros((N,9))
    print("DR is NDR")
    for i in tqdm(xrange(N)):
        #from distanceHDR import dim_reduction, dim_reduction_test
        Level, Train = dim_reduction(X_train, i_dim = X_train.shape[1], o_dim = n_comp, g_size=g_size)
        Test = dim_reduction_test(X_test, Level, i_dim = X_train.shape[1], o_dim = n_comp, g_size=g_size)
        names, scores[i,:] = comparison_class(Train, y_train, Test, y_test)
    print(scores)
    np.savetxt("NDR"+str(dataset)+".csv",scores)

# dim_reduction_comparison("sensorless", 4, 2)
