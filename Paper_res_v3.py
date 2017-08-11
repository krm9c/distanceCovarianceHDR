import Paper_res_v1


from distanceHDR import dim_reduction, dim_reduction_test


# Lets do this, get the rolling element data in and run the logistic regression for it
from tqdm import tqdm
import os, sys
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
    acc = []
    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        clf.fit(X, y)
        acc.append(clf.score(XT, yT))
        labels = clf.predict(XT)
        i = int(max(labels)-1)
        p_value =0
        index = [p for p,v in enumerate(yT) if v == i]
        index = [ int(x) for x in index ]
        yT= [ int(x) for x in yT ]
        L = [v for p,v in enumerate(labels) if p not in index]
        p_value = ( (list(L).count(i)) )/float(len(labels));
        s.append(p_value)
    return np.array(s).reshape(1,9), np.array(acc).reshape(1,9)


dataset = "sensorless"
No, y_train, T, y_test = Paper_res_v1.import_pickled_data(dataset)
No = No
T  = T
# Transform the train data-set
scaler = preprocessing.StandardScaler(with_mean = True,\
 with_std = True).fit(No)
X_train = scaler.transform(No)
X_test = scaler.transform(T)
N = 1
sco = np.zeros((1,9))
acc = np.zeros((1,9))
n_comp = 290
g_size = 20
for i in tqdm(xrange(N)):
    #from distanceHDR import dim_reduction, dim_reduction_test
    # start = time.time()
    # Level, Train = dim_reduction(X_train[0:100,:], i_dim = X_train.shape[1], o_dim = n_comp, g_size=g_size)
    # print("Time", time.time()-start)
    # start = time.time()
    # Train = dim_reduction_test(X_train, Level, i_dim = X_train.shape[1], o_dim = n_comp, g_size=g_size)
    # Test = dim_reduction_test(X_test, Level, i_dim = X_train.shape[1], o_dim = n_comp, g_size=g_size)
    # print("Time", time.time()-start)
    sco[i,:], acc[i,:] = comparison_class(X_train, y_train, X_test, y_test)

print (dataset, No.shape, T.shape)
print("p-values are")
print(sco)
print("accuracies are")
print(acc)
