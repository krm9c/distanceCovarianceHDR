"""
Testing For Paper-II
"""
# import all the required Libraries
import math
import numpy as np
from scipy import linalg as LA
import time, os, sys
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
import Paper_res_v1

# Append all the path
sys.path.append('..//CommonLibrariesDissertation')
path_store = '../FinalDistSamples/'
## We have to now import our libraries
from Data_import import *
from Library_Paper_two  import *
from Library_Paper_one import *
paper = 'paper_2'

import time
def print_matrix(S):
    L = []
    for element in S:
        string ="\t"
        for element1 in element:
            element1 = round(element1,2)
            string = string +str(element1)+"\t"
        # print "{0:.2f}".format()
        print(string).format()
def test_distance_covariance(D_scaled):
    start = time.time()
    Sigma = dependence_calculation(D_scaled)
    print("Time", time.time()-start)
    e, v = LA.eig(Sigma)
    print_matrix(Sigma)
    print(e)
    S = np.corrcoef(D_scaled.T)
    e, v = LA.eig(S)
    print_matrix(S)
    print(e)

dataset = "rolling"
No, y_train, T, y_test = Paper_res_v1.import_pickled_data(dataset)
No = No
T  = T
# Transform the train data-set
scaler = preprocessing.StandardScaler(with_mean = True,\
 with_std = True).fit(No)
X_train = scaler.transform(No)
X_test = scaler.transform(T)
test_distance_covariance(No)
