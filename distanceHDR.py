"""
Testing For Paper-II
"""
# import all the required Libraries
import math
import numpy as np
import time, os, sys
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD


# Append all the path
sys.path.append('..//CommonLibrariesDissertation')
path_store = '../FinalDistSamples/'
## We have to now import our libraries
from Data_import import *
from Library_Paper_two  import *
paper = 'paper_2'
from scipy import linalg as LA
from scipy.sparse.linalg import eigsh
class level():
    def __init__(self):
        self.level_shuffling=[]
        self.group_transformation = []
        self.scaler= []
        self.G_LIST=[]
        self.flag = 0


def extract_samples(X, y, p):
    index_1= [i for i,v in enumerate(y) if v == p]
    N = X[index_1,:];
    return N


import SparsePCA
## Data In
def Low_Rank_Approx(r, Sigma):
    U, V= SparsePCA.sparse_pca(Sigma,r , 0.0001, maxit=10,\
    method='lars', n_jobs=1, verbose=0)
    return (U.reshape(Sigma.shape[1],r))


def dim_reduction(X, i_dim, o_dim, g_size):
    Level =[];
    # First check if the number of dimensions required are worthy of performing dimension reduction in the first place
    if (i_dim/float(g_size))< o_dim:
        Temp_proj =[]
        Level.append(level())
        Level[len(Level)-1].scaler.append(preprocessing.StandardScaler(with_mean = False, with_std = True).fit(X))
        D_scaled = Level[len(Level)-1].scaler[0].transform(X)
        Sigma = dependence_calculation(D_scaled)
        V = Low_Rank_Approx(o_dim, Sigma)
        Level[len(Level)-1].group_transformation.append(V)
        for pc in np.transpose(V):
            Temp_proj.append( np.array([ np.dot(D_scaled[p,:], pc) for p in xrange(D_scaled.shape[0])]) )
        return Level, np.transpose( np.array(Temp_proj))
    i_len = X.shape[0]
    ## if the group wise reduction is larger than the required number of dimensions
    prev = 0
    while i_dim >= o_dim:
        # Stopping conditions
        if (i_dim/float(g_size)) < o_dim:
            Final = X[:,0:o_dim]
            break
        elif prev == i_dim and Level[len(Level)-1].flag == 1:
            Final = X[:,0:prev]
            print("cannot reduce beyond this")
            break
        # Initilize for the first level
        Level.append(level())

        # Define the initial arrays for our calculation
        Temp_proj =np.zeros([i_len,1])
        if prev == i_dim:
            Level[len(Level)-1].flag = 1
        prev = i_dim;
        # First create all the groups
        for i in xrange(0, i_dim, g_size):
            if (i+g_size) < i_dim and (i+2*g_size) > i_dim:
                F = i_dim;
            else:
                F = i+g_size;
            if F <= i_dim:
                Level[len(Level)-1].G_LIST.append([j for j in xrange(i,F)])
        if len(Level[len(Level)-1].G_LIST) == 0:
            break
        eigen_final = [];
        start = time.time()
        for element in Level[len(Level)-1].G_LIST:
            temp = np.array(X[:, np.array(element)]).astype(float)
            Level[len(Level)-1].scaler.append(preprocessing.StandardScaler(with_mean = True, with_std = False).fit(temp))
            D_scaled = Level[len(Level)-1].scaler[len(Level[len(Level)-1].scaler)-1].transform(temp)
            # Get the dependency matrix for the group
            Sigma = dependence_calculation(D_scaled)
            # Next achieve the parameters for transformation
            e_vals = LA.eigvals(Sigma)
            # Sort both the eigen value and eigenvector in descending order
            arg_sort  = e_vals.argsort()[::-1][:]
            s_eigvals = e_vals[arg_sort]
            s_eigvals = np.divide(s_eigvals, float(np.sum(s_eigvals)));
            tempsum = np.cumsum(s_eigvals)
            mask = tempsum>=0.90
            temp_number = ((len(tempsum)-len(tempsum[mask]))+1)
            V = Low_Rank_Approx(temp_number, Sigma)
            Temp_proj = np.column_stack([Temp_proj, D_scaled.dot(V)])
            # Finally get the eigen values and eigenvectors we are carrying
            # forward from this group
            Level[len(Level)-1].group_transformation.append(V)
            eigen_final.extend(e_vals[0:temp_number].astype(np.float).tolist());
            # Transform the required arrays
        # Next prepare for the level transformaiton
        T = Temp_proj[:,1:Temp_proj.shape[1]]
        pre_shuffle = np.divide(eigen_final, np.sum(eigen_final)).argsort()[::-1][:]
        T = T[:,pre_shuffle]
        # Get the next set of groupings and store the shuffling inside an array
        X, t_shuffling= novel_groups(T, g_size)
        Level[len(Level)-1].level_shuffling.append(pre_shuffle)
        Level[len(Level)-1].level_shuffling.append(t_shuffling)
        # I can start the next transformation
        i_dim = X.shape[1]
    return Level, Final
def dim_reduction_test(X, Level, i_dim, o_dim, g_size):
    # First check if the number of dimensions required are worthy of performing dimension reduction in the first place
    if (i_dim/float(g_size) <= o_dim):
        # S = preprocessing.StandardScaler(with_std =True, with_mean = False).fit(X)
        D_scaled = Level[len(Level)-1].scaler[0].transform(X)
        # D_scaled = S.transform(D_scaled)
        # Transform the required arrays
        Temp_proj = []
        for pc in np.transpose(Level[len(Level)-1].group_transformation[0]):
            Temp_proj.append( np.array([ np.dot(D_scaled[p,:], pc) for p in xrange(D_scaled.shape[0])]) )
        print("Too many components required, Proposed method cannot be used, doing PCA instead")
        return (np.transpose(np.array(Temp_proj)))
    p = 0;
    while p <len(Level):
        # print("The level is", p, "length is", len(Level[p].group_transformation))
        # Define the initial arrays for our calculation
        Temp_proj =[]
        eigenv = []
        # For every element in the group generations, let evaluate our dimensions
        for j, element in enumerate(Level[p].G_LIST):
            temp = np.array(X[:, np.array(element)]).astype(float)
            # temp_scalar = preprocessing.StandardScaler(with_std =True, with_mean = False).fit(temp)
            D_scaled = Level[p].scaler[j].transform(temp)
            # print("\nAt level", p, "The shape of the aray", D_scaled.shape)
            for pc in np.transpose(Level[p].group_transformation[j]):
                # print("Principal Components", pc.shape)
                Temp_proj.append( np.array([ np.dot(D_scaled[k,:], pc) for k in xrange(D_scaled.shape[0])]) )

        # print("Level shuffling is", Level[p].level_shuffling)
        T = np.transpose(np.array(Temp_proj))
        # print "T shape is", T.shape
        X = T[:,Level[p].level_shuffling[0]]
        X = X[:,Level[p].level_shuffling[1]]
        i_dim = X.shape[1]
        p = p+1;

    if Level[len(Level)-1].flag is not 1:
        return X[:,0:o_dim]
    elif Level[len(Level)-1].flag is 1:
        return X

from Lib_new_group import dependence_calculation
#  The new group reduction methodology
def novel_groups(T, g_size):
    R_label = [i for i in xrange(T.shape[1])]
    T_label = []
    start = 0;
    step = int(len(R_label)/float(g_size));
    for start in xrange(len(R_label)):
        T_label.extend( [ R_label[i] for i in xrange(start,int(len(R_label)),step)] )
        if(len(T_label)>=len(R_label)):
            break
    return np.array(T[:, T_label]), T_label


#  The new group reduction methodology
def new_groups(T, g_size):
    # Next let us create the new set of groups since we passed the last two checks
    R_label = [i for i in xrange(T.shape[1])]
    T_label = []
    while len(R_label) > g_size:
        start = (len(R_label)/2);
        T_label.extend([ R_label[i] for i in\
         xrange( (start-(g_size/2)), (start+(g_size/2)) ) ])
        R_label = list(set(R_label)-set(T_label))
    if len(R_label) <= g_size:
        T_label.extend(R_label)
    # My next time groupings are all done now
    return np.array(T[:, T_label])
