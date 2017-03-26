"""
Testing For Paper-II
"""
# import all the required Libraries
import math
import numpy as np
import time, os, sys
from tqdm import tqdm

# Append all the path
sys.path.append('/Users/krishnanraghavan/Documents/Research/CommonLibrariesDissertation')
path_store = '//Users/krishnanraghavan/Documents/Research/HierarchicalDimensionReduction/FinalDistSamples/'
## We have to now import our libraries
from Data_import import *
from Library_Paper_two  import *
paper = 'paper_2'

def extract_samples(X, y, p):
    index_1= [i for i,v in enumerate(y) if v == p]
    N = X[index_1,:];
    return N

def dim_reduction(X, i_dim, o_dim, g_size):
    print i_dim, o_dim, g_size

    while (i_dim) >= o_dim:
        # Define the initial arrays for our calculation
        Temp_proj =[]
        eigenv = []
        G_LIST = []
        # First create all the groups
        for i in xrange(0, i_dim, g_size):
            if (i+g_size) < i_dim and (i+2*g_size) > i_dim:
                F = i_dim;
            else:
                F = i+g_size;
            G_LIST.append([i for i in xrange(i,F)])
        if len(G_LIST) == 0:
            break
        # For every element in the group generations, let evaluate our dimensions
        for element in G_LIST:
            temp = np.array(X[:, np.array(element)]).astype(float)
            temp_scalar = preprocessing.StandardScaler(with_mean = True, with_std = True).fit(temp)

            D_scaled = temp_scalar.transform(temp)
            Sigma = np.corrcoef(np.transpose(D_scaled))

            from scipy import linalg as LA
            e_vals, e_vecs = LA.eig(Sigma)

            for eigen in e_vals:
                eigenv.append(eigen)
            for pc in e_vecs:
                Temp_proj.append( np.array( [ np.dot(D_scaled[p,:], pc) for p in xrange(D_scaled.shape[0]) ] ) )

        print "First done"
        T =np.transpose( np.array(Temp_proj))
        temp_scalar = preprocessing.StandardScaler(with_mean = False, with_std = True).fit(eigenv)
        e = temp_scalar.transform(eigenv)
        p = e.argsort()[::-1]
        T = T[:,p]
        T= T[:, 0:(i_dim/g_size)]
        i_dim = T.shape[1]
        print "Is the thing still working"
        if (i_dim/g_size) < o_dim:
            Final = T[:,0:o_dim]
            break
        if paper =='paper_2':
            X = new_group(T, g_size)
        i_dim = X.shape[1]
    print "The dimension reduced data is of shape", Final.shape
    return Final

#  The new group reduction methodology
def new_groups(T, g_size):
    # Next let us create the new set of groups since we passed the last two checks
    R_label = [i for i in xrange(T.shape[1])]
    print "R_label", R_label
    T_label = []
    while len(R_label) > g_size:
        start = (len(R_label)/2);
        print "start", start
        T_label.extend([ R_label[i] for i in\
         xrange( (start-(g_size/2)), (start+(g_size/2)) ) ])
        print T_label
        R_label = list(set(R_label)-set(T_label))
        print R_label
    if len(R_label) <= g_size:
        T_label.extend(R_label)
    # My next time groupings are all done now
    return np.array(T[:, T_label])



#  Distance calculation for just plain testing
def test_distance_calculation(X, y):

    # Start afresh again
    # Let us now test how our distance behaves
    # Get the first sample
    N = extract_samples(X, y, 1)
    print "1"
    Samp_Size = 10000
    print "2"
    rng = np.random.RandomState(0)
    rand = rng.randint(N.shape[0], size=Samp_Size)
    N = N[rand,:]
    scaler = preprocessing.StandardScaler().fit(N)
    N_transform = scaler.transform(N)
    print "3"
    Ref = dim_reduction(N_transform, i_dim=N_transform.shape[1], o_dim=10, g_size=3)
    print "4"
    # Get the other sample
    T = extract_samples(X, y, 4)
    rand = rng.randint(T.shape[0], size=Samp_Size)
    T = T[rand,:]
    T_transform = scaler.transform(T)
    Test = dim_reduction( T_transform, i_dim=T_transform.shape[1], o_dim=10, g_size=3)
    Tmp = traditional_MTS(Ref, Ref, par=1)
    import matplotlib.pyplot as plt
    plt.plot(Tmp)
    plt.show()



if __name__ == "__main__":
    ## Next let us figure out parsing the sensorless dataset
    X, y = DataImport(num=3)
    # First let us define the number of input dimensions
    i_dim = X.shape[1]
    # Next, we pick the total number of output dimensions in the data
    o_dim = 10
    # Next initialize the first group transformation
    g_size = 4;
    test_distance_calculation(X, y)
