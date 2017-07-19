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

## Data In
def Low_Rank_Approx(r, X, Sigma):
    # U, S_1, V, E = sparse_pca(Sigma,n_comp , x, maxit=100,\
    # method='lars', n_jobs=1, verbose=0)
    # T = np.dot(U,np.diag(S_1))

    if r < X.shape[1]:
        from sklearn.linear_model import ElasticNet
        e_net = ElasticNet(alpha= 0.1, copy_X=False, fit_intercept=False, l1_ratio=0.7,
        max_iter=1000, normalize=True, positive=False, precompute=False,
        random_state=None, selection='cyclic', tol=0.000001, warm_start=False)

        P = np.identity(Sigma.shape[0])
        ## Define a low rank matri
        svd = TruncatedSVD(n_components=r, n_iter=7, random_state=42)
        P = svd.fit_transform(P)
        e_net.fit(np.dot(Sigma, Sigma), P)
        return e_net.coef_.reshape((r, Sigma.shape[0]))

    else:
        P, Q, R = np.linalg.svd(Sigma)
        return R.reshape([r, Sigma.shape[0]])


def dim_reduction(X, i_dim, o_dim, g_size):
    Level =[];
    # First check if the number of dimensions required are worthy of performing dimension reduction in the first place
    if (i_dim/float(g_size))< o_dim:
        print(" Too many components required, Proposed method cannot be used, doing PCA instead")
        Temp_proj =[]
        Level.append(level())
        Level[len(Level)-1].scaler = preprocessing.StandardScaler(with_mean = True, with_std = False).fit(X)
        D_scaled = Level[len(Level)-1].scaler.transform(X)

        temp_scalar = preprocessing.StandardScaler(with_mean = False, with_std = True).fit(X)
        D_scaled = temp_scalar.transform(D_scaled)

        # Get the dependency matrix for the group
        Sigma = dependence_calculation(D_scaled);
        V = Low_Rank_Approx(o_dim, X, Sigma)
        Level[len(Level)-1].group_transformation.append(V)
        for pc in V:
            Temp_proj.append( np.array([ np.dot(D_scaled[p,:], pc) for p in xrange(D_scaled.shape[0])]) )


        return Level, np.transpose( np.array(Temp_proj))

    prev = 0;
    while i_dim >= o_dim:
        print"i_dim", i_dim
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
        Temp_proj =[]
        eigenv = []
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
        for element in Level[len(Level)-1].G_LIST:
            temp = np.array(X[:, np.array(element)]).astype(float)

            Level[len(Level)-1].scaler.append(preprocessing.StandardScaler(with_mean = True, with_std = False).fit(temp))
            D_scaled = Level[len(Level)-1].scaler[len(Level[len(Level)-1].scaler)-1].transform(temp)

            temp_scalar = preprocessing.StandardScaler(with_mean = False, with_std = True).fit(temp)
            D_scaled = temp_scalar.transform(D_scaled)

            # Get the dependency matrix for the group
            Sigma = dependence_calculation(D_scaled)

            # Next achieve the parameters for transformation
            from scipy import linalg as LA
            e_vals, e_vecs = LA.eig(Sigma)

            # Sort both the eigen value and eigenvector in descending order
            arg_sort  = e_vals.argsort()[::-1][:]
            s_eigvals = e_vals[arg_sort]
            temp_sum = 0;
            temp_number = 0;
            s_eigvals = np.divide(s_eigvals, np.sum(s_eigvals));

            for eigen in s_eigvals:
                temp_number = temp_number+1;
                temp_sum = temp_sum + eigen;
                if temp_sum > 0.90:
                    break

            V = Low_Rank_Approx(temp_number, D_scaled, Sigma)
            # Finally get the eigen values and eigenvectors we are carrying
            # forward from this group
            Level[len(Level)-1].group_transformation.append(V)
            eigen_final.extend(s_eigvals[0:temp_number].astype(np.float).tolist());
            # Transform the required arrays
            for pc in V:
                Temp_proj.append( np.array([ np.dot(D_scaled[p,:], pc) for p in xrange(D_scaled.shape[0])]) )

        # Next prepare for the level transformaiton
        T = np.transpose( np.array(Temp_proj))
        p = np.divide(eigen_final, np.sum(eigen_final)).argsort()[::-1][:]
        T = T[:,p]
        # Get the next set of groupings and store the shuffling inside an array
        X, Level[len(Level)-1].level_shuffling = novel_groups(T, g_size);
        # I can start the next transformation
        i_dim = X.shape[1]
    return Level, Final


def dim_reduction_test(X, Level, i_dim, o_dim, g_size):
    # First check if the number of dimensions required are worthy of performing dimension reduction in the first place
    if (i_dim/float(g_size) <= o_dim):
        S = preprocessing.StandardScaler(with_std =True, with_mean = False).fit(X)
        D_scaled = Level[len(Level)-1].scaler.transform(X)
        D_scaled = S.transform(D_scaled)
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
        print("i_dim", i_dim)
        # For every element in the group generations, let evaluate our dimensions
        for j, element in enumerate(Level[p].G_LIST):
            print("the group is", element, " j is", j)
            temp = np.array(X[:, np.array(element)]).astype(float)
            temp_scalar = preprocessing.StandardScaler(with_std =True, with_mean = False).fit(temp)
            D_scaled = Level[p].scaler[j].transform(temp)
            D_scaled = temp_scalar.transform(D_scaled)
            # print("\nAt level", p, "The shape of the aray", D_scaled.shape)
            for pc in Level[p].group_transformation[j]:
                # print("Principal Components", pc.shape)
                Temp_proj.append( np.array([ np.dot(D_scaled[k,:], pc) for k in xrange(D_scaled.shape[0])]) )

        # print("Level shuffling is", Level[p].level_shuffling)
        T = np.transpose(np.array(Temp_proj))
        # print "T shape is", T.shape
        X = T[:,Level[p].level_shuffling]
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
