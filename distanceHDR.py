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
if __name__ == "__main__":
    print "hello let us start working on the second paper"
     # The inputs
     # Get the data
    ## 2 -- Next let us figure out parsing the sensorless dataset
    X, y = DataImport(num=12)
    C = 4
    print "data", X.shape
    print "labels", y.shape

    # First let us define the number of input dimensions
    i_dim = X.shape[1]
    # Next, we pick the total number of output dimensions in the data
    o_dim = 3
    
