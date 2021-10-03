import numpy as np 
import tensorflow as tf 

def hello():
    print("Listen to Gracie Abrams' Feels Like now")

def compute_distance_two_loop(x_train , x_test):
    """
    Helps on compute_distance_two_loops : 
    This function just implicitly loop through the training set and test set 
    to compute SQUARED EUCLID distance between each training examples and tes
    examples

    Inputs : 
    x_train : [num_train x H x W x C] tensor , contain information of num_train 
    training examples
    x_test  : [num_test x H x W x C] tensor , contain information of num_test test 
    examples

    Returns : 
    dist    : [num_train x num_test] tensor , where the entry dist[i][j] stores 
    distance of train[i] and test[j]
    
    """

    dist  = np.zeros

    for (i , train_ex) in enumerate(x_train):
        for(j , test_ex) in enumerate(x_test):
            dist[i][j] = sum((train_x - test_x)**2) 
   
    return dist[i][j]

def compute_distance_one_loop(x_train , x_test):
    """
    Helps on compute_distance_one_loop : 
    This is the vectorized version of compute_distance_two loops , this one 
    will only loop through x_train and subtract it from the vector of test set
    and return the result on dist

    Inputs : 
    x_train : [num_train x H x W x C] tensor , contain information of num_train 
    training examples
    x_test  : [num_test x H x W x C] tensor , contain information of num_test test 
    examples

    Returns : 
    dist    : [num_train x num_test] tensor , where the entry dist[i][j] stores 
    distance of train[i] and test[j]
    """

    pass
def compute_distances_no_loops(x_train , x_test):


class KnnClassifier
    def __init__(self , x_train , y_train)
        self.x_train = x_train
        self.y_train = y_train

    def predict(self , x_test , k = 1 )
    
    def check_accuracy(self , x_test , y_test) 


def knn_cross_validate(x_train , y_train , x_test , y_test , kfolds = None , k_choices)

























