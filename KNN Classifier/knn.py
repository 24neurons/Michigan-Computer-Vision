import numpy as np 
import tensorflow as tf 

def hello():
    print("Listen to Gracie Abrams' Feels Like now")

def compute_distance_two_loops(x_train , x_test):
    """
    Helps on compute_distance_two_loops : 
    This function just implicitly loop through the training set and test set 
    to compute SQUARED EUCLID distance between each training examples and tes
    examples . Images should be flattened out and treated as vector

    Inputs : 
    x_train : [num_train x H x W x C] tensor , contain information of num_train 
    training examples . dtype = 'int8'
    x_test  : [num_test x H x W x C] tensor , contain information of num_test test 
    examples , dtype = 'int8'

    Returns : 
    dist    : [num_train x num_test] numpy array, where the entry dist[i][j] stores 
    distance of train[i] and test[j]
    
    """
    num_train = x_train.shape[0]
    num_test  = x_test.shape[0]

    x_train   = tf.reshape(x_train , shape = (num_train , -1))
    x_test    = tf.reshape(x_test  , shape = (num_test  , -1))

    dist      = np.zeros((num_train , num_test) , dtype =  'float32')

    for (i , train_ex) in enumerate(x_train):
        for(j , test_ex) in enumerate(x_test):
            dist[i][j] =np.sqrt(np.sum( (train_ex - test_ex)**2 ) ) 
   
    return dist

def compute_distance_one_loop(x_train , x_test):
    """
    Helps on compute_distance_one_loop : 
    This is the vectorized version of compute_distance_two loops , this one 
    will only loop through x_train and subtract it from the vector of test set
    and return the result on dist

    Inputs : 
    x_train : [num_train x H x W x C] tensor , dtype = 'uint8'
    x_test  : [num_test x H x W x C] tensor , contain information of num_test, 
    dtype = 'int8'


    Returns : 
    dist    : [num_train x num_test] tensor , where the entry dist[i][j] stores 
    distance of train[i] and test[j]
    """

    num_train = x_train.shape[0]
    num_test  = x_test.shape[0]

    x_train   = tf.reshape(x_train , (x_train.shape[0] , -1))
    x_test    = tf.reshape(x_test  , (x_test.shape[0]  , -1))
    dist      = np.zeros((num_train , num_test) , dtype = 'float32')


    for (i_train , x_train_ ) in x_train:
        dist[i] = np.sqrt( np.sum( (x_train - x_test)**2  , axis = 1) )
    return dist

def compute_distances_no_loops(x_train , x_test):
    """
    This function computes distance between x_train and x_test by vectorization.
    Both input will be multiplied with a intermediate matrix to press them down.


    Inputs :
    x_train : [num_train x H x W x C] tensor , dtype = 'uint8' 
    x_test  : [num_test x H x W x C] tensor , dtype = 'uint8'

    Returns : 
    dist    : [num_train x num_test] tensor , dtype = 'uint8'
    with dist[i][j] is the squared euclidian distance of train[i] and test[j]

    """

    num_train = x_train.shape[0]
    num_test  = x_test.shape[0]


    x_train   = tf.reshape(x_train , (x_train.shape[0] , -1))
    x_test    = tf.reshape(x_test  , (x_test.shape[0] ,  -1))
    dist      = tf.zeros((num_train , num_test) , dtype = x_train.dtype)
    img_size  = x_train.shape[1]

    inter_matrix = tf.ones((num_train , num_test , img_size) , dtype = x_train.dtype)

    train_to_matrix  = tf.transpose( tf.multiply(x_train , tf.transpose(inter_matrix , perm = [1 , 0 , 2])) , perm = [1 , 0, 2] )
    test_to_matrix   = tf.multiply(x_test , inter_matrix)

    dist             = tf.sqrt( tf.reduce_sum( (train_to_matrix - test_to_matrix)**2 , axis = 2 ) )
    return dist
    
"""
class KnnClassifier
    def __init__(self , x_train , y_train)
        self.x_train = x_train
        self.y_train = y_train

    def predict(self , x_test , k = 1 )
    
    def check_accuracy(self , x_test , y_test , k = 1)

        y_pred = self.predict(x_test , k)

        return (y_pred == y_test).nonzero()
        



def knn_cross_validate(x_train , y_train , x_test , y_test , kfolds = None , k_choices)

"""























