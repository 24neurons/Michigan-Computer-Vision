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
            dist[i][j] = np.sqrt(np.sum( (train_ex - test_ex)**2 ) ) 
   
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


    for (i, cur_img ) in enumerate(x_train):
        dist[i] =  np.sqrt( np.sum( (cur_img - x_test)**2  , axis = 1) )
    return dist

def compute_distance_no_loop(x_train , x_test):
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
def predict_label(distance , y_train , k = 1):
    """
    This function take the distance computed between x_train and x_test
    and take k neaest sample from x_train's label : y_train to assign to x_test

    Inputs : 
    Distance : [num_train x num_test] tensor
    y_train  : Paired label of x_train
    
    Returns  : 
    y_pred   : Predicted label for each sample/image on x_test

    """
    #tf.math.top_k returns the largest, so we need to set to negative to compute
    #the lowest 
    #tranpose because tf.math.top_k computes k highest on the last axis, so we
    #need to make test indexes on the last axis
    k_nearest_distance      = tf.math.top_k(-distance.transpose() , k).indices.numpy()
    k_nearest_label         = y_train.numpy()[k_nearest_distance]
   
   
    y_pred          = np.apply_along_axis(
        lambda x : np.bincount(x).argmax() , 
        axis = 1 , 
        arr = k_nearest_label
    )
    return y_pred 

class KnnClassifier:
    def __init__(self , x_train , y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self , x_test , k = 1 ):
        distance = compute_distance_no_loop(self.x_train , x_test)
        y_pred   = predict_label(distance , self.y_train)
        return y_pred
    def check_accuracy(self , x_test , y_test , k = 1):

        y_pred = self.predict(x_test , k)

        return (y_pred == y_test).numpy().nonzero()

def knn_cross_validate(x_train , y_train , num_folds = 5, k_list = [1 , 2, 3]):
    
    """
    This function implement cross validate algorithm :
    Divide them to num_folds, default = 5 folders, then train model on num_folds - 1
    folds, use the remaining fold to test performance 

    Inputs : 
    x_train : [num_train x H x W x C] tensor, contain the training information
    y_train : [num_test  x H x W x C]
    """
    x_train_fold = []
    y_train_fold = []
    num_train    = x_train.shape[0]
    frs_folds    = int (num_train / num_folds ) * (num_folds - 1 )

    #You can use tf.split here , but only if num_train % num_folds = 0 
    if num_train % num_folds == 0:
        x_train_fold = tf.split(x_train , num_folds)
        y_train_fold = tf.split(y_train , num_folds)
    else: 
        
        x_train_fold = tf.split(x_train[:frs_folds] , num_folds - 1)
        y_train_fold = tf.split(y_train[:frs_folds] , num_folds - 1)

        x_train_fold.append(x_train[frs_folds:])
        y_train_fold.append(y_train[frs_folds:]) 

    k_to_accuracy = {}
    
    for k in k_list:
        k_to_accuracy[k] = []
        for (fold_index , cv_x ) in enumerate(x_train_fold):

            front_X  = x_train_fold[:fold_index]
            front_y  = y_train_fold[:fold_index]

            back_X   = x_train_fold[fold_index:]
            back_y   = y_train_fold[fold_index:]

            cv_y     = y_train_fold[fold_index]

            if type(front_X) is not list:
                front_X  = [front_X]
                front_y  = [front_y]
            if type(back_X) is not list:
                back_X   = [back_X]
                back_y   = [back_y]

            train_X = tf.concat(front_X + back_X, axis = 0)
            train_y = tf.concat(front_y + back_y , axis = 0)
            
            sub_classifier = KnnClassifier(train_X , train_y)
            k_to_accuracy[k].append(sub_classifier.check_accuracy(cv_x , cv_y))
        return k_to_accuracy
        



        
        

























