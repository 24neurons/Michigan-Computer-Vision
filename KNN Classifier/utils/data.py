
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

def extract_samples(dset , num = None):
    '''
    Convert the dset from tensorflow.data.Datasets to tensors and sample first
    num samples

    Input :
    dset : Cifar10 datasets ( data  + label)
    num  : number of samples we want to sample
    Return : 

    X    : The image data
           shape : (num x W x H x C)
    y    : The label data
           shape : (num , )

    '''

    X = tf.convert_to_tensor(dset[0] , name = 'data')
    y = tf.convert_to_tensor(dset[1] , name = 'label')

    return X[:num] / 255, y[:num] / 255
def data_loading(num_train , num_test):
    '''
    Note : Tensorflow datasets are often stored in the format : tf.dataset.Dataset

    This function use tensorflow_datasets.load to load cifar10 into two dicts : 
    train_ , test_ 

    Input : 
    num_train : number of training samples
    num_test  : number of test samples

    Returns : 

    train_image : num_train x W x H x C
    train_label : num_train 
    test_image  : num_test x W x H x C
    test_label  : num_test

    *** Terminology :  
    C : Number of channels that a single image has 
    H : Height of an image
    W : Width of an image      

    Permutation : If you want to permute a single image Z to C x H x W, use
    Z.transpose(2 , 1 , 0)

    '''
    train_ , test_ = tfds.load('cifar10' , split = ['train' , 'test'] , as_supervised=True , batch_size=-1 )
  
    train_image , train_label = extract_samples(train_ , num_train)
    test_image  , test_label  = extract_samples(test_ , num_test)
    
    return train_image , train_label , test_image , test_label
