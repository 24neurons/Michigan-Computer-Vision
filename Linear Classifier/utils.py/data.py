import numpy as np 
import tensorflow as tf
import tensorflow_datasets as tfds

def extract_tensor(dset , num = None):
   """
   -Inputs : 
    dset : A tuple of both image and labels with 
    num  : Optional. If provided, return the first num examples


   -Returns : 
    X   : image examples with shape (num , C , H , W ) 
        dtype = tf.float32 
    y   : label respective to the image (num, )
        dtype = tf.int32
   """
    X = dset['image'] / 255
    y = dset['label'] 

    if num is not None:
    	X = X[:num]
    	y = y[:num]
    return X , y
def cifar10(num_train = None , num_test = None):

	"""
	This function will download the cifar10 dataset from tensorflow dataset and return 4 file:
	train image , train label, test image and test label

	-Inputs: 
	    num_train : Optional. If provided return num_train training examples
	    num_test  : Optional. If provided return num_test test examples
	-Returns: 
	    X_train   : Training image with shape : (num_train , C , H , W) , dtype = tf.float32
	    y_train   : Training labels, dtype = tf.int32
	    X_test    : Test image with shape (num_test , C , H , W), dtype = tf.int32
	    y_test    : Test labels, dtype = tf.int32

	"""
	train , test = tfds.load('cifar10' , split = ['train' , 'test'] , as_supervised = False , batch_size = -1)
	X_train , y_train = extract_tensor(train, num_train)
	X_test  , y_test  = extract_tensor(test , num_test)

	return X_train , y_train , X_test , y_test
 