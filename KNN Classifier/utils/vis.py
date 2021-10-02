import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt


"Utilities that help visualize a set of image"


def visualize(list_image_tensor, class_dict , sper_row):
    """
    This function receive a list of tensor ( M x N ) image tensor
    Then return just a figure where every images on a single have the same
    label

    Input : 
    list_image_tensor : [N x (tensor of image)] where N is size of batch
    class_dict        : name of classes
    s_per_row   : number of image per each row
    Return : 
    fig               : on the left is name of class, on the right is 
    s_per_row images displayed
    """

    fig , ax = plt.subplots(nrows = sper_row , ncols = len(class_dict) , sharex = True , sharey = True)
    plt.yticks([])
    plt.xticks([])

    image_half_height = list_image_tensor[0].shape[1]

    for (idxs , image) in enumerate(list_image_tensor):

        r_id = idxs / sper_row      # row id of it on the subplots
        c_id = idxs % sper_row      # col id of it on the subplots
        ax[r_id][c_id].imshow(image.numpy())

    for (class_row , class_name) in enumerate(class_dict):
        ax[class_row][0].text(-4 , image_half_height , class_name) 
        # this loop print the class name
        
    fig.tight_layout(pad = 0.001)

    return fig
def random_samples(X , y , class_dict , samples_per_row):
    """
    Randomly select samples_per_row for each class on the whole available dataset
    then visualize it

    Inputs : 

    X : a tensor of [N x H x W x C] shape where N is size of batch
    y : the paired label for X
    class_dict : the name of classes
    samples_per_row : the number of sample images we want to show for each class

    Returns : 
    sample_fig : a subplots of shape samples x samples that have the class name
    on the left
    """
    Samples = []

    for cur_y in range(len(class_dict)):
        idxs = (y == cur_y).nonzero()[0]
        choosen_idxs = np.random.choice(idxs , size = samples_per_row)

        choosen_Xs    =  tf.unstack(X[choosen_idxs])

        Samples.append(choosen_Xs)
    
    return visualize(Samples , class_dict , samples_per_row)



    
    

            




















