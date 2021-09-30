{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "data.ipynb",
      "provenance": [],
      "authorship_tag": "ABX9TyOU7azNFvMFE+f0TIIhjHzF",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Miku0204/Michigan-Computer-Vision/blob/main/KNN%20Classifier/utils/data.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "MeO1UX3t9v1N"
      },
      "source": [
        "'''\n",
        "Insert your github information to mount colab to github\n",
        "'''\n",
        "user_email = \n",
        "user_name  = \n",
        "\n",
        "!git config --global user.email {user_email}\n",
        "!git config --global user.name  {user_name}"
      ],
      "execution_count": 77,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "DqTqVdLZ9u3e"
      },
      "source": [
        "'''\n",
        "This is where we clone the repository to make changes from colab\n",
        "Note!!!\n",
        "Do not share the token to anyone\n",
        "'''\n",
        "git_token = \n",
        "repository = \n",
        "\n",
        "!git clone https://{git_token}@github.com/{user_name}/{repository}"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "tO6NcDvTAuhQ"
      },
      "source": [
        "import numpy as np\n",
        "import tensorflow as tf\n",
        "import tensorflow_datasets as tfds"
      ],
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "BrQdiRedEiCn"
      },
      "source": [
        "def extract_samples(dset , num = None):\n",
        "    '''\n",
        "    Convert the dset from tensorflow.data.Datasets to tensors and sample first\n",
        "    num samples\n",
        "\n",
        "    Input :\n",
        "    dset : Cifar10 datasets ( data  + label)\n",
        "    num  : number of samples we want to sample\n",
        "    Return : \n",
        "\n",
        "    X    : The image data\n",
        "           shape : (num x W x H x C)\n",
        "    y    : The label data\n",
        "           shape : (num , )\n",
        "\n",
        "    '''\n",
        "\n",
        "    X = tf.convert_to_tensor(dset[0] , name = 'data')\n",
        "    y = tf.convert_to_tensor(dset[1] , name = 'label')\n",
        "\n",
        "    return X[:num] , y[:num]"
      ],
      "execution_count": 67,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "CtVUVcB4GRmC"
      },
      "source": [
        "def data_loading(num_train , num_test):\n",
        "    '''\n",
        "    Note : Tensorflow datasets are often stored in the format : tf.dataset.Dataset\n",
        "\n",
        "    This function use tensorflow_datasets.load to load cifar10 into two dicts : \n",
        "    train_ , test_ \n",
        "\n",
        "    Input : \n",
        "    num_train : number of training samples\n",
        "    num_test  : number of test samples\n",
        "\n",
        "    Returns : \n",
        "\n",
        "    train_image : num_train x W x H x C\n",
        "    train_label : num_train \n",
        "    test_image  : num_test x W x H x C\n",
        "    test_label  : num_test\n",
        "\n",
        "    *** Terminology :  \n",
        "    C : Number of channels that a single image has \n",
        "    H : Height of an image\n",
        "    W : Width of an image      \n",
        "\n",
        "    Permutation : If you want to permute a single image Z to C x H x W, use\n",
        "    Z.transpose(2 , 1 , 0)\n",
        "\n",
        "    '''\n",
        "    train_ , test_ = tfds.load('cifar10' , split = ['train' , 'test'] , as_supervised=True , batch_size=-1 )\n",
        "  \n",
        "    train_image , train_label = extract_samples(train_ , num_train)\n",
        "    test_image  , test_label  = extract_samples(test_ , num_test)\n",
        "    \n",
        "    return train_image , train_label , test_image , test_label\n",
        "    \n",
        "    "
      ],
      "execution_count": 68,
      "outputs": []
    }
  ]
}