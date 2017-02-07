#!/usr/bin/env python

from __future__ import print_function

__author__ = "Xupeng Tong"
__copyright__ = "Copyright 2016, Deep Feature Selection at Regeneron"
__email__ = "tongxupeng.cpu@gmail.com"

import tensorflow as tf
import numpy as np

def activate(layer, name):
    """ Activate one layer with specified activation function

    Parameters
    ----------
    layer: Tensor
        The layer to be activated
    name: string, with options "sigmoid", "softmax", "tanh", "relu" and "linear"
        The name of the activation function
    """

    if name == 'sigmoid':
        return tf.nn.sigmoid(layer)
    elif name == 'softmax':
        return tf.nn.softmax(layer)
    elif name == 'tanh':
        return tf.nn.tanh(layer)
    elif name == 'relu':
        return tf.nn.relu(layer)
    elif name == 'linear':
        return layer

def optimize(cost, learning_rate, optimizer):
    """ Optimize the cost

    Parameters
    ----------
    learning_rate: float32
        Learning rate for gradient descent
    name: string, with options "FTRL", "Adam", "SGD"
        The name of the optimization function
        Adam optimizer generally gives us the best result
    """

    optimizer = {'FTRL':tf.train.FtrlOptimizer, 'Adam':tf.train.AdamOptimizer, \
                 'SGD':tf.train.GradientDescentOptimizer}[optimizer]

    return optimizer(learning_rate=learning_rate).minimize(cost)

def one_hot(y):
    """ Generate the one hot representation of Y

    Parameters
    ----------
    y: numpy array
    """
    n_classes = len(np.unique(y))
    one_hot_Y = np.zeros((len(y), n_classes))
    for i,j in enumerate(y):
        one_hot_Y[i][j] = 1
        
    return one_hot_Y

def init_layer_weight(dims, X, name):
    """ Initialize the weights for layers, and return the initialized result

    Parameters
    ----------
    dims: list, with each element stands for the number of nodes in each layer

    name: string, with options "sda" and "uniform"
        The name of the initialization method
    """

    weights, biases = [], []
    if name == 'sda':
        from sda import StackedDenoisingAutoencoder
        sda = StackedDenoisingAutoencoder(dims=dims)
        sda._fit(X)
        weights, biases = sda.weights, sda.biases
    elif name == 'uniform':
        n_in = X.shape[1]
        for d in dims:
            r = 4*np.sqrt(6.0/(n_in+d))
            weights.append(tf.random_uniform([n_in, d], minval=-r, maxval=r))
            biases.append(tf.zeros([d,]))
            n_in = d
            
    return weights, biases
    
def get_random_batch(X, Y, size):
    """
    Alternative method of getting a random batch each time
    """
    assert len(X) == len(Y)
    a = np.random.choice(len(X), size, replace=False)
    return X[a], Y[a]

class GenBatch():
    """ The batch generator for training

    Parameters
    ----------
    X: numpy array
    Y: numpy array
    batch_size: int
    """
    def __init__(self, X, y, batch_size):
        self.X = X
        self.Y = y
        self.batch_size = batch_size
        self.n_batch = (len(X) / batch_size)
        self.index = 0

    def get_batch(self):
        """
        Get the next batch
        """
        batch_range = xrange(self.index, (self.index+1)*self.batch_size)
        if self.index == self.n_batch:
            batch_range = xrange(self.index, len(self.X))
        self.index += 1

        return self.X[batch_range], self.Y[batch_range]

    def resetIndex(self):
        self.index = 0