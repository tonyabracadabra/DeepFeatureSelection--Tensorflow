#!/usr/bin/env python

__author__ = "Xupeng Tong"
__copyright__ = "Copyright 2016, Deep Feature Selection at Regeneron"
__email__ = "tongxupeng.cpu@gmail.com"

import tensorflow as tf
import numpy as np
from helpers import *

class One2OneInputLayer(object):
    """ One to One input layer

    Parameters
    ----------

    input: Tensor
        The output from the last layer
    weight_init:
        initial value for weights
    """
    # One to One Mapping!
    def __init__(self, input, weight_init=None):
        n_in = input.get_shape()[1].value
        
        self.input = input
        
        # Initiate the weight for the input layer
        r = 4*np.sqrt(3.0/n_in)

        if weight_init is None:
            self.w = tf.Variable(tf.random_uniform([n_in,],-r, r), name='w')
        else: 
            self.w = tf.Variable(weight_init, name='w')

        self.output = self.w * self.input
    
class DenseLayer(object):
    """ Canonical dense layer

    Parameters
    ----------

    input: Tensor
        The output from the last layer
    init_w: numpy array
        initial value for weights
    init_b: numpy array
        initial value for b
    """
    def __init__(self, input, init_w, init_b, activation='sigmoid'):

        n_in = input.get_shape()[1].value
        self.input = input

        # Initiate the weight for the input layer
        
        w = tf.Variable(init_w, name='w')
        b = tf.Variable(init_b, name='b')

        output = tf.add(tf.matmul(input, w), b)
        output = activate(output, activation)
        
        self.w = w
        self.b = b
        self.output = output
        self.params = [w]
        
class SoftmaxLayer(object):
    """ Softmax layer for classification

    Parameters
    ----------

    input: Tensor
        The output from the last layer
    n_out: int
        Number of labels
    y: numpy array
        True label for the data
    """
    def __init__(self, input, n_out, y):
        n_in = input.get_shape()[1].value
        self.input = input

        # Initiate the weight and biases for this layer
        r = 4*np.sqrt(6.0/(n_in + n_out))
        w = tf.Variable(tf.random_uniform([n_in, n_out], minval=-r, maxval=r))
        b = tf.Variable(tf.zeros([n_out]), name='b')

        pred = tf.add(tf.matmul(input, w), b)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        self.y = y
        self.w = w
        self.b = b
        self.cost = cost
        self.params= [w]