import tensorflow as tf
import numpy as np
from helpers import *

class One2OneInputLayer(object):
    # One to One Mapping!
    def __init__(self, input):
        """
            The second dimension of the input,
            for each input, each row is a sample
            and each column is a feature, since 
            this is one to one mapping, n_in equals 
            the number of features
        """
        n_in = input.get_shape()[1].value
        
        self.input = input
        
        # Initiate the weight for the input layer
        r = 4*np.sqrt(3.0/n_in)
        w = tf.Variable(tf.random_uniform([n_in,],-r, r), name='w')
        
        self.w = w
        self.output = self.w * self.input
        self.params = [w]
    
class DenseLayer(object):
    # Canonical dense layer
    def __init__(self, input, init_w, init_b, activation='sigmoid'):
        """
            The second dimension of the input,
            for each input, each row is a sample
            and each column is a feature, since 
            this is one to one mapping, n_in equals 
            the number of features
            
            n_out defines how many nodes are there in the 
            hidden layer
        """

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
    def __init__(self, input, n_out, y):
        """
            The second dimension of the input,
            for each input, each row is a sample
            and each column is a feature, since 
            this is one to one mapping, n_in equals 
            the number of features
            
            n_out defines how many nodes are there in the 
            hidden layer
        """
        n_in = input.get_shape()[1].value
        self.input = input

        # Initiate the weight and biases for this layer
        w = tf.Variable(tf.random_normal([n_in, n_out]), name='w')
        b = tf.Variable(tf.random_normal([n_out]), name='b')

        pred = tf.add(tf.matmul(input, w), b)
        ################
        temp = tf.nn.softmax(pred)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        self.y = y
        self.w = w
        self.b = b
        self.cost = cost
        ###############
        self.temp = temp
        self.params= [w]