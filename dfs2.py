import tensorflow as tf
from nncomponents import *
from helpers import *
from sda import StackedDenoisingAutoencoder
import numpy as np

class DeepFeatureSelectionNew:
    def __init__(self, X_train, X_test, y_train, y_test, weight_init='sda', n_input = 2, hidden_dims=[1000], activation='sigmoid',epochs=1000,
                 lambda1=0.001, lambda2=1.0, alpha1=0.001, alpha2=0.0, learning_rate=0.1, optimizer='FTRL', print_step=1000, tolerance = 5):
        # Initiate the input layer
        
        # Get the dimension of the input X
        n_sample, n_feat = X_train.shape
        n_classes = len(np.unique(y_train))
        
        self.epochs = epochs
        self.n_input = n_input
        self.print_step = print_step
        self.tolerance = tolerance
        
        # Store up original value
        self.X_train = X_train
        self.y_train = one_hot(y_train)
        self.X_test = X_test
        self.y_test = one_hot(y_test)
        
        # Two variables with undetermined length is created
        self.var_X = tf.placeholder(dtype=tf.float32, shape=[None, n_feat], name='x')
        self.var_Y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='y')
        
        input_hidden = 0
        self.L1_input, self.L2_input = 0, 0
        # If there is no input layer
        if n_input != 0:
        # Create several one to one layers
            self.input_layers = []
            input_1to1 = self.var_X

            # regularization terms on coefficients of input layer
            L1_input, L2_input = [], []

            for i in xrange(n_input):
                self.input_layers.append(One2OneInputLayer(input_1to1))
                input_1to1 = self.input_layers[-1].output
                L1_input.append(tf.reduce_sum(tf.abs(self.input_layers[i].w)))
                L2_input.append(tf.nn.l2_loss(self.input_layers[i].w))

            input_hidden = self.input_layers[-1].output

            # Add it up
            self.L1_input = tf.add_n(L1_input)
            self.L2_input = tf.add_n(L2_input)

        else:
            input_hidden = self.var_X
        
        # Create list of hidden layers
        self.hidden_layers = []
        # Initialize the network weights
        weights, biases = init_layer_weight(hidden_dims, X_train, weight_init)
                
        # Create regularization terms on weights of hidden layers        
        L1s, L2_sqrs = [], []
        # Create hidden layers
        for init_w, init_b in zip(weights, biases):
            self.hidden_layers.append(DenseLayer(input_hidden, init_w, init_b, activation=activation))
            input_hidden = self.hidden_layers[-1].output
            L1s.append(tf.reduce_sum(tf.abs(self.hidden_layers[-1].w)))
            L2_sqrs.append(tf.nn.l2_loss(self.hidden_layers[-1].w))
        
        # Final classification layer, variable Y is passed
        self.softmax_layer = SoftmaxLayer(self.hidden_layers[-1].output, n_classes, self.var_Y)
           
        L1s.append(tf.reduce_sum(tf.abs(self.softmax_layer.w)))
        L2_sqrs.append(tf.nn.l2_loss(self.softmax_layer.w))

        self.L1 = tf.add_n(L1s)
        self.L2_sqr = tf.add_n(L2_sqrs)
        
        # Cost with two regularization terms
        self.cost = self.softmax_layer.cost \
                    + lambda1*(1.0-lambda2)*0.5*self.L2_input + lambda1*lambda2*self.L1_input \
                    + alpha1*(1.0-alpha2)*0.5 * self.L2_sqr + alpha1*alpha2*self.L1
        
        # FTRL optimizer is used to produce more zeros
        # self.optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate).minimize(self.cost)
        
        self.optimizer = optimize(self.cost, learning_rate, optimizer)
        
        self.accuracy = self.softmax_layer.accuracy

        self.y = self.softmax_layer.y
        ##################
        self.temp = self.softmax_layer.temp
        
    def train(self, batch_size=100):
        sess = tf.Session()
        self.sess = sess
        sess.run(tf.initialize_all_variables())
        batch_generator = GenBatch(self.X_train, self.y_train, batch_size)
        n_batch = batch_generator.n_batch

        self.losses, self.train_Accs, self.test_Accs = [], [], []
        for i in xrange(self.epochs):
            # x_batch, y_batch = get_batch(self.X_train, self.y_train, batch_size)
            batch_generator.resetIndex()
            for j in xrange(n_batch+1):
                x_batch, y_batch = batch_generator.get_batch()
                sess.run(self.optimizer, feed_dict={self.var_X: x_batch, self.var_Y: y_batch})

            self.train_Accs.append(sess.run(self.accuracy, \
                feed_dict={self.var_X: self.X_train, self.var_Y: self.y_train}))
            self.test_Accs.append(sess.run(self.accuracy, \
                feed_dict={self.var_X: self.X_test, self.var_Y: self.y_test}))
            self.losses.append(sess.run(self.cost, \
                feed_dict={self.var_X: x_batch, self.var_Y: y_batch}))

            # if np.abs(self.loss - prevLoss) < 0.001:
            #     count += 1
            #     prevLoss = self.loss

            if i % self.print_step == 0:
                print('epoch {0}: global loss = {1}'.format(i, self.losses[-1]))
                print("Train accuracy:", self.train_Accs[-1])
                print("Test accuracy:", self.test_Accs[-1])

            # if count >= self.tolerance:
            #     print("Program early terminated")
            #     break
        
        self.selected_ws = [sess.run(self.input_layers[i].w) for i in xrange(self.n_input)]
        # print("Input layer w: ", self.selected_ws)
        print("Final train accuracy:", self.train_Accs[-1])
        print("Final test accuracy:", self.test_Accs[-1])

    def refine_init_weight(self, threshold=0.001):
        refined_ws = [np.copy(w) for w in self.selected_ws]
        for i, refined_w in enumerate(refined_ws):
            refined_w[refined_w < threshold] = 0
            self.sess.run(self.input_layers[i].w.assign(refined_w))
        # self.input_layer.w.assign(refined_w)
        print("Test accuracy refined:",self.sess.run(self.accuracy, feed_dict={self.var_X: self.X_test, self.var_Y: self.y_test}))