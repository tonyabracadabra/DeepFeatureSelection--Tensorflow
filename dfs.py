import tensorflow as tf
from nncomponents import *
from helpers import *
from sda import StackedDenoisingAutoencoder

class DeepFeatureSelection:
    def __init__(self, X_train, X_test, y_train, y_test, weight_init='sda', hidden_dims=[100, 100, 100], epochs=1000,
                 lambda1=0.001, lambda2=1.0, alpha1=0.001, alpha2=0.0, learning_rate=0.1, optimizer='FTRL'):
        # Initiate the input layer
        
        # Get the dimension of the input X
        n_sample, n_feat = X_train.shape
        n_classes = len(np.unique(y_train))
        
        self.epochs = epochs
        
        # Store up original value
        self.X_train = X_train
        self.y_train = one_hot(y_train)
        self.X_test = X_test
        self.y_test = one_hot(y_test)
        
        # Two variables with undetermined length is created
        self.var_X = tf.placeholder(dtype=tf.float32, shape=[None, n_feat], name='x')
        self.var_Y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='y')
        
        self.input_layer = One2OneInputLayer(self.var_X)
        
        self.hidden_layers = []
        layer_input = self.input_layer.output
        
        # Initialize the network weights
        weights, biases = init_layer_weight(hidden_dims, X_train, weight_init)
        
        print(type(weights[0]))
        
        # Create hidden layers
        for init_w,init_b in zip(weights, biases):
            self.hidden_layers.append(DenseLayer(layer_input, init_w, init_b))
            layer_input = self.hidden_layers[-1].output
        
        # Final classification layer, variable Y is passed
        self.softmax_layer = SoftmaxLayer(self.hidden_layers[-1].output, n_classes, self.var_Y)
    
        n_hidden = len(hidden_dims)
        
        # regularization terms on coefficients of input layer 
        self.L1_input = tf.reduce_sum(tf.abs(self.input_layer.w))
        self.L2_input = tf.nn.l2_loss(self.input_layer.w)
        
        # regularization terms on weights of hidden layers        
        L1s = []
        L2_sqrs = []
        for i in xrange(n_hidden):
            L1s.append(tf.reduce_sum(tf.abs(self.hidden_layers[i].w)))
            L2_sqrs.append(tf.nn.l2_loss(self.hidden_layers[i].w))
            
        L1s.append(tf.reduce_sum(tf.abs(self.softmax_layer.w)))
        L2_sqrs.append(tf.nn.l2_loss(self.softmax_layer.w))

        self.L1 = tf.add_n(L1s)
        self.L2_sqr = tf.add_n(L2_sqrs)
        
        # Cost with two regularization terms
        self.cost = self.softmax_layer.cost \
                    + lambda1*(1.0-lambda2)*0.5*self.L2_input + lambda1*lambda2*self.L1_input \
                    + alpha1*(1.0-alpha2)*0.5 * self.L2_sqr + alpha1*alpha2*self.L1
        
        # FTRL optimizer is used to produce more zeros
#         self.optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate).minimize(self.cost)
        
        self.optimizer = optimize(self.cost, learning_rate, optimizer)
        
        self.accuracy = self.softmax_layer.accuracy

        self.y = self.softmax_layer.y
        
    def train(self, batch_size=100):
        sess = tf.Session()
        self.sess = sess
        sess.run(tf.initialize_all_variables())
        
        for i in xrange(self.epochs):
            x_batch, y_batch = get_batch(self.X_train, self.y_train, batch_size)
            sess.run(self.optimizer, feed_dict={self.var_X: x_batch, self.var_Y: y_batch})
            if i % 2 == 0:
                l = sess.run(self.cost, feed_dict={self.var_X: x_batch, self.var_Y: y_batch})
                print('epoch {0}: global loss = {1}'.format(i, l))
                self.selected_w = sess.run(self.input_layer.w)
                print("Train accuracy:",sess.run(self.accuracy, feed_dict={self.var_X: self.X_train, self.var_Y: self.y_train}))
                print("Test accuracy:",sess.run(self.accuracy, feed_dict={self.var_X: self.X_test, self.var_Y: self.y_test}))
                print(self.selected_w)
                print(len(self.selected_w[self.selected_w==0]))
        print("Final test accuracy:",sess.run(self.accuracy, feed_dict={self.var_X: self.X_test, self.var_Y: self.y_test}))
    
    def refine_init_weight(self, threshold=0.001):
        refined_w = np.copy(self.selected_w)
        refined_w[refined_w < threshold] = 0
        self.sess.run(self.input_layer.w.assign(refined_w))
        print("Test accuracy refined:",self.sess.run(self.accuracy, feed_dict={self.var_X: self.X_test, self.var_Y: self.y_test}))