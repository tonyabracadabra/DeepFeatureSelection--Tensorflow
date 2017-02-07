if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from scipy import io as sio
from tensorflow.python.framework import ops
from supporting_files.dfs2 import DeepFeatureSelectionNew
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import normalize

ourdata = sio.loadmat("./data/Q_AsthmaCOPD_NoAgeSmoke.mat")

inputX = ourdata['X']
inputY = ourdata['Y'][0,:]
columnNames = ourdata['columnNames']

weights = []

for random_state in xrange(50):
    # Resplit the data
    X_train, X_test, y_train, y_test = train_test_split(inputX, inputY, test_size=0.2, random_state=random_state)
    
    # Change number of epochs to control the training time
    dfsMLP = DeepFeatureSelectionNew(X_train, X_test, y_train, y_test, n_input=1, hidden_dims=[5], learning_rate=0.01, \
                                         lambda1=0.001, lambda2=1, alpha1=0.001, alpha2=0, activation='tanh', \
                                         weight_init='uniform',epochs=100, optimizer='Adam', print_step=10)
    dfsMLP.train(batch_size=2000)
    print("Train finised for random state:" + str(random_state))
    weights.append(dfsMLP.selected_ws[0])

# The generated weights will be in the weights folder
np.save("./weights/Q_weights_AsthmaCOPD", weights)