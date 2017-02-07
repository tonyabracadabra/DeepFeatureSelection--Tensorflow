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


ourdata = sio.loadmat("../data/B_AsthmaAcos_mean_scaled_7159.mat")
inputX = ourdata['X']
inputY = ourdata['Y'][0,:]
columnNames = ourdata['columnNames']

index_Acos = np.where(inputY==0)[0]
index_Asthma = np.where(inputY==1)[0]

weights = []
for i in xrange(50):
    # made random choice of asthma patients
    choice = np.random.choice(a=len(index_Asthma), size=len(index_Acos))
    index_Asthma_chosen = index_Asthma[choice]

    # Concatenate the indexes for Asthma and Acos patients
    indexes = np.array(index_Acos.tolist()+index_Asthma_chosen.tolist())
    # Shuffle the indexes
    np.random.shuffle(indexes)
    indexes = indexes.tolist()

    # inputX and inputY for this round
    inputX_ = inputX[indexes,:]
    inputY_ = inputY[indexes]
    
    X_train, X_test, y_train, y_test = train_test_split(inputX_, inputY_, test_size=0.2)
    
    # Change number of epochs to control the training time
    dfsMLP = DeepFeatureSelectionNew(X_train, X_test, y_train, y_test, n_input=1, hidden_dims=[150], learning_rate=0.01, \
                                         lambda1=0.005, lambda2=1, alpha1=0.001, alpha2=0, activation='tanh', \
                                         weight_init='uniform',epochs=30, optimizer='Adam', print_step=10)
    dfsMLP.train(batch_size=500)
    print("Train finised for random state:" + str(i))
    weights.append(dfsMLP.selected_ws[0])
    
np.save("./weights/weights_AsthmaAcos_rerun", weights)