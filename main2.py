from sklearn import datasets
from sklearn.cross_validation import train_test_split
from scipy import io as sio
from tensorflow.python.framework import ops
from dfs import DeepFeatureSelection
from dfs2 import DeepFeatureSelectionNew
import numpy as np
from sklearn.preprocessing import normalize

ourdataB = sio.loadmat("/home/REGENERON/xupeng.tong/newDataB_2labels.mat")

ourdataB = sio.loadmat("/Users/xupeng.tong/Documents/Data/OriginalData/newDataB_2labels.mat")

# digits = datasets.load_digits()

# inputX = digits.data  
# inputY = digits.target

inputX = ourdataB['X'][:,:1000]
inputX = normalize(inputX, axis=0)
# inputY = ourdataB['Y'].reshape([inputX.shape[0],])
inputY = ourdataB['Y'][0,:]
columnNames = ourdataB['columnNames']

X_train, X_test, y_train, y_test = train_test_split(inputX, inputY, test_size=0.2, random_state=42)

ops.reset_default_graph()

E_adam = []
# for i in xrange(10):

dfsMLP = DeepFeatureSelectionNew(X_train, X_test, y_train, y_test, n_input=0, hidden_dims=[500],weight_init='mlp', learning_rate = 0.01, \
								 epochs=100000, lambda1=0.01, lambda2=0.01, alpha1=0.01, alpha2=0.01, optimizer='Adam', print_step=100)
dfsMLP.train(batch_size=2000)
eliminated = np.where(abs(dfsMLP.selected_w)==0)
E_adam.append(eliminated)

np.save("E_adam_small1000", np.array(E_adam))

# E_ftrl = []
# for i in xrange(10):
# 	dfsMLP = DeepFeatureSelection(X_train, X_test, y_train, y_test, n_input=0, hidden_dims=[3000],weight_init='mlp', \
# 								     epochs=100000, lambda1=0, lambda2=0, alpha1=0, alpha2=0.0, optimizer='FTRL')
# 	dfsMLP.train(batch_size=150)
# 	eliminated = np.where(abs(dfsMLP.selected_w)==0)
# 	E_ftrl.append(eliminated)

# np.save("E_ftrl", np.array(E_adam))