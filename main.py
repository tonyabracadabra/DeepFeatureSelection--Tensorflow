from sklearn import datasets
from sklearn.cross_validation import train_test_split
from scipy import io as sio
from tensorflow.python.framework import ops
from dfs import DeepFeatureSelection
from dfs2 import DeepFeatureSelectionNew
import numpy as np

# ourdataB = sio.loadmat("/home/REGENERON/xupeng.tong/newDataB_2labels.mat")

# # ourdataB = sio.loadmat("/Users/xupeng.tong/Documents/Data/OriginalData/ourdataB.mat")


inputX = ourdataB['X']
print inputY.shape

inputY = ourdataB['Y'].reshape([inputX.shape[0],])
print inputY.shape

columnNames = ourdataB['columnNames']

digits = datasets.load_digits()

inputX = digits.data

inputY = digits.target


X_train, X_test, y_train, y_test = train_test_split(inputX, inputY, test_size=0.2, random_state=42)

ops.reset_default_graph()

E_adam = []
for i in xrange(10):
	dfsMLP = DeepFeatureSelectionNew(X_train, X_test, y_train, y_test, n_input=0, hidden_dims=[4000],weight_init='mlp',epochs=500,optimizer='Adam')
	dfsMLP.train(batch_size=100)
	eliminated = np.where(abs(dfsMLP.selected_w)==0)
	E_adam.append(eliminated)

np.save("E_adam", np.array(E_adam))

E_ftrl = []
for i in xrange(10):
	dfsMLP = DeepFeatureSelection(X_train, X_test, y_train, y_test, hidden_dims=[4000],weight_init='mlp',epochs=500,optimizer='FTRL')
	dfsMLP.train(batch_size=100)
	eliminated = np.where(abs(dfsMLP.selected_w)==0)
	E_ftrl.append(eliminated)

np.save("E_ftrl", np.array(E_adam))
