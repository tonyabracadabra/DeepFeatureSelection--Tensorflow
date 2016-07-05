from sklearn.linear_model import RandomizedLasso
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from scipy import io as sio
from tensorflow.python.framework import ops
from dfs2 import DeepFeatureSelectionNew
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import normalize

# ourdataB = sio.loadmat("/Volumes/TONY/Regeneron/Data/OriginalData/newDataB_2labels.mat")
ourdataB = sio.loadmat("/Users/xupeng.tong/Documents/Data/OriginalData/newDataB_2labels.mat")
# ourdataB = sio.loadmat("/home/REGENERON/xupeng.tong/newDataB_2labels.mat")

inputX = ourdataB['X']
inputX = normalize(inputX, axis=0)
inputY = ourdataB['Y'][0,:]
columnNames = ourdataB['columnNames']

X_train, X_test, y_train, y_test = train_test_split(inputX, inputY, test_size=0.2, random_state=42)

randomized_lasso = RandomizedLasso()
randomized_lasso.fit(X_train, y_train)

featureMask = randomized_lasso.get_support()

X_train_lasso = X_train[:,featureMask]
X_test_lasso = X_train[:,featureMask]

columnNames[0][:100][featureMask]

sio.savemat('RandomLasso-result', {'X_train_lasso':X_train_lasso, \
			'X_train_lasso':X_test_lasso, 'featureMask':featureMask})