
__author__ = 'Xin'
import numpy as np
from sklearn.linear_model import Lasso
from sklearn import preprocessing

class RandomizedLasso(object):
    """using coordinate descent lasso and combine several lasso models like random forest
    Parameters
    ----------
    alpha : float, optional
        Constant that multiplies the L1 term. Defaults to 1.0.
    q1: int
        select q1 variables randomly in "Generate"
    q2: int
        select q2 viriables randomly in "Select"
    X: training data
    y: target
    B: int
        bootstrap samples
    """
    def __init__(self,q1,q2,X,y,B,alpha=1):
        self.q1,self.q2 = q1 , q2
        self.alpha = alpha
        self.y = y
        self.n_samples,self.n_features = X.shape
        self.n_bootsrap = B
        self.X = preprocessing.scale(X) #center data
        self.importance = np.zeros(self.n_features)
        self.coef = np.zeros(self.n_features)
        self.intercept = 0

    def lasso(self,training,target,feature_index_list):
        clf=Lasso(self.alpha,fit_intercept=False)
        clf.fit(training,target)
        coef=np.zeros(self.n_features)
        for index,feature_index in enumerate(feature_index_list):
            coef[feature_index]=clf.coef_[index]
        return coef

    def bootstrap(self):
        sample_index=np.random.choice(self.n_samples, self.n_samples)
        return self.X[sample_index,:],self.y[sample_index]

    def random_select_features(self,X,mode):
        if mode=='Generate':
            feature_index_list=np.random.choice(self.n_features,self.q1)
        elif mode=='Select':
            feature_index_list=np.random.choice(self.n_features,self.q2,p=self.importance)
        return X[:,feature_index_list],feature_index_list

    def Generate(self):
        for count in range(self.n_bootsrap):
            X,y=self.bootstrap()
            train_x,feature_index_list=self.random_select_features(X,'Generate')
            coef=self.lasso(train_x,y,feature_index_list)
            self.importance+=coef
        self.importance/=self.n_bootsrap
        self.importance=np.absolute(self.importance)
        self.importance/=np.sum(self.importance)

    def Select(self):
        for count in range(self.n_bootsrap):
            X,y=self.bootstrap()
            train_x,feature_index_list=self.random_select_features(X,'Select')
            coef=self.lasso(train_x,y,feature_index_list)
            self.coef+=coef
        self.coef/=self.n_bootsrap
        self.intercept = np.mean(self.y) - np.dot(np.mean(self.X,axis=0), self.coef.T)

    def predict(self,train_x):
        train_x=preprocessing.scale(train_x)
        return np.dot(train_x,self.coef.T)+self.intercept