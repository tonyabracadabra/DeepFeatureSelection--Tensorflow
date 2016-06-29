from sklearn import datasets, metrics
from tensorflow.contrib import learn

classifier = learn.TensorFlowLinearClassifier(n_classes=3)
