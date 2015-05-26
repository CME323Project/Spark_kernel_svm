from sklearn import svm
from sklearn.datasets import load_svmlight_file
import numpy as np

X_train, y_train = load_svmlight_file("../data/a8a.txt")
X_train = X_train[1:1000]
y_train = y_train[1:1000]
clf = svm.SVC(C = 0.001, gamma = 0.5, kernel = 'rbf')
clf.fit(X_train, y_train)
y_out = clf.predict(X_train) 
print np.sum(y_train==y_out), y_out.size
