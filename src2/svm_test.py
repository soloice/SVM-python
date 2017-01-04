from __future__ import division
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn import svm as sk_svm
from sklearn.metrics import confusion_matrix, classification_report
import time
from sklearn.decomposition import PCA
import svm
import collections
import numpy as np
import matplotlib.pyplot as plt

mnist = fetch_mldata('MNIST original', data_home='../data')
print type(mnist.data)
print "shape (full):", mnist.data.shape, mnist.target.shape
mnist.data = np.array(mnist.data, dtype='float64') / 255.0
# mnist.data /= 255.0
# print mnist.data[0]
print type(mnist.data[0, 0])
# mnist.target = mnist.target.astype(int)

# choose 2 classes only
subset = 700
mnist.data = np.r_[mnist.data[:subset, :], mnist.data[7000:7000+subset,:]]
mnist.target = np.r_[mnist.target[:subset], mnist.target[7000:7000+subset]]
pca = PCA(n_components=2)
mnist.data = pca.fit_transform(mnist.data)
plt.scatter(mnist.data[:subset,0], mnist.data[:subset,1], marker='o')
plt.scatter(mnist.data[subset:,0], mnist.data[subset:,1], marker='x')
plt.show()
print mnist.data.shape, mnist.target.shape

data_train, data_test, target_train, target_test = train_test_split(mnist.data, mnist.target, train_size=6.0/7.0)
print data_train.shape, collections.Counter(target_train)
print data_test.shape, collections.Counter(target_test)

MY_SVM = 01
if MY_SVM:
    # clf = svm.BinarySVM(kernel='linear', alg='dual', C=1.0)
    clf = svm.MultiSVM(kernel='linear', alg='SMO', C=1.0)
    # clf = svm.MultiSVM(kernel='rbf', C=1.0)
else:
    clf = sk_svm.SVC(kernel='linear', decision_function_shape='ovr')

t1 = time.time()
print data_train.shape, target_train.shape
# print type(data_train[0,0])
# print data_train[0]
clf.fit(data_train, target_train)
t2 = time.time()
print "Training time: ", t2 - t1
target_predict = clf.predict(data_test)
t3 = time.time()
print "Predicting time: ", t3 - t2
print 'predicted classes:', target_predict

print("\n\nClassification report for classifier %s:\n%s\n"
      % (clf, classification_report(target_test, target_predict)))
print("Confusion matrix:\n%s" % confusion_matrix(target_test, target_predict))


print '\n\nFor training set!'
# target_predict = [clf.predict(sample) for sample in data_train]
target_predict = clf.predict(data_train)
print("\n\nClassification report for classifier %s:\n%s\n"
      % (clf, classification_report(target_train, target_predict)))
print("Confusion matrix:\n%s" % confusion_matrix(target_train, target_predict))
