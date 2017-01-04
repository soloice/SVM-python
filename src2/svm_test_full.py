from __future__ import division
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn import svm as sk_svm
from sklearn.metrics import confusion_matrix, classification_report
import time
import svm
import collections
import numpy as np

mnist = fetch_mldata('MNIST original', data_home='../data')
print type(mnist.data)
print "shape (full):", mnist.data.shape, mnist.target.shape
mnist.data = np.array(mnist.data, dtype='float64') / 255.0
# print type(mnist.data[0, 0])
# print mnist.data[0]

training_data_portion = 6.0 / 7.0
data_train, data_test, target_train, target_test = train_test_split(mnist.data, mnist.target,
                                                                    train_size=training_data_portion)
print data_train.shape, collections.Counter(target_train)
print data_test.shape, collections.Counter(target_test)

kernel_, algorithm_ = 'rbf', 'SMO'
MY_SVM = 01
if MY_SVM:
    clf = svm.MultiSVM(kernel=kernel_, alg=algorithm_, C=1.0)
else:
    clf = sk_svm.SVC(kernel=kernel_, decision_function_shape='ovo')

t1 = time.time()
print data_train.shape, target_train.shape
clf.fit(data_train, target_train)
if MY_SVM:
    clf.save('-'.join([algorithm_, kernel_]) + '.pkl')
t2 = time.time()
print "Total Training time: ", t2 - t1
target_predict = clf.predict(data_test)
t3 = time.time()
print "Total Predicting time: ", t3 - t2
print 'predicted classes:', target_predict

print("\n\nClassification report for classifier %s:\n%s\n"
      % (clf, classification_report(target_test, target_predict)))
print("Confusion matrix:\n%s" % confusion_matrix(target_test, target_predict))


print '\n\nFor training set!'
target_predict = clf.predict(data_train)
print("\n\nClassification report for classifier %s:\n%s\n"
      % (clf, classification_report(target_train, target_predict)))
print("Confusion matrix:\n%s" % confusion_matrix(target_train, target_predict))
