import svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm as sk_svm
from sklearn.metrics import confusion_matrix, classification_report


def data_generator(data_type='separable', n=200):
    # data_type = 'soft'
    # data_type = 'circle'
    n1, n2, n3 = n, n, n
    if data_type == 'circle':
        c1 = np.random.random((n1, 2)) - 0.5
        angle, radius = 2 * np.pi * np.random.random(n2), 1.5
        c2 = radius * np.c_[np.cos(angle), np.sin(angle)] + (np.random.random((n2, 2)) - 0.5)
        angle, radius = 2 * np.pi * np.random.random(n3), 3.0
        c3 = radius * np.c_[np.cos(angle), np.sin(angle)] + (np.random.random((n3, 2)) - 0.5)
    elif data_type == 'soft':
        c1 = np.random.random((n1, 2)) + np.array([-0.45, 0.20])
        c2 = np.random.random((n2, 2)) + np.array([0.45, 0.20])
        c3 = np.random.random((n2, 2)) + np.array([0., -0.70])
    else:
        c1 = np.random.random((n1, 2)) + np.array([-1., 1.])
        c2 = np.random.random((n2, 2)) + np.array([1., 1.])
        c3 = np.random.random((n3, 2)) + np.array([0., -1.])
    return c1, c2, c3, n1, n2, n3, data_type


def multi_test(clf, test_data=data_generator()):
    c1, c2, c3, n1, n2, n3, data_type = test_data
    x, y = np.r_[c1, c2, c3], np.array([0.]*n1 + [1.]*n2 + [2.]*n3)
    clf.fit(x, y)
    predicted_y = clf.predict(x)
    # print predicted_y
    target_test = y
    # target_predict = [clf.predict(sample) for sample in data_train]
    print("\n\nClassification report for classifier %s:\n%s\n"
          % (clf, classification_report(target_test, predicted_y)))
    print("Confusion matrix:\n%s" % confusion_matrix(target_test, predicted_y))

# data_type = 'separable'
# data_type = 'soft'
data_type = 'circle'
data = data_generator(data_type, n=500)
c1, c2, c3 = data[0], data[1], data[2]
plt.scatter(c1[:,0], c1[:,1], marker='o')
plt.scatter(c2[:,0], c2[:,1], marker='x')
plt.scatter(c3[:,0], c3[:,1], marker='+')
plt.show()

if data_type != 'circle':
    clf = svm.MultiSVM(kernel='linear', alg='SMO', C=1.0)
    multi_test(clf, data)
    multi_test(sk_svm.SVC(kernel='linear', C=1.0), data)
else:
    clf = svm.MultiSVM(kernel='rbf', alg='SMO', C=1.0)
    multi_test(clf, data)
    multi_test(sk_svm.SVC(kernel='rbf', C=1.0), data)
