This project implements the SMO algorithm for SVM in Python.

Author: Soloice.

Here are some instructions for the project:

Source code structure
---------------------
- All source codes are in the folder `src2/`.
- Two classes BinarySVM and MultiSVM are defined in the file `svm.py`.
- `demo_test.py`, `multi_test.py` and `svm_test.py` all used to debug the SMO algorithm: 
  * `demo_test.py` includes a data generator which generates 2-dimensional linear separable/almost-separable/circular data of 2 classes, then visualize the data points and train a BinarySVM.  
  * Similarly, `multi_test.py` serves for testing MultiSVM.  
  * In `svm_test.py`, some real data are extracted from the MNIST dataset and are visualized using the PCA technique.
- Finally, `svm_test_full.py` trains a SVM classifier on the whole MNIST data.


Performance and Observations
---------------------
In my experiment, I found training an SVM with 'RBF' kernel is much faster than that with linear kernel.  I don't why.  Perhaps it is because in RKHS the data points are more separable thus facilitates the training procedure.
For your reference, Training a MultiSVM classifier with 'RBF' kernel on 6/7 MNIST data (i.e., using 60k examples as the training set) takes 11462s on my workstation (32GB RAM, 1 CPU with 8 Intel(R) Xeon(R) CPU E5-1620 v2 @ 3.70GHz cores.)

Overall, the results can be summarized as follows:

| algorithm	| running time(s)	| average precision	| average recall	| average F1-score |
|:-------------:|:-------------:|:-------------:|:-------------:|:------------:|
| SMO + Linear Kernel	| 9684 + 12	| 0.91	| 0.91	| 0.91 |
| SMO + RBF Kernel	| 666 + 54	| 0.92	| 0.92	| 0.92 |
| QP + Linear Kernel	| 225 + 11	| 0.91	| 0.91	| 0.91 |
| QP + RBF Kernel	| 267 + 58	| 0.92	| 0.92	| 0.92 |
| Sklearn svm + Linear Kernel	| 12 + 118	| 0.92	| 0.92	| 0.92 |
| Sklearn svm + RBF Kernel	| 30 + 232	| 0.92	| 0.92	| 0.92 |
