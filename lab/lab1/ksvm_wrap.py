from sklearn import svm
import numpy as np
from lab.lab1.data import *
from lab.lab1.tf_deep import TFDeep
import tensorflow as tf

class KSVMWrap(object):

    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        self.X = X
        self.Y_ = Y_
        self.model = svm.SVC(C=param_svm_c, gamma=param_svm_gamma, probability=True)
        self.model.fit(X, Y_)
        self.support = self.model.support_

    def predict(self, X):
        return self.model.predict(X)

    def get_scores(self, X):
        return self.model.predict_proba(X)


if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    #np.random.seed(100)

    # instanciraj podatke X i labele Yoh_
    #X, Y_ = sample_gauss(2, 100)
    X, Y_ = sample_gmm(6, 2, 10)

    ksvm = KSVMWrap(X, Y_)

    # nauči parametre:
    # dohvati vjerojatnosti na skupu za učenje
    probs = ksvm.get_scores(X)
    Y = ksvm.predict(X)


    # ispiši performansu (preciznost i odziv po razredima)
    accuracy, pr, M = eval_perf_multi(Y, Y_)
    print("Accuracy SVM: %f" % accuracy)

    # iscrtaj rezultate, decizijsku plohu
    rect = (np.min(X, axis=0), np.max(X, axis=0))

    graph_surface(lambda x: ksvm.predict(x), rect, offset=0)
    graph_data(X, Y_, Y, special=ksvm.support)

    plt.show()

    tflr = TFDeep([2,10, 10, 2], 0.1, 1e-4, activation=tf.nn.relu)
    Yoh_ = class_to_onehot(Y_)
    tflr.train(X, Yoh_, 10000)
    probs = tflr.eval(X)

    Y = np.argmax(probs, axis=1)
    accuracy, pr, M = eval_perf_multi(Y, Y_)
    print("Accuracy Deep: %f" % accuracy)
