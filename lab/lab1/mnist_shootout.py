import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.app.flags.DEFINE_string('data_dir', 'mnist_data', "Directory for storing data")
mnist = input_data.read_data_sets(tf.app.flags.FLAGS.data_dir, one_hot=True)
import numpy as np
from lab.lab1.data import *
from lab.lab1.ksvm_wrap import *

N=mnist.train.images.shape[0]
D=mnist.train.images.shape[1]
C=mnist.train.labels.shape[1]

import lab.lab1.tf_deep as tf_deep

#tflr = tf_deep.TFDeep([784, 10], 1e-2, 1e-3)

#tflr.train_mb(mnist.train.images, mnist.train.labels, 100, 10)

#tf_deep.test(mnist.train.images, np.argmax(mnist.train.labels, axis=1), [784, 10])
#tf_deep.test(mnist.train.images, mnist.train.labels, [784, 100, 10])

Y_ = np.argmax(mnist.train.labels, axis=1)

ksvm = KSVMWrap(mnist.train.images, Y_)

# nauči parametre:
# dohvati vjerojatnosti na skupu za učenje
probs = ksvm.get_scores(X)
Y = ksvm.predict(X)

# ispiši performansu (preciznost i odziv po razredima)
accuracy, pr, M = eval_perf_multi(Y, Y_)
print("Accuracy SVM: %f" % accuracy)

"""
probs = tflr.eval(mnist.train.images)
Y = np.argmax(probs, axis=1)

# ispiši performansu (preciznost i odziv po razredima)
accuracy, pr, M = eval_perf_multi(Y, np.argmax(mnist.train.labels, axis=1))
print("Train: %f" % accuracy)

accuracy, pr, M = eval_perf_multi(np.argmax(tflr.eval(mnist.test.images), axis=1), np.argmax(mnist.test.labels, axis=1))
print("Test: %f" % accuracy)
"""
