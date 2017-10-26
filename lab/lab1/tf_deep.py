import tensorflow as tf
import numpy as np
from lab.lab1.data import *
from sklearn.utils import shuffle

class TFDeep:
  def __init__(self, layers, param_delta=0.5, param_lambda=1e-3, activation=tf.nn.relu):
    """Arguments:
       - layers: List of layer sizes
       - param_delta: training step
    """

    # definicija podataka i parametara:
    # definirati self.X, self.Yoh_, self.W, self.b
    # ...

    self.X = tf.placeholder(dtype=tf.float32, shape=[None, layers[0]], name="X")
    self.Y = tf.placeholder(dtype=tf.float32, shape=[None, layers[-1]], name="Y")

    self.w = []
    self.b = []

    activations = [activation] * (len(layers) - 2)
    activations += [tf.nn.softmax]

    for i in range(1, len(layers)):
        self.w.append(tf.get_variable("W_%d" % i, shape=(layers[i - 1], layers[i]), initializer=tf.random_normal_initializer(stddev=0.1)))
        self.b.append(tf.get_variable("b_%d" % i, shape=(layers[i],), initializer=tf.random_normal_initializer(stddev=0.1)))

    hs = [self.X]

    for i in range(len(layers)-1):
        hs.append(activations[i](tf.matmul(hs[i], self.w[i]) + self.b[i]))


    # formulacija modela: izračunati self.probs
    #   koristiti: tf.matmul, tf.nn.softmax
    # ...

    self.probs = hs[-1]

    # formulacija gubitka: self.loss
    #   koristiti: tf.log, tf.reduce_sum, tf.reduce_mean
    # ...

    self.reg = 0
    for w in self.w:
        self.reg += param_lambda * tf.nn.l2_loss(w)

    self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.probs)) \
                + self.reg

    # formulacija operacije učenja: self.train_step
    #   koristiti: tf.train.GradientDescentOptimizer,
    #              tf.train.GradientDescentOptimizer.minimize
    # ...

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(param_delta, global_step=global_step, decay_steps=1, decay_rate=1-1e-4)

    self.train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss, global_step=global_step)

    # instanciranje izvedbenog konteksta: self.session
    #   koristiti: tf.Session
    # ...
    self.session = tf.Session()

  def count_params(self):
      variables = tf.trainable_variables()
      var_names = [v.name for v in variables]
      count = 0

      for v in variables:
          count += v.shape.num_elements()

      return (var_names, count)

  def train(self, X, Yoh_, param_niter):
    """Arguments:
       - X: actual datapoints [NxD]
       - Yoh_: one-hot encoded labels [NxC]
       - param_niter: number of iterations
    """
    # incijalizacija parametara
    #   koristiti: tf.initialize_all_variables
    # ...
    initializer = tf.global_variables_initializer()

    # optimizacijska petlja
    #   koristiti: tf.Session.run
    # ...

    self.session.run(initializer)
    for i in range(param_niter):
        _, l = self.session.run([self.train_step, self.loss], feed_dict={self.X: X, self.Y: Yoh_})
        if i % 1000 == 0:
            print("Loss at %d: %f" % (i, l))

  def train_mb(self, X, Yoh_, param_niter, n_batches):
    initializer = tf.global_variables_initializer()
    self.session.run(initializer)
    for i in range(param_niter):
        X_shuffle, Y_shuffle = shuffle(X, Yoh_)
        batch_X = np.split(X_shuffle, n_batches)
        batch_Y = np.split(Y_shuffle, n_batches)
        for j in range(n_batches):
            _, l = self.session.run([self.train_step, self.loss], feed_dict={self.X: batch_X[j], self.Y: batch_Y[j]})
        if i % 1000 == 0:
            print("Loss at %d: %f" % (i, l))

  def eval(self, X):
    """Arguments:
       - X: actual datapoints [NxD]
       Returns: predicted class probabilites [NxC]
    """
    #   koristiti: tf.Session.run
    return self.session.run(self.probs, feed_dict={self.X: X})


def test(X, Y_, configuration, param_niter=10000, param_delta=0.1, param_lambda=1e-3, activation = tf.nn.relu):
    print("Configuration: %s" % str(configuration))
    Yoh_ = class_to_onehot(Y_)

    # izgradi graf:
    tflr = TFDeep(configuration, param_delta, param_lambda, activation=activation)

    # nauči parametre:
    tflr.train(X, Yoh_, param_niter)

    # dohvati vjerojatnosti na skupu za učenje
    probs = tflr.eval(X)
    Y = np.argmax(probs, axis=1)

    accuracy, pr, M = eval_perf_multi(Y, Y_)

    print("Accuracy: %f" % accuracy)
    # iscrtaj rezultate, decizijsku plohu
    rect = (np.min(X, axis=0), np.max(X, axis=0))

    graph_surface(lambda x: np.argmax(tflr.eval(x), axis=1), rect, offset=0)
    graph_data(X, Y_, Y, special=[])

if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva

    # instanciraj podatke X i labele Yoh_
    #X, Y_ = sample_gauss(3, 100)
    #X, Y_ = sample_gmm(6, 2, 10)

    np.random.seed(100)
    #tf.set_random_seed(100)
    #(X1, Y_1) = sample_gmm(4, 2, 40)
    (X2, Y_2) = sample_gmm(6, 2, 10)

    np.random.seed()

    test(X2, Y_2, [2, 10, 10, 2], param_niter=10000, param_delta=0.01, param_lambda=1e-4, activation=tf.nn.relu)
    plt.show()

    '''
    configs = [[2, 2], [2, 10, 2], [2, 10, 10, 2]]

    for config in configs:
        with tf.variable_scope(str(configs.index(config)) + "DATA_1"):
            plt.figure()
            print("DATA_1")
            test(X1, Y_1, config)
        with tf.variable_scope(str(configs.index(config)) + "DATA_2"):
            plt.figure()
            print("DATA_2")
            test(X2, Y_2, config)

    plt.show()
    '''