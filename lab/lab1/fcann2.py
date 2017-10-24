import numpy as np
from lab.lab1.data import *
import sklearn.datasets
from sklearn.metrics import log_loss
from scipy.special import expit

def fcann2_classify(x, w, b):
    (_, _, _, p) = forward_pass(x, w, b)
    return p[:, 1]


def forward_pass(x, w, b):
    (w1, w2) = w
    (b1, b2) = b
    s1 = x.dot(w1) + b1
    h1 = relu(s1)
    s2 = h1.dot(w2) + b2
    p = softmax(s2)
    return s1, h1, s2, p


def fcann2_train(x, y, num_hidden=5):
    D = len(x[0])
    N = len(x)

    w1 = np.random.randn(D, num_hidden)
    b1 = np.random.randn(1, num_hidden)

    w2 = np.random.randn(num_hidden, 2)
    b2 = np.random.randn(1, 2)
    Yoh_ = class_to_onehot(y)

    for i in range(100000):
        (s1, h1, s2, p) = forward_pass(x, (w1, w2), (b1, b2))

        reg = np.sum(np.linalg.norm(w1))
        reg += np.sum(np.linalg.norm(w2))
        probs = np.clip(p, 1e-15, 1 - 1e-15)
        corect_logprobs = -np.log(probs[range(N), y])
        data_loss = np.sum(corect_logprobs)
        # Add regulatization term to loss (optional)
        data_loss += param_lambda / 2 * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
        loss = 1. / N * data_loss

        print(loss)

        d_L = p - Yoh_

        d_w2 = h1.T.dot(d_L)
        d_b2 = np.sum(d_L, axis=0)

        d_L_h1 = d_L.dot(w2.T) * 1.0*(h1>0)
        d_w1 = np.dot(X.T, d_L_h1)
        d_b1 = np.sum(d_L_h1, axis=0)

        d_w2 += param_lambda * w2
        d_w1 += param_lambda * w1

        w1 += -param_delta * d_w1
        w2 += -param_delta * d_w2
        b1 += -param_delta * d_b1
        b2 += -param_delta * d_b2

    return (w1, w2), (b1, b2)


def softmax(x):
    exp_x_shifted = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs = exp_x_shifted / np.sum(exp_x_shifted, axis=1, keepdims=True)
    return probs

def relu(x):
    return x * (x > 0)


def logloss(true_label, predicted, eps=1e-15):
  p = np.clip(predicted, eps, 1 - eps)
  if true_label == 1:
    return -np.log(p)
  else:
    return -np.log(1 - p)

np.random.seed(100)
X, Y_ = sample_gmm(6, 2, 10)
np.random.seed()

#X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#Y_ = np.array([0, 1, 1, 0])

param_niter = int(1e3)
param_delta = 0.001
param_lambda = 0.00001

w, b = fcann2_train(X, Y_)
# get the class predictions
print(fcann2_classify(X, w, b))
Y = np.array(fcann2_classify(X, w, b) >= 0.5, dtype=int)

# graph the decision surface
rect = (np.min(X, axis=0), np.max(X, axis=0))
graph_surface(lambda x: fcann2_classify(x, w, b), rect, offset=0)
accuracy, pr, M = eval_perf_multi(Y, Y_)

print(accuracy)

# graph the data points
graph_data(X, Y_, Y, special=[])
plt.show()
print(Y_)
print(Y)