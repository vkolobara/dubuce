import os
import pickle
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import matplotlib.pyplot as plt

import skimage as ski
import skimage.io

import time
import math

from sklearn.metrics import confusion_matrix


config = {}
config['max_epochs'] = 35
config['batch_size'] = 50
config['weight_decay'] = 1e-4
config['lr_policy'] = {1: {'lr': 1e-1}, 3: {'lr': 1e-2}, 5: {'lr': 1e-3}, 7: {'lr': 1e-4}}


def draw_image(img, mean, std):
    img *= std
    img += mean
    img = img.astype(np.uint8)
    ski.io.imshow(img)
    ski.io.show()

def build_model(inputs, labels, num_classes):
    weight_decay = config["weight_decay"]

    with tf.contrib.framework.arg_scope([layers.convolution2d],
                                        kernel_size=5, stride=1, padding='SAME', activation_fn=tf.nn.relu,
                                        weights_initializer=layers.variance_scaling_initializer(),
                                        weights_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.convolution2d(inputs, 16, scope='conv1')
        # ostatak konvolucijskih i pooling slojeva
        net = layers.max_pool2d(net, 3, 2, scope='pool1')
        net = layers.convolution2d(net, 32, scope='conv2')
        net = layers.max_pool2d(net, 3, 2, scope='pool2')

    with tf.contrib.framework.arg_scope([layers.fully_connected],
                                        activation_fn=tf.nn.relu,
                                        weights_initializer=layers.variance_scaling_initializer(),
                                        weights_regularizer=layers.l2_regularizer(weight_decay)):
        # sada definiramo potpuno povezane slojeve
        # ali najprije prebacimo 4D tenzor u matricu
        net = layers.flatten(net)
        net = layers.fully_connected(net, 256, scope='fc1')
        net = layers.fully_connected(net, 128, scope='fc2')

    logits = layers.fully_connected(net, num_classes, activation_fn=None, scope='logits')
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    return logits, loss


def shuffle_data(data_x, data_y):
    indices = np.arange(data_x.shape[0])
    np.random.shuffle(indices)
    shuffled_data_x = np.ascontiguousarray(data_x[indices])
    shuffled_data_y = np.ascontiguousarray(data_y[indices])
    return shuffled_data_x, shuffled_data_y


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

DATA_DIR = 'cifar10'

img_height = 32
img_width = 32
num_channels = 3
num_classes = 10

train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32)
train_y = []
for i in range(1, 6):
    subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
    train_x = np.vstack((train_x, subset['data']))
    train_y += subset['labels']

subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
train_x = np.vstack((train_x, subset['data']))
train_y += subset['labels']

train_x = train_x.reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1)
train_y = np.array(train_y, dtype=np.int32)


train_x, train_y_ = (train_x, train_y)
train_y = np.zeros(shape=(len(train_y_), num_classes))
train_y[np.arange(len(train_y_)), train_y_] = 1

data_mean = train_x.mean((0, 1, 2))
data_std = train_x.std((0, 1, 2))

train_x = (train_x - data_mean) / data_std

X = tf.placeholder(tf.float32, [None, 32, 32, 3], name="X")
Y = tf.placeholder(tf.float32, [None, 10], name="Y")
(logits, loss) = build_model(X, Y, 10)

sess = tf.Session()
tf.train.Saver().restore(sess, 'models/cifar.ckpt')

(loss_val, logits_val) = sess.run([loss, logits], feed_dict={X: [train_x[0]], Y: [train_y[0]]})

train_err = []

for i in range(len(train_x)):
    loss_val = sess.run(loss, feed_dict={X: [train_x[i]], Y: [train_y[i]]})
    train_err.append(loss_val)


train_err = np.array(train_err)

#20 worst indice
train_ix = np.argpartition(train_err, -20)[-20:]


for i in range(len(train_ix)):
    ind = train_ix[i]
    logits_val = sess.run(logits, feed_dict={X: [train_x[ind]], Y: [train_y[ind]]})
    print(logits_val)
    best_classes = np.argpartition(logits_val, -3)[-3:]

    print(best_classes)
    print(np.argmax(train_y[ind]))
    print()
    draw_image(train_x[ind], data_mean, data_std)



