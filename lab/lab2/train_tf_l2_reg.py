import time

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import os
import math

import skimage as ski
import skimage.io

DATA_DIR = 'mnist'
SAVE_DIR = "save_tf_reg"

config = {}
config['max_epochs'] = 8
config['batch_size'] = 50
config['save_dir'] = SAVE_DIR
config['weight_decay'] = 1e-3
config['lr_policy'] = {1: {'lr': 1e-1}, 3: {'lr': 1e-2}, 5: {'lr': 1e-3}, 7: {'lr': 1e-4}}

# np.random.seed(100)
np.random.seed(int(time.time() * 1e6) % 2 ** 31)
dataset = input_data.read_data_sets(DATA_DIR, one_hot=True)
train_x = dataset.train.images
train_x = train_x.reshape([-1, 28, 28, 1])
train_y = dataset.train.labels
valid_x = dataset.validation.images
valid_x = valid_x.reshape([-1, 28, 28, 1])
valid_y = dataset.validation.labels
test_x = dataset.test.images
test_x = test_x.reshape([-1, 28, 28, 1])
test_y = dataset.test.labels
train_mean = train_x.mean()
train_x -= train_mean
valid_x -= train_mean
test_x -= train_mean

import tensorflow.contrib.layers as layers


def build_model(inputs, labels, num_classes):
    weight_decay = config["weight_decay"]
    conv1sz = 16
    fc3sz = 512
    with tf.contrib.framework.arg_scope([layers.convolution2d],
                                        kernel_size=5, stride=1, padding='SAME', activation_fn=tf.nn.relu,
                                        weights_initializer=layers.variance_scaling_initializer(),
                                        weights_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.convolution2d(inputs, conv1sz, scope='conv1')
        # ostatak konvolucijskih i pooling slojeva
        net = layers.max_pool2d(net, 2, 2, scope='pool1')
        net = layers.convolution2d(net, 32, scope='conv2')
        net = layers.max_pool2d(net, 2, 2, scope='pool2')

    with tf.contrib.framework.arg_scope([layers.fully_connected],
                                        activation_fn=tf.nn.relu,
                                        weights_initializer=layers.variance_scaling_initializer(),
                                        weights_regularizer=layers.l2_regularizer(weight_decay)):
        # sada definiramo potpuno povezane slojeve
        # ali najprije prebacimo 4D tenzor u matricu
        net = layers.flatten(net)
        net = layers.fully_connected(net, fc3sz, scope='fc3')

    logits = layers.fully_connected(net, num_classes, activation_fn=None, scope='logits')
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    return logits, loss


sess = tf.Session()
# inputs = np.random.randn(config['batch_size'], 1, 28, 28)

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])
(logits, loss) = build_model(X, Y, 10)


def draw_conv_filters(epoch, step, weights, save_dir):
    w = weights.copy()
    num_filters = w.shape[3]
    num_channels = w.shape[2]
    k = w.shape[0]
    assert w.shape[0] == w.shape[1]
    w = w.reshape(k, k, num_channels, num_filters)
    w -= w.min()
    w /= w.max()
    border = 1
    cols = 8
    rows = math.ceil(num_filters / cols)
    width = cols * k + (cols - 1) * border
    height = rows * k + (rows - 1) * border
    img = np.zeros([height, width, num_channels])
    for i in range(num_filters):
        r = int(i / cols) * (k + border)
        c = int(i % cols) * (k + border)
        img[r:r + k, c:c + k, :] = w[:, :, :, i]
    filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
    print(img.shape)
    ski.io.imsave(os.path.join(save_dir, filename), img[:,:,0])


def train(train_x, train_y, valid_x, valid_y, optimizer, logits, loss, config, sess):
    lr_policy = config['lr_policy']
    batch_size = config['batch_size']
    max_epochs = config['max_epochs']
    save_dir = config['save_dir']
    num_examples = train_x.shape[0]
    assert num_examples % batch_size == 0
    num_batches = num_examples // batch_size
    for epoch in range(1, max_epochs + 1):
        cnt_correct = 0
        # for i in range(num_batches):
        # shuffle the data at the beggining of each epoch
        permutation_idx = np.random.permutation(num_examples)
        train_x = train_x[permutation_idx]
        train_y = train_y[permutation_idx]
        # for i in range(100):
        for i in range(num_batches):
            # store mini-batch to ndarray
            batch_x = train_x[i * batch_size:(i + 1) * batch_size, :]
            batch_y = train_y[i * batch_size:(i + 1) * batch_size, :]
            (loss_val, _, logits_val) = sess.run([loss, optimizer, logits], feed_dict={X: batch_x, Y: batch_y})
            yp = np.argmax(logits_val, 1)
            yt = np.argmax(batch_y, 1)
            cnt_correct += (yp == yt).sum()

            if i % 5 == 0:
                print("epoch %d, step %d/%d, batch loss = %.2f" % (epoch, i * batch_size, num_examples, loss_val))
            if i % 100 == 0:
                conv1_var = tf.contrib.framework.get_variables('conv1/weights:0')[0]
                weights = conv1_var.eval(session=sess)
                draw_conv_filters(epoch, i * batch_size, weights, save_dir)
            if i > 0 and i % 50 == 0:
                print("Train accuracy = %.2f" % (cnt_correct / ((i + 1) * batch_size) * 100))
        print("Train accuracy = %.2f" % (cnt_correct / num_examples * 100))
        evaluate("Validation", valid_x, valid_y, loss, logits, config)
    return (loss, optimizer)


def evaluate(name, x, y, loss, logits, config):
    print("\nRunning evaluation: ", name)
    batch_size = config['batch_size']
    num_examples = x.shape[0]
    assert num_examples % batch_size == 0
    num_batches = num_examples // batch_size
    cnt_correct = 0
    loss_avg = 0
    for i in range(num_batches):
        batch_x = x[i * batch_size:(i + 1) * batch_size, :]
        batch_y = y[i * batch_size:(i + 1) * batch_size, :]
        (loss_val, logits_val) = sess.run([loss, logits], feed_dict={X: batch_x, Y: batch_y})
        yp = np.argmax(logits_val, 1)
        yt = np.argmax(batch_y, 1)
        cnt_correct += (yp == yt).sum()
        loss_avg += loss_val
        # print("step %d / %d, loss = %.2f" % (i*batch_size, num_examples, loss_val / batch_size))
    valid_acc = cnt_correct / num_examples * 100
    loss_avg /= num_batches
    print(name + " accuracy = %.2f" % valid_acc)
    print(name + " avg loss = %.2f\n" % loss_avg)


optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
sess.run(tf.global_variables_initializer())
train(train_x, train_y, valid_x, valid_y, optimizer, logits, loss, config, sess)
evaluate("Test", test_x, test_y, loss, logits, config)
