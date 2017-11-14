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
    ski.io.imsave(os.path.join(save_dir, filename), img)


def draw_image(img, mean, std):
    img *= std
    img += mean
    img = img.astype(np.uint8)
    ski.io.imshow(img)
    ski.io.show()


def plot_training_progress(save_dir, data):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))

    linewidth = 2
    legend_size = 10
    train_color = 'm'
    val_color = 'c'

    num_points = len(data['train_loss'])
    x_data = np.linspace(1, num_points, num_points)
    ax1.set_title('Cross-entropy loss')
    ax1.plot(x_data, data['train_loss'], marker='o', color=train_color,
             linewidth=linewidth, linestyle='-', label='train')
    ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color,
             linewidth=linewidth, linestyle='-', label='validation')
    ax1.legend(loc='upper right', fontsize=legend_size)
    ax2.set_title('Average class accuracy')
    ax2.plot(x_data, data['train_acc'], marker='o', color=train_color,
             linewidth=linewidth, linestyle='-', label='train')
    ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color,
             linewidth=linewidth, linestyle='-', label='validation')
    ax2.legend(loc='upper left', fontsize=legend_size)
    ax3.set_title('Learning rate')
    ax3.plot(x_data, data['lr'], marker='o', color=train_color,
             linewidth=linewidth, linestyle='-', label='learning_rate')
    ax3.legend(loc='upper left', fontsize=legend_size)

    save_path = os.path.join(save_dir, 'training_plot.pdf')
    print('Plotting in: ', save_path)
    plt.savefig(save_path)


def evaluate(logits, loss, x, y):
    num_examples = x.shape[0]
    assert num_examples % batch_size == 0
    num_batches = num_examples // batch_size
    cnt_correct = 0
    loss_avg = 0
    cm = np.zeros((10, 10))
    for i in range(num_batches):
        batch_x = x[i * batch_size:(i + 1) * batch_size, :]
        batch_y = y[i * batch_size:(i + 1) * batch_size, :]
        (loss_val, logits_val) = sess.run([loss, logits], feed_dict={X: batch_x, Y: batch_y})
        cm += confusion_matrix(np.argmax(batch_y, 1), np.argmax(logits_val, 1), labels=[0,1,2,3,4,5,6,7,8,9])
        yp = np.argmax(logits_val, 1)
        yt = np.argmax(batch_y, 1)
        cnt_correct += (yp == yt).sum()
        loss_avg += loss_val
        # print("step %d / %d, loss = %.2f" % (i*batch_size, num_examples, loss_val / batch_size))
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    valid_acc = cnt_correct / num_examples * 100
    loss_avg /= num_batches

    print(" accuracy = %.2f" % valid_acc)
    print(" avg loss = %.2f\n" % loss_avg)
    return (loss_avg, valid_acc)


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
train_x = train_x.reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1)
train_y = np.array(train_y, dtype=np.int32)

subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))

test_x = subset['data'].reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1).astype(np.float32)
test_y_ = np.array(subset['labels'], dtype=np.int32)
test_y = np.zeros(shape=(len(test_y_), num_classes))
test_y[np.arange(len(test_y_)), test_y_] = 1

valid_size = 5000
train_x, train_y_ = shuffle_data(train_x, train_y)
train_y = np.zeros(shape=(len(train_y_), num_classes))
train_y[np.arange(len(train_y_)), train_y_] = 1

valid_x = train_x[:valid_size, ...]
valid_y = train_y[:valid_size, ...]

train_x = train_x[valid_size:, ...]
train_y = train_y[valid_size:, ...]
data_mean = train_x.mean((0, 1, 2))
data_std = train_x.std((0, 1, 2))

train_x = (train_x - data_mean) / data_std
valid_x = (valid_x - data_mean) / data_std
test_x = (test_x - data_mean) / data_std

plot_data = {}
plot_data['train_loss'] = []
plot_data['valid_loss'] = []
plot_data['train_acc'] = []
plot_data['valid_acc'] = []
plot_data['lr'] = []

num_epochs = config['max_epochs']
batch_size = config['batch_size']

num_batches = train_x.shape[0]//batch_size

X = tf.placeholder(tf.float32, [None, 32, 32, 3], name="X")
Y = tf.placeholder(tf.float32, [None, 10], name="Y")
(logits, loss) = build_model(X, Y, 10)
global_step = tf.Variable(0, trainable=False)
lr=tf.train.exponential_decay(0.01, global_step, num_batches, 0.96)

sess = tf.Session()
op = tf.train.GradientDescentOptimizer(learning_rate=lr)
train_op = op.minimize(loss, global_step=global_step)
sess.run(tf.global_variables_initializer())

SAVE_DIR = 'save_cifar_plots'
save_dir = 'save_cifar_filter'
for epoch_num in range(1, num_epochs + 1):
    train_x, train_y = shuffle_data(train_x, train_y)

    for step in range(num_batches):
        offset = step * batch_size
        # s ovim kodom pazite da je broj primjera djeljiv s batch_size
        batch_x = train_x[offset:(offset + batch_size), ...]
        batch_y = train_y[offset:(offset + batch_size)]
        feed_dict = {X: batch_x, Y: batch_y}
        start_time = time.time()
        run_ops = [train_op, loss, logits]
        ret_val = sess.run(run_ops, feed_dict=feed_dict)
        _, loss_val, logits_val = ret_val
        duration = time.time() - start_time
        if (step + 1) % 50 == 0:
            sec_per_batch = float(duration)
            format_str = 'epoch %d, step %d / %d, loss = %.2f (%.3f sec/batch)'
            print(format_str % (epoch_num, step + 1, num_batches, loss_val, sec_per_batch))
        if step % 100 == 0:
            conv1_var = tf.contrib.framework.get_variables('conv1/weights:0')[0]
            weights = conv1_var.eval(session=sess)
            draw_conv_filters(epoch_num, i * batch_size, weights, save_dir)

    print('Train error:')
    train_loss, train_acc = evaluate(logits, loss, train_x, train_y)
    print('Validation error:')
    valid_loss, valid_acc = evaluate(logits, loss, valid_x, valid_y)
    plot_data['train_loss'] += [train_loss]
    plot_data['valid_loss'] += [valid_loss]
    plot_data['train_acc'] += [train_acc]
    plot_data['valid_acc'] += [valid_acc]
    plot_data['lr'] += [lr.eval(session=sess)]
    plot_training_progress(SAVE_DIR, plot_data)

print("EVALUATE TEST: ")
evaluate(logits, loss, test_x, test_y)

tf.train.Saver().save(sess, "models/cifar.ckpt")