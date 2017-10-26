import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

## 1. definicija računskog grafa
# podatci i parametri
X = tf.placeholder(tf.float32, [None])
Y_ = tf.placeholder(tf.float32, [None])
a = tf.Variable(0.0)
b = tf.Variable(0.0)

# afini regresijski model
Y = a * X + b

# kvadratni gubitak
loss = (Y - Y_) ** 2

# optimizacijski postupak: gradijentni spust
trainer = tf.train.GradientDescentOptimizer(0.000000002)


# izracunaj gradijente sa tensorflowom
grads = trainer.compute_gradients(loss, [a, b])
train_op = trainer.apply_gradients(grads)

# izracunaj gradijente rucno
dB = tf.reduce_sum(2 * (Y - Y_))
dA = tf.reduce_sum(2 * (Y - Y_) * X)


# train_op = trainer.minimize(loss)


## 2. inicijalizacija parametara
sess = tf.Session()
sess.run(tf.global_variables_initializer())


## 3. učenje
# neka igre počnu!

X_train = range(1, 1000)
Y_train = [x*2.0 + 1.0 for x in X_train]

for i in range(1000):
    val_loss, _, val_a, val_b = sess.run([loss, train_op, a, b], feed_dict={X: X_train, Y_: Y_train})
    print(i, val_loss, val_a, val_b)
    p1 = tf.Print(loss, [grads], message="Gradijenti tensorflow: ")
    p2 = tf.Print(loss, [dA, dB], message="Gradijenti rucno: ")
    sess.run([p1, p2], feed_dict={X: X_train, Y_: Y_train})
    print("")